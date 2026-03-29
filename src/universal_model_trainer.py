"""
universal_model_trainer.py — 2023 MLB 전체 데이터 기반 범용 전이 확률 모델 학습

역할:
    단일 투수 데이터(~수천 건)가 아닌 2023 시즌 MLB 전체 Statcast 데이터(~72만 건)를
    사용해 TransitionProbabilityModel을 학습합니다. 결과물인 범용 전이 모델은
    main.py의 USE_UNIVERSAL_MODEL=True 플래그로 로드하여 사용합니다.

    단일 투수 모델(val_acc ~47%)과 달리, 전체 MLB 데이터를 학습하므로
    val_acc 65% 이상이 목표입니다.

선행 조건:
    uv run src/batter_clustering.py   → data/batter_clusters_2023.csv (타자 군집, K=8)
    uv run src/pitcher_clustering.py  → data/pitcher_clusters_2023.csv (투수 군집, K=4)
    두 CSV가 없으면 모든 군집 ID가 "0"으로 fallback되어 정확도 저하

산출물:
    best_transition_model_universal.pth     학습된 MLP 가중치 (gitignore 대상 *.pth)
    data/feature_columns_universal.json     입력 특징 컬럼 순서 (MDP/RL 연동 필수)
    data/target_classes_universal.json      투구 결과 레이블 목록

    → W&B Artifact: project=SmartPitch-Portfolio, name=universal_transition_mlp
    → .pth 파일은 git으로 추적하지 않음. W&B Artifact 또는 로컬 재실행으로 복구.

실행 방법:
    uv run src/universal_model_trainer.py
    → 최초 실행 시 약 20~40분 (pybaseball ~10분 + 학습 ~20분)
    → 재실행 시 pybaseball 캐시 활용으로 데이터 수집 약 1분

메모리 주의:
    One-Hot 인코딩 후 약 1GB RAM 사용 (72만 건 × ~350 컬럼)
    8GB 이상 RAM 환경을 권장합니다.
"""
import os
import sys
import json
import shutil

import pandas as pd
import wandb
from pybaseball import statcast, cache

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import TransitionProbabilityModel

cache.enable()  # 동일 기간 재요청 시 로컬 캐시 사용

# ── Statcast pitch_type 코드 → 구종명 매핑 ─────────────────────────────────
# clustering.py의 구종명 규칙과 동일하게 맞춤
# (Fastball / Slider / Changeup / Curveball / Cutter / Sinker / Sweeper / Splitter / Knuckleball)
PITCH_TYPE_MAP = {
    'FF': 'Fastball',
    'FA': 'Fastball',
    'SI': 'Sinker',
    'FC': 'Cutter',
    'SL': 'Slider',
    'ST': 'Sweeper',
    'SV': 'Slider',
    'CH': 'Changeup',
    'EP': 'Changeup',
    'SC': 'Changeup',
    'FS': 'Splitter',
    'FO': 'Splitter',
    'CU': 'Curveball',
    'KC': 'Curveball',
    'CS': 'Curveball',
    'KN': 'Knuckleball',
    'PO': None,  # pitchout — 전술구, 학습 데이터에서 제외
}

# ── description(투구 결과) 12클래스 → 5그룹 병합 매핑 ──────────────────────
# pitch_env.py _apply_outcome()의 처리 그룹과 1:1 대응
# MDP/DQN이 실제로 구분하는 단위와 모델 출력을 정합시킴
DESCRIPTION_MAP = {
    # strike 그룹: 스트라이크 판정 (삼진 가능)
    "called_strike":           "strike",
    "swinging_strike":         "strike",
    "foul_tip":                "strike",
    "swinging_strike_blocked": "strike",
    "missed_bunt":             "strike",
    "bunt_foul_tip":           "strike",  # 기존 버그: pitch_env/mdp_solver에서 else 처리됐음

    # foul 그룹: 파울 (2스트라이크 이전만 스트라이크 추가)
    "foul":                    "foul",
    "foul_bunt":               "foul",

    # ball 그룹: 볼 (볼넷 가능)
    "ball":                    "ball",
    "blocked_ball":            "ball",    # 기존 버그: pitch_env/mdp_solver에서 else 처리됐음

    # 나머지: 변경 없음
    "hit_by_pitch":            "hit_by_pitch",
    "hit_into_play":           "hit_into_play",
}

# ── 파일 경로 ───────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_BASE, '..')
DATA_DIR = os.path.join(_ROOT, 'data')
MODEL_TMP_PATH    = os.path.join(_ROOT, 'best_transition_model.pth')           # train_model()이 임시 저장
MODEL_SAVE_PATH   = os.path.join(_ROOT, 'best_transition_model_universal.pth') # 최종 저장 위치
FEATURE_COLS_PATH = os.path.join(DATA_DIR, 'feature_columns_universal.json')
TARGET_CLS_PATH   = os.path.join(DATA_DIR, 'target_classes_universal.json')
MODEL_CONFIG_PATH = os.path.join(DATA_DIR, 'model_config_universal.json')      # 아키텍처 설정(hidden_dims 등)

# ── 하이퍼파라미터 ──────────────────────────────────────────────────────────
START_DATE = "2023-03-30"
END_DATE   = "2023-10-01"
EPOCHS     = 20
BATCH_SIZE = 1024
LR         = 0.001

# ── 실험 설정 ────────────────────────────────────────────────────────────────
# 각 실험은 독립된 W&B run으로 기록됩니다.
# Baseline: hidden_dims=[128,64], no scheduler, no feature engineering (이전 run 참고)
EXPERIMENTS = [
    {
        "run_name":                "Universal_MLP_Exp1_BiggerModel",
        "description":             "모델 크기 확장: [256, 128, 64]",
        "hidden_dims":             [256, 128, 64],
        "use_lr_scheduler":        False,
        "use_class_weights":       False,
        "use_feature_engineering": False,
    },
    {
        "run_name":                "Universal_MLP_Exp2_LRScheduler",
        "description":             "ReduceLROnPlateau 스케줄러 추가 (factor=0.5, patience=3)",
        "hidden_dims":             [128, 64],
        "use_lr_scheduler":        True,
        "use_class_weights":       False,
        "use_feature_engineering": False,
    },
    {
        "run_name":                "Universal_MLP_Exp3_ClassWeights",
        "description":             "클래스 가중 CrossEntropyLoss: 소수 클래스(foul, hit_into_play) recall 개선",
        "hidden_dims":             [128, 64],
        "use_lr_scheduler":        False,
        "use_class_weights":       True,
        "use_feature_engineering": False,
    },
]


def _preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    [내부 함수] raw statcast() 결과물을 TransitionProbabilityModel 입력 형식으로 변환

    수행 작업:
      1. pitch_type 코드 → mapped_pitch_name 문자열 (PITCH_TYPE_MAP 사용)
      2. on_1b/2b/3b float/NaN → "0"/"1" 문자열
         (model.py _prepare_data()의 count_state 문자열 연결 요구사항)
      3. description 12클래스 → 5그룹 병합 (DESCRIPTION_MAP 사용)
         pitch_env.py / mdp_solver.py와 정합 맞춤
      4. 불필요한 레코드 제거 (pitchout, description/zone NaN 등)
      5. 필요 컬럼만 추출

    :param df: statcast() 반환 DataFrame
    :return:   전처리 완료 DataFrame
    """
    print(f"  전처리 시작 (원본 {len(df):,}건)...")
    df = df.copy()

    # 1. pitch_type → mapped_pitch_name
    df['mapped_pitch_name'] = df['pitch_type'].map(PITCH_TYPE_MAP)
    before = len(df)
    df = df[df['mapped_pitch_name'].notna()]
    print(f"  pitch_type 매핑 후: {len(df):,}건 (제거: {before - len(df):,}건)")

    # 2. on_1b/2b/3b float/NaN → "0"/"1" 문자열
    for col in ['on_1b', 'on_2b', 'on_3b']:
        df[col] = df[col].notna().astype(int).astype(str)

    # 3. description 12클래스 → 5그룹 병합
    df['description'] = df['description'].map(DESCRIPTION_MAP)
    before = len(df)
    df = df[df['description'].notna()]  # 매핑 안 된 희귀 결과 코드 제거
    if before - len(df) > 0:
        print(f"  description 매핑 후: {len(df):,}건 (매핑 없음 제거: {before - len(df):,}건)")

    # 4. 필수 컬럼 유지 및 NaN 제거
    required = [
        'balls', 'strikes', 'outs_when_up',
        'on_1b', 'on_2b', 'on_3b',
        'mapped_pitch_name', 'zone', 'description',
        'batter', 'pitcher',
    ]
    df = df[required].dropna(subset=['balls', 'strikes', 'outs_when_up', 'zone', 'description'])
    print(f"  필수 컬럼 필터링 후: {len(df):,}건")

    # zone float → int 변환 (1~14 정수)
    df['zone'] = df['zone'].astype(float).astype(int)

    return df.reset_index(drop=True)


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Exp3용] 기존 피처에서 파생된 수치 피처 5개를 추가합니다.

    추가 피처:
      is_two_strike      : 2스트라이크 여부 (투수가 finishing pitch를 선택하는 상황)
      is_first_pitch     : 첫 투구 여부 (0-0 카운트, 투수가 선제 스트라이크를 노리는 상황)
      balls_minus_strikes: 볼-스트라이크 차이 (양수=타자 카운트, 음수=투수 카운트)
      zone_row           : 존의 세로 위치 (1=상단, 2=중단, 3=하단, 0=스트라이크존 외)
      zone_col           : 존의 가로 위치 (1=안쪽, 2=중앙, 3=바깥쪽, 0=스트라이크존 외)

    zone_row/col는 존의 공간적 구조를 one-hot 손실 없이 수치로 표현합니다.
    Statcast 존 번호 기준:
        행1: 1,2,3 (상단) / 행2: 4,5,6 (중단) / 행3: 7,8,9 (하단)
        열1: 1,4,7 (안쪽) / 열2: 2,5,8 (중앙) / 열3: 3,6,9 (바깥쪽)
        11~14: 섀도우 존(스트라이크존 외) → row=0, col=0
    """
    df = df.copy()
    balls   = df['balls'].astype(int)
    strikes = df['strikes'].astype(int)
    zone    = df['zone'].astype(int)

    df['is_two_strike']       = (strikes == 2).astype(int)
    df['is_first_pitch']      = ((balls == 0) & (strikes == 0)).astype(int)
    df['balls_minus_strikes'] = balls - strikes

    _zone_row = {1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 11:0, 12:0, 13:0, 14:0}
    _zone_col = {1:1, 2:2, 3:3, 4:1, 5:2, 6:3, 7:1, 8:2, 9:3, 11:0, 12:0, 13:0, 14:0}
    df['zone_row'] = zone.map(_zone_row).fillna(0).astype(int)
    df['zone_col'] = zone.map(_zone_col).fillna(0).astype(int)

    print(f"  피처 엔지니어링 완료: is_two_strike, is_first_pitch, balls_minus_strikes, zone_row, zone_col 추가")
    return df


def _run_single_experiment(exp: dict, df_base: pd.DataFrame) -> dict:
    """
    단일 실험을 실행하고 결과를 반환합니다.
    각 실험은 독립된 W&B run으로 기록됩니다.

    :param exp:     EXPERIMENTS 항목 (run_name, hidden_dims, use_lr_scheduler, use_feature_engineering)
    :param df_base: _preprocess_raw() 완료된 기본 DataFrame
    :return:        {"run_name": ..., "best_val_loss": ..., "final_val_acc": ...}
    """
    print(f"\n{'='*60}")
    print(f"실험 시작: {exp['run_name']}")
    print(f"  설명: {exp['description']}")
    print(f"{'='*60}")

    run = wandb.init(
        project="SmartPitch-Portfolio",
        name=exp["run_name"],
        config={
            "data":                    "2023 MLB Full Season",
            "epochs":                  EPOCHS,
            "batch_size":              BATCH_SIZE,
            "learning_rate":           LR,
            "hidden_dims":             exp["hidden_dims"],
            "dropout_rate":            0.2,
            "use_lr_scheduler":        exp["use_lr_scheduler"],
            "use_class_weights":       exp["use_class_weights"],
            "use_feature_engineering": exp["use_feature_engineering"],
            "description":             exp["description"],
        }
    )

    try:
        df = df_base.copy()
        if exp["use_feature_engineering"]:
            df = _add_engineered_features(df)

        model = TransitionProbabilityModel(df=df, batch_size=BATCH_SIZE, lr=LR)
        model.run_modeling_pipeline(
            epochs=EPOCHS,
            hidden_dims=exp["hidden_dims"],
            upload_artifact=False,
            use_lr_scheduler=exp["use_lr_scheduler"],
            use_class_weights=exp["use_class_weights"],
        )

        # train_model()이 wandb.run.summary에 best_val_loss/acc를 세팅함
        # 구버전 run과의 호환을 위해 fallback 키도 허용
        best_val_loss = wandb.run.summary.get("best_val_loss",
                        wandb.run.summary.get("val_loss",      float('nan')))
        final_val_acc = wandb.run.summary.get("best_val_acc",
                        wandb.run.summary.get("val_accuracy",  float('nan')))

        # 실험별 모델 가중치 및 메타데이터를 각자 경로에 보관
        # → main()에서 best 실험의 파일을 universal 경로로 복사
        os.makedirs(DATA_DIR, exist_ok=True)
        exp_model_path = os.path.join(_ROOT, f'best_model_{exp["run_name"]}.pth')
        exp_feat_path  = os.path.join(DATA_DIR, f'feature_columns_{exp["run_name"]}.json')
        exp_cls_path   = os.path.join(DATA_DIR, f'target_classes_{exp["run_name"]}.json')

        exp_cfg_path = os.path.join(DATA_DIR, f'model_config_{exp["run_name"]}.json')

        if os.path.exists(MODEL_TMP_PATH):
            shutil.copy(MODEL_TMP_PATH, exp_model_path)
        with open(exp_feat_path, 'w', encoding='utf-8') as f:
            json.dump(model.feature_columns, f, ensure_ascii=False, indent=2)
        with open(exp_cls_path, 'w', encoding='utf-8') as f:
            json.dump(model.target_classes, f, ensure_ascii=False, indent=2)
        # 아키텍처 설정 저장 — load_from_checkpoint()가 hidden_dims를 정확히 재현하기 위해 필수
        with open(exp_cfg_path, 'w', encoding='utf-8') as f:
            json.dump({"hidden_dims": exp["hidden_dims"], "dropout_rate": 0.2}, f, indent=2)

        return {
            "run_name":       exp["run_name"],
            "description":    exp["description"],
            "best_val_loss":  best_val_loss,
            "final_val_acc":  final_val_acc,
            "model_path":     exp_model_path,
            "feat_path":      exp_feat_path,
            "cls_path":       exp_cls_path,
            "cfg_path":       exp_cfg_path,
        }

    finally:
        wandb.finish()


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("=" * 60)
    print("SmartPitch 범용 전이 모델 개선 실험 (3종)")
    print(f"  데이터: {START_DATE} ~ {END_DATE} (2023 MLB 전체 시즌)")
    print(f"  Baseline: hidden_dims=[128,64], val_acc=57.9%, val_loss=1.0219")
    print("=" * 60)

    # ── 1. 전체 시즌 데이터 수집 (1회, 캐시 재활용) ──────────────────────────
    print(f"\n[공통] 전체 Statcast 데이터 수집 ({START_DATE} ~ {END_DATE})")
    raw_df = statcast(start_dt=START_DATE, end_dt=END_DATE)
    if raw_df.empty:
        raise ValueError("데이터 수집 실패. 날짜 범위를 확인하세요.")
    print(f"  수집 완료: {len(raw_df):,}건")

    # ── 2. 기본 전처리 (모든 실험 공통) ─────────────────────────────────────
    print("\n[공통] raw statcast 데이터 전처리")
    df_base = _preprocess_raw(raw_df)
    del raw_df

    # ── 3. 실험 순차 실행 ────────────────────────────────────────────────────
    results = []
    for exp in EXPERIMENTS:
        results.append(_run_single_experiment(exp, df_base))
    del df_base

    # ── 4. 최고 실험의 모델+메타데이터를 universal 경로로 복사 ──────────────
    # 선정 기준: best_val_acc (val_loss는 class_weights 사용 시 스케일이 달라 비교 불가)
    best_result = max(results, key=lambda r: r["final_val_acc"])
    print(f"\n[저장] 최고 실험: {best_result['run_name']}"
          f" (val_acc={best_result['final_val_acc']:.4f}, val_loss={best_result['best_val_loss']:.4f})")

    shutil.copy(best_result["model_path"], MODEL_SAVE_PATH)
    shutil.copy(best_result["feat_path"],  FEATURE_COLS_PATH)
    shutil.copy(best_result["cls_path"],   TARGET_CLS_PATH)
    shutil.copy(best_result["cfg_path"],   MODEL_CONFIG_PATH)
    print(f"  가중치 저장:          {MODEL_SAVE_PATH}")
    print(f"  feature_columns 저장: {FEATURE_COLS_PATH}")
    print(f"  target_classes 저장:  {TARGET_CLS_PATH}")
    print(f"  model_config 저장:    {MODEL_CONFIG_PATH}")

    # ── 5. W&B Artifact 업로드 ───────────────────────────────────────────────
    _upload_run = wandb.init(
        project="SmartPitch-Portfolio",
        name="Universal_MLP_BestModel_Upload",
        config={"best_experiment": best_result["run_name"]},
    )
    try:
        artifact = wandb.Artifact(name="universal_transition_mlp", type="model")
        artifact.add_file(MODEL_SAVE_PATH)
        artifact.add_file(FEATURE_COLS_PATH)
        artifact.add_file(TARGET_CLS_PATH)
        artifact.add_file(MODEL_CONFIG_PATH)
        wandb.log_artifact(artifact)
        print("  W&B Artifact 업로드 완료: universal_transition_mlp")
    finally:
        wandb.finish()

    # 실험별 임시 파일 정리
    for r in results:
        for fpath in [r["model_path"], r["feat_path"], r["cls_path"], r["cfg_path"]]:
            if os.path.exists(fpath):
                os.remove(fpath)
    if os.path.exists(MODEL_TMP_PATH):
        os.remove(MODEL_TMP_PATH)

    # ── 5. 결과 비교 테이블 출력 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("실험 결과 비교")
    print("=" * 70)
    print(f"{'실험':<40} {'Val Acc':>10} {'Val Loss':>10}  (선정기준: Val Acc↑)")
    print("-" * 70)
    print(f"{'[Baseline] hidden=[128,64], no scheduler':<40} {'57.9%':>10} {'1.0219':>10}")
    for r in results:
        acc_str  = f"{r['final_val_acc']:.1%}"  if r['final_val_acc']  == r['final_val_acc']  else "N/A"
        loss_str = f"{r['best_val_loss']:.4f}"  if r['best_val_loss']  == r['best_val_loss']  else "N/A"
        label    = r['run_name'].replace("Universal_MLP_", "")
        marker   = " ✓" if r is best_result else ""
        print(f"  {label:<38} {acc_str:>10} {loss_str:>10}{marker}")
    print("=" * 70)

    print("\n범용 전이 모델 개선 실험 완료.")
    print("main.py에서 USE_UNIVERSAL_MODEL=True로 설정하면 저장된 모델을 사용합니다.")


if __name__ == "__main__":
    main()
