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

# ── 하이퍼파라미터 ──────────────────────────────────────────────────────────
START_DATE = "2023-03-30"
END_DATE   = "2023-10-01"
EPOCHS     = 20
BATCH_SIZE = 1024
LR         = 0.001


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


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("=" * 60)
    print("SmartPitch 범용 전이 모델 학습")
    print(f"  데이터: {START_DATE} ~ {END_DATE} (2023 MLB 전체 시즌)")
    print(f"  epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")
    print("=" * 60)

    run = wandb.init(
        project="SmartPitch-Portfolio",
        name="Universal_MLP_2023",
        config={
            "data": "2023 MLB Full Season",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "hidden_dims": [128, 64],
            "dropout_rate": 0.2,
        }
    )

    try:
        # ── 1. 전체 시즌 데이터 수집 ─────────────────────────────────────────
        print(f"\n[단계 1] 전체 Statcast 데이터 수집 ({START_DATE} ~ {END_DATE})")
        raw_df = statcast(start_dt=START_DATE, end_dt=END_DATE)
        if raw_df.empty:
            raise ValueError("데이터 수집 실패. 날짜 범위를 확인하세요.")
        print(f"  수집 완료: {len(raw_df):,}건")

        # ── 2. 전처리 ────────────────────────────────────────────────────────
        print("\n[단계 2] raw statcast 데이터 전처리")
        df_processed = _preprocess_raw(raw_df)
        del raw_df  # 원본 72만 건 메모리 즉시 해제

        # ── 3. 모델 학습 ─────────────────────────────────────────────────────
        print(f"\n[단계 3] TransitionProbabilityModel 학습")
        model = TransitionProbabilityModel(df=df_processed, batch_size=BATCH_SIZE, lr=LR)
        model.run_modeling_pipeline(epochs=EPOCHS, upload_artifact=False)
        del df_processed  # One-Hot 배열 메모리 해제

        # ── 4. 범용 경로로 모델 이동, 메타데이터 저장 ───────────────────────
        print("\n[단계 4] 모델 및 메타데이터 저장")
        shutil.move(MODEL_TMP_PATH, MODEL_SAVE_PATH)
        print(f"  가중치 저장: {MODEL_SAVE_PATH}")

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(FEATURE_COLS_PATH, 'w', encoding='utf-8') as f:
            json.dump(model.feature_columns, f, ensure_ascii=False, indent=2)
        print(f"  feature_columns 저장: {FEATURE_COLS_PATH}")

        with open(TARGET_CLS_PATH, 'w', encoding='utf-8') as f:
            json.dump(model.target_classes, f, ensure_ascii=False, indent=2)
        print(f"  target_classes 저장: {TARGET_CLS_PATH}")

        # ── 5. W&B Artifact 업로드 ───────────────────────────────────────────
        print("\n[단계 5] W&B Artifact 업로드 (universal_transition_mlp)")
        artifact = wandb.Artifact(name="universal_transition_mlp", type="model")
        artifact.add_file(MODEL_SAVE_PATH)
        artifact.add_file(FEATURE_COLS_PATH)
        artifact.add_file(TARGET_CLS_PATH)
        wandb.log_artifact(artifact)
        print("  업로드 완료")

        print("\n" + "=" * 60)
        print("범용 전이 모델 학습 완료!")
        print(f"  모델:      {MODEL_SAVE_PATH}")
        print(f"  컬럼 목록: {FEATURE_COLS_PATH}")
        print(f"  클래스:    {TARGET_CLS_PATH}")
        print("  main.py에서 USE_UNIVERSAL_MODEL=True로 설정하면 이 모델을 사용합니다.")
        print("=" * 60)

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
