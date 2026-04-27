"""
main_cease.py — Dylan Cease 개인 맞춤 DQN 학습 파이프라인

Cole 2019와 동일한 방식으로 Cease의 2024-2025 시즌 데이터를 사용하여
개인 맞춤 DQN 모델을 학습합니다.

데이터 범위:
    2024 시즌: 2024-03-28 ~ 2024-10-31
    2025 시즌: 2025-03-27 ~ 2025-07-31

산출물:
    dqn_cease_2024_2025.zip               — 최종 DQN 모델
    best_dqn_model_cease/best_model.zip   — EvalCallback 기준 최고 모델

실행:
    uv run scripts/main_cease.py
"""
import os
import sys
import hashlib

import numpy as np
import pandas as pd
import wandb
from pybaseball import statcast_pitcher, cache

# 프로젝트 루트 등록
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from src.data_loader import PitchDataLoader
from src.clustering import PitchClustering
from src.model import TransitionProbabilityModel
from src.mdp_solver import MDPOptimizer
from src.pitch_env import PitchEnv
from src.rl_trainer import DQNTrainer

cache.enable()

# ── 투수 정보 ────────────────────────────────────────────────────────────
PITCHER_FIRST = "Dylan"
PITCHER_LAST = "Cease"
PITCHER_ID = 656302
PITCHER_CLUSTER = 0  # 파워 FB/SL

# ── 시즌 범위 ────────────────────────────────────────────────────────────
SEASONS = [
    ("2024-03-28", "2024-10-31", "2024"),
    ("2025-03-27", "2025-07-31", "2025"),
]

# ── Cole 모델 해시 (변경 없음 보증용) ────────────────────────────────────
COLE_MODEL_PATH = os.path.join(_ROOT, "best_dqn_model", "best_model.zip")


def _md5(path: str) -> str:
    """파일의 MD5 해시 반환"""
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _print_pitch_distribution(df: pd.DataFrame, label: str):
    """pitch_type 분포 출력"""
    dist = df['pitch_type'].value_counts(normalize=True).round(3)
    print(f"\n  [{label}] pitch_type 분포 ({len(df)} 투구):")
    for pt, pct in dist.items():
        print(f"    {pt:>4s}: {pct*100:5.1f}%")
    return dist


def _check_pitch_shift(dist_2024: pd.Series, dist_2025: pd.Series):
    """2024 vs 2025 구종 비율 급변 경고"""
    all_types = set(dist_2024.index) | set(dist_2025.index)
    warnings = []
    for pt in sorted(all_types):
        p24 = dist_2024.get(pt, 0.0) * 100
        p25 = dist_2025.get(pt, 0.0) * 100
        diff = p25 - p24
        if abs(diff) >= 10:
            warnings.append(f"    [WARNING] {pt}: {p24:.1f}% -> {p25:.1f}% (변화: {diff:+.1f}pp)")
    if warnings:
        print("\n  === 구종 분포 급변 경고 (±10pp 이상) ===")
        for w in warnings:
            print(w)
    else:
        print("\n  구종 분포 급변 없음 (모든 구종 ±10pp 이내)")


def _lookup_pitcher_cluster(pitcher_mlbam_id: int) -> str:
    csv_path = os.path.join(_ROOT, "data", "pitcher_clusters_2023.csv")
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            row = df[df['pitcher_id'] == pitcher_mlbam_id]
            if not row.empty:
                return str(int(row['cluster'].values[0]))
    except Exception:
        pass
    return "0"


def _get_all_pitcher_clusters() -> list:
    csv_path = os.path.join(_ROOT, "data", "pitcher_clusters_2023.csv")
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return [str(c) for c in sorted(df['cluster'].unique())]
    except Exception:
        pass
    return ["0"]


def main():
    print("=" * 60)
    print(f"  SmartPitch — {PITCHER_FIRST} {PITCHER_LAST} 개인 DQN 학습")
    print(f"  MLBAM ID: {PITCHER_ID}, 투수 군집: {PITCHER_CLUSTER}")
    print("=" * 60)

    # ── Cole 모델 해시 기록 (실행 전) ────────────────────────────────────
    cole_hash_before = _md5(COLE_MODEL_PATH)
    print(f"\n[Cole 모델 해시 (실행 전)] {cole_hash_before}")

    # ── 1. 데이터 수집 (2시즌 결합) ──────────────────────────────────────
    print("\n[단계 1] 투구 데이터 수집 (2시즌)")
    season_dfs = {}
    for start, end, label in SEASONS:
        print(f"  {label} 시즌: {start} ~ {end} ...")
        df_s = statcast_pitcher(start, end, player_id=PITCHER_ID)
        season_dfs[label] = df_s
        print(f"  -> {len(df_s)} 투구 수집")

    df_raw = pd.concat(list(season_dfs.values()), ignore_index=True)
    print(f"\n  총 raw 데이터: {len(df_raw)} 투구")

    # ── 구종 분포 비교 ───────────────────────────────────────────────────
    dists = {}
    for label, df_s in season_dfs.items():
        dists[label] = _print_pitch_distribution(df_s, label)

    if len(dists) == 2:
        _check_pitch_shift(dists["2024"], dists["2025"])

    # ── 2. 전처리 (PitchDataLoader._preprocess_data 재사용) ──────────────
    print("\n[단계 2] 데이터 전처리")
    loader = PitchDataLoader(PITCHER_FIRST, PITCHER_LAST, SEASONS[0][0], SEASONS[-1][1])
    loader.pitcher_mlbam_id = PITCHER_ID
    df_processed = loader._preprocess_data(df_raw)

    # 시즌별 전처리 후 건수
    n_2024 = len(df_processed[df_processed['pitcher'] == PITCHER_ID]) if 'pitcher' in df_processed.columns else len(df_processed)
    print(f"\n  전처리 후 합계: {len(df_processed)} 투구")

    if len(df_processed) < 2000:
        print("[ERROR] 투구 수가 2,000건 미만입니다. 학습을 중단합니다.")
        return

    # ── 3. 구종 식별 (PitchClustering) ───────────────────────────────────
    print("\n[단계 3] UMAP + K-Means 구종 자동 식별")
    clustering = PitchClustering(df_processed)
    df_clustered = clustering.run_clustering_pipeline()

    identified_pitch_names = list(clustering.pitch_map.values())
    best_k = len(identified_pitch_names)
    print(f"\n  PitchClustering 결과:")
    print(f"  - 선택된 K: {best_k}")
    print(f"  - 식별된 구종: {identified_pitch_names}")

    # ── 4. 범용 전이 모델 로드 ───────────────────────────────────────────
    print("\n[단계 4] 범용 전이 확률 모델 로드")
    model_module = TransitionProbabilityModel.load_from_checkpoint(
        model_path=os.path.join(_ROOT, "best_transition_model_universal.pth"),
        feature_columns_path=os.path.join(_ROOT, "data", "feature_columns_universal.json"),
        target_classes_path=os.path.join(_ROOT, "data", "target_classes_universal.json"),
        model_config_path=os.path.join(_ROOT, "data", "model_config_universal.json"),
    )

    feature_cols = model_module.feature_columns
    target_classes = model_module.target_classes
    strike_zones = sorted(list(df_clustered['zone'].dropna().unique()))

    print(f"  구종 수: {len(identified_pitch_names)}, 존 수: {len(strike_zones)}")
    print(f"  Action space: {len(identified_pitch_names) * len(strike_zones)}")

    # ── 5. W&B 초기화 ────────────────────────────────────────────────────
    pitcher_cluster_id = _lookup_pitcher_cluster(PITCHER_ID)
    pitcher_clusters_for_mdp = _get_all_pitcher_clusters()

    run = wandb.init(
        project="SmartPitch-Portfolio",
        name=f"{PITCHER_FIRST}_{PITCHER_LAST}_Pipeline",
        config={
            "pitcher": f"{PITCHER_FIRST} {PITCHER_LAST}",
            "pitcher_id": PITCHER_ID,
            "pitcher_cluster": PITCHER_CLUSTER,
            "seasons": "2024+2025",
            "n_pitches_total": len(df_processed),
            "identified_pitches": identified_pitch_names,
            "action_space": len(identified_pitch_names) * len(strike_zones),
            "model_type": "범용 MLP (pre-trained)",
            "dqn_total_timesteps": 300_000,
            "dqn_buffer_size": 100_000,
            "dqn_learning_rate": 1e-4,
            "dqn_exploration_fraction": 0.30,
            "dqn_exploration_final_eps": 0.05,
            "dqn_gamma": 0.99,
        },
    )

    try:
        # ── 6. MDP 최적 정책 ─────────────────────────────────────────────
        print("\n[단계 5] MDP 최적 정책 도출")
        optimizer = MDPOptimizer(
            transition_model=model_module,
            feature_columns=feature_cols,
            target_classes=target_classes,
            pitch_names=identified_pitch_names,
            zones=strike_zones,
            pitcher_clusters=pitcher_clusters_for_mdp,
            season=2024,
        )
        optimal_policy = optimizer.run_optimizer()

        # ── 7. DQN 학습 ─────────────────────────────────────────────────
        print("\n[단계 6] DQN 강화학습 에이전트 학습 (300K timesteps)")
        train_env = PitchEnv(
            transition_model=model_module,
            pitch_names=identified_pitch_names,
            zones=strike_zones,
            pitcher_cluster=int(pitcher_cluster_id),
            season=2024,
        )
        eval_env = PitchEnv(
            transition_model=model_module,
            pitch_names=identified_pitch_names,
            zones=strike_zones,
            pitcher_cluster=int(pitcher_cluster_id),
            season=2024,
        )

        trainer = (
            DQNTrainer(env=train_env, eval_env=eval_env)
            .build(
                learning_rate=1e-4,
                buffer_size=100_000,
                exploration_fraction=0.30,
                exploration_final_eps=0.05,
                gamma=0.99,
            )
        )

        save_dir = f"best_dqn_model_{PITCHER_LAST.lower()}"
        trainer.train(
            total_timesteps=300_000,
            save_path=save_dir,
            use_wandb=True,
        )

        # 최종 모델을 투수별 이름으로 이동
        default_final = os.path.join(_ROOT, "smartpitch_dqn_final.zip")
        target_final = os.path.join(_ROOT, f"dqn_{PITCHER_LAST.lower()}_2024_2025.zip")
        if os.path.exists(default_final):
            import shutil
            shutil.move(default_final, target_final)
            print(f"\n  모델 저장: {target_final}")

        # ── 8. 평가 (1000 에피소드) ──────────────────────────────────────
        print("\n[단계 7] DQN 정책 평가 (1000 에피소드)")
        trainer.evaluate(n_episodes=1000)

        # 정책 샘플 출력
        trainer.print_policy_sample(train_env)

    finally:
        wandb.finish()

    # ── Cole 모델 해시 확인 (실행 후) ────────────────────────────────────
    cole_hash_after = _md5(COLE_MODEL_PATH)
    print(f"\n[Cole 모델 해시 (실행 후)] {cole_hash_after}")
    if cole_hash_before == cole_hash_after:
        print("[OK] Cole 모델 변경 없음 확인")
    else:
        print("[WARNING] Cole 모델 해시가 변경되었습니다!")

    print(f"\n{'='*60}")
    print(f"  {PITCHER_FIRST} {PITCHER_LAST} DQN 학습 완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
