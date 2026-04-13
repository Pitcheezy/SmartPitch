"""
train_dqn_all_clusters.py — 투수 군집별(K=4) DQN 학습 + 평가 스크립트

군집 0~3 각각에 대해 독립적인 DQN을 학습시키고 1000 에피소드 평가를 수행합니다.
이미 학습된 군집(data/dqn_cluster_{cid}.zip 존재)은 --skip-existing 옵션으로 스킵 가능.

학습 설정 (main.py 기존 설정과 동일):
    total_timesteps    : 300,000
    buffer_size        : 100,000
    learning_rate      : 1e-4
    exploration_fraction: 0.30
    exploration_final_eps: 0.05
    gamma              : 0.99
    net_arch           : [128, 64]

산출물:
    data/dqn_cluster_{cid}.zip          — 각 군집별 최종 DQN 모델
    data/dqn_cluster_{cid}_best/        — EvalCallback 기준 최고 모델
    data/dqn_cluster_{cid}_eval.json    — 1000 에피소드 평가 결과

실행:
    uv run scripts/train_dqn_all_clusters.py
    uv run scripts/train_dqn_all_clusters.py --skip-existing   # 이미 학습된 군집 스킵
    uv run scripts/train_dqn_all_clusters.py --clusters 1 2 3  # 특정 군집만 학습
    uv run scripts/train_dqn_all_clusters.py --eval-only        # 학습 없이 평가만
"""
import os
import sys
import json
import argparse
import time

import numpy as np

# 프로젝트 루트 등록
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import TransitionProbabilityModel
from src.pitch_env import PitchEnv, get_valid_pitches
from src.rl_trainer import DQNTrainer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DATA_DIR = os.path.join(_ROOT, 'data')


def _load_universal_model():
    """범용 전이 모델 로드"""
    model = TransitionProbabilityModel.load_from_checkpoint(
        model_path=os.path.join(_ROOT, "best_transition_model_universal.pth"),
        feature_columns_path=os.path.join(_DATA_DIR, "feature_columns_universal.json"),
        target_classes_path=os.path.join(_DATA_DIR, "target_classes_universal.json"),
        model_config_path=os.path.join(_DATA_DIR, "model_config_universal.json"),
    )
    return model


def _parse_pitch_names_and_zones(feature_columns):
    """feature_columns에서 pitch_names와 zones를 파싱"""
    pitch_names = []
    zones = []
    for col in feature_columns:
        if col.startswith("mapped_pitch_name_"):
            pitch_names.append(col.replace("mapped_pitch_name_", ""))
        elif col.startswith("zone_"):
            zones.append(float(col.replace("zone_", "")))
    return sorted(pitch_names), sorted(zones)


def _evaluate_dqn(model_path, transition_model, pitch_names, zones, cluster_id,
                  n_episodes=1000, seed_base=0):
    """학습된 DQN 모델을 로드하여 n_episodes 에피소드 평가"""
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    # 군집별 유효 구종 필터링 (Task 18)
    valid_pitches = get_valid_pitches(cluster_id, pitch_names)

    env = PitchEnv(
        transition_model=transition_model,
        pitch_names=valid_pitches,
        zones=zones,
        pitcher_cluster=cluster_id,
    )
    env = Monitor(env)

    dqn = DQN.load(model_path, env=env)

    episode_rewards = []
    pitch_counts = {}
    total_pitches = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = dqn.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
            total_pitches += 1
            p = info.get("pitch", "")
            pitch_counts[p] = pitch_counts.get(p, 0) + 1
        episode_rewards.append(ep_reward)

    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))
    mean_pitches = total_pitches / n_episodes

    # 구종 분포 퍼센트
    total = sum(pitch_counts.values())
    pitch_pct = {k: round(v / total * 100, 1) for k, v in
                 sorted(pitch_counts.items(), key=lambda x: -x[1])}

    return {
        "cluster": cluster_id,
        "mean_reward": round(mean_r, 4),
        "std_reward": round(std_r, 4),
        "mean_pitches_per_ep": round(mean_pitches, 2),
        "n_episodes": n_episodes,
        "pitch_distribution_pct": pitch_pct,
        "action_space": len(valid_pitches) * len(zones),
    }


def train_and_evaluate(cluster_id, transition_model, pitch_names, zones,
                       total_timesteps=300_000, skip_existing=False, eval_only=False,
                       use_wandb=True):
    """단일 군집에 대해 DQN 학습 + 평가 수행"""
    import wandb

    model_path = os.path.join(_DATA_DIR, f"dqn_cluster_{cluster_id}")
    model_zip = model_path + ".zip"
    best_model_dir = os.path.join(_DATA_DIR, f"dqn_cluster_{cluster_id}_best")
    eval_json = os.path.join(_DATA_DIR, f"dqn_cluster_{cluster_id}_eval.json")

    # 스킵 조건 확인
    if skip_existing and os.path.exists(model_zip):
        print(f"\n[cluster {cluster_id}] 이미 학습된 모델 존재 ({model_zip}), 스킵")
        # 평가만 수행
        if not os.path.exists(eval_json):
            print(f"[cluster {cluster_id}] 평가 결과 없음 — 평가 실행")
            result = _evaluate_dqn(model_path, transition_model, pitch_names, zones, cluster_id)
            with open(eval_json, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[cluster {cluster_id}] 평가 완료: mean={result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
        return

    if eval_only:
        if os.path.exists(model_zip):
            print(f"\n[cluster {cluster_id}] 평가만 수행")
            result = _evaluate_dqn(model_path, transition_model, pitch_names, zones, cluster_id)
            with open(eval_json, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[cluster {cluster_id}] mean={result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
            print(f"[cluster {cluster_id}] 구종: {result['pitch_distribution_pct']}")
        else:
            print(f"\n[cluster {cluster_id}] 모델 없음 ({model_zip}), 스킵")
        return

    # 군집별 유효 구종 필터링 (Task 18)
    valid_pitches = get_valid_pitches(cluster_id, pitch_names)

    print(f"\n{'='*60}")
    print(f"  DQN 학습 시작 — pitcher_cluster={cluster_id}")
    print(f"  유효 구종: {valid_pitches} ({len(valid_pitches)}종)")
    print(f"  액션 스페이스: {len(valid_pitches) * len(zones)}")
    print(f"  총 {total_timesteps:,} 타임스텝")
    print(f"{'='*60}")

    # W&B run 초기화 (군집별 독립 run)
    run = None
    if use_wandb:
        try:
            run = wandb.init(
                project="SmartPitch-Portfolio",
                name=f"DQN_cluster_{cluster_id}",
                config={
                    "pitcher_cluster": cluster_id,
                    "total_timesteps": total_timesteps,
                    "buffer_size": 100_000,
                    "learning_rate": 1e-4,
                    "exploration_fraction": 0.30,
                    "exploration_final_eps": 0.05,
                    "gamma": 0.99,
                    "net_arch": [128, 64],
                    "action_space": len(valid_pitches) * len(zones),
                    "pitch_names": valid_pitches,
                    "n_zones": len(zones),
                },
                reinit=True,
            )
        except Exception as e:
            print(f"[W&B] 초기화 실패: {e}. W&B 없이 계속합니다.")
            use_wandb = False

    try:
        # 환경 생성 (유효 구종만 사용)
        train_env = PitchEnv(
            transition_model=transition_model,
            pitch_names=valid_pitches,
            zones=zones,
            pitcher_cluster=cluster_id,
        )
        eval_env = PitchEnv(
            transition_model=transition_model,
            pitch_names=valid_pitches,
            zones=zones,
            pitcher_cluster=cluster_id,
        )

        # DQN 학습
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

        start_time = time.time()
        trainer.train(
            total_timesteps=total_timesteps,
            eval_freq=10_000,
            n_eval_episodes=20,
            save_path=best_model_dir,
            use_wandb=use_wandb,
        )
        elapsed = time.time() - start_time
        print(f"[cluster {cluster_id}] 학습 완료 ({elapsed/60:.1f}분)")

        # 모델 저장 (rl_trainer가 smartpitch_dqn_final.zip으로 저장하므로 이동)
        default_path = os.path.join(_ROOT, "smartpitch_dqn_final.zip")
        if os.path.exists(default_path):
            import shutil
            shutil.move(default_path, model_zip)
            print(f"[cluster {cluster_id}] 모델 저장: {model_zip}")

        # 1000 에피소드 평가
        print(f"[cluster {cluster_id}] 1000 에피소드 평가 중...")
        result = _evaluate_dqn(model_path, transition_model, pitch_names, zones, cluster_id)
        result["train_time_min"] = round(elapsed / 60, 1)

        with open(eval_json, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[cluster {cluster_id}] 평가 완료:")
        print(f"  mean={result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
        print(f"  pitches/ep={result['mean_pitches_per_ep']}")
        print(f"  구종: {result['pitch_distribution_pct']}")

        # W&B에 평가 결과 로깅
        if use_wandb and wandb.run:
            wandb.log({
                "eval/mean_reward": result["mean_reward"],
                "eval/std_reward": result["std_reward"],
                "eval/mean_pitches_per_ep": result["mean_pitches_per_ep"],
            })

    finally:
        if run:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="투수 군집별 DQN 학습 + 평가")
    parser.add_argument("--clusters", nargs="+", type=int, default=[0, 1, 2, 3],
                        help="학습할 군집 ID 목록 (기본: 0 1 2 3)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="이미 학습된 군집(zip 존재) 스킵")
    parser.add_argument("--eval-only", action="store_true",
                        help="학습 없이 기존 모델 평가만 수행")
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="군집당 학습 타임스텝 (기본: 300000)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="W&B 로깅 비활성화")
    args = parser.parse_args()

    print("=" * 60)
    print("SmartPitch — 투수 군집별 DQN 학습")
    print(f"  대상 군집: {args.clusters}")
    print(f"  타임스텝: {args.timesteps:,}")
    print(f"  skip-existing: {args.skip_existing}")
    print(f"  eval-only: {args.eval_only}")
    print(f"  W&B: {'OFF' if args.no_wandb else 'ON'}")
    print("=" * 60)

    # 범용 모델 로드 (1회만)
    print("\n[모델 로드] 범용 전이 모델...")
    transition_model = _load_universal_model()
    pitch_names, zones = _parse_pitch_names_and_zones(transition_model.feature_columns)
    print(f"  pitch_names({len(pitch_names)}): {pitch_names}")
    print(f"  zones({len(zones)}): {zones}")
    print(f"  action_space: {len(pitch_names) * len(zones)}")

    # 군집별 학습 + 평가
    for cid in args.clusters:
        train_and_evaluate(
            cluster_id=cid,
            transition_model=transition_model,
            pitch_names=pitch_names,
            zones=zones,
            total_timesteps=args.timesteps,
            skip_existing=args.skip_existing,
            eval_only=args.eval_only,
            use_wandb=not args.no_wandb,
        )

    # 전체 결과 요약
    print(f"\n{'='*60}")
    print("전체 결과 요약")
    print(f"{'='*60}")
    print(f"| Cluster | Mean Reward ± Std | Pitches/Ep | Action Space |")
    print(f"|---------|-------------------|------------|--------------|")
    for cid in args.clusters:
        eval_json = os.path.join(_DATA_DIR, f"dqn_cluster_{cid}_eval.json")
        if os.path.exists(eval_json):
            with open(eval_json) as f:
                r = json.load(f)
            print(f"| {cid} | +{r['mean_reward']:.3f} ± {r['std_reward']:.3f} | "
                  f"{r['mean_pitches_per_ep']} | {r['action_space']} |")
        else:
            print(f"| {cid} | — | — | — |")

    print("\n[done]")


if __name__ == "__main__":
    main()
