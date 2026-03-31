"""
rl_trainer.py — DQN 강화학습 에이전트 학습 및 평가

역할:
    pitch_env.py의 PitchEnv 위에서 Stable-Baselines3 DQN 에이전트를 학습하고 평가합니다.
    MDP Solver와의 차이: MDP는 모델 기반(Model-Based)이지만, DQN은 환경과 직접 상호작용하며
    Q-함수를 학습하는 모델 프리(Model-Free) 방식입니다.

DQN 선택 이유:
    - 이산 행동 공간 (구종 × 존 = 수십 개의 선택지) → DQN이 적합
    - 관측 벡터 8D로 단순한 상태 표현 → MLP Policy로 충분
    - Off-Policy(Replay Buffer) → 데이터 효율성 높음

핵심 컴포넌트:
    WandbDQNCallback : 에피소드마다 보상/길이를 W&B에 실시간 로깅
    DQNTrainer.build(): DQN 모델 초기화 (메서드 체이닝 지원)
    DQNTrainer.train(): EvalCallback + W&B 콜백으로 학습 실행
    DQNTrainer.evaluate(): 결정론적 정책으로 100이닝 평가
    DQNTrainer.print_policy_sample(): 주요 볼카운트별 추천 구종 출력

학습 설정 (main.py에서 wandb.config로 주입):
    total_timesteps    : 300,000  (목표: 500,000)
    buffer_size        : 100,000  (Replay Buffer)
    learning_rate      : 1e-4
    exploration_fraction: 0.30   (전체 스텝의 30%는 ε-greedy 탐색)
    exploration_final_eps: 0.05  (탐색 이후 최소 탐색률 5%)
    gamma              : 0.99    (미래 보상 할인율)
    net_arch           : [128, 64] (MLP Policy 은닉층, model.py와 동일)

관측 공간 (8D):
    [balls(0-3), strikes(0-2), outs(0-2), on_1b(0/1), on_2b(0/1), on_3b(0/1),
     batter_cluster(0-7), pitcher_cluster(0-K-1)]

보상 함수 (PitchEnv에서 정의):
    아웃 발생: +RE24 (현 상황 기대실점 감소분)
    안타/볼넷: -RE24 (현 상황 기대실점 증가분)
    이닝 종료(3아웃): 에피소드 종료

저장 파일:
    best_dqn_model/best_model.zip   — EvalCallback 기준 최고 성능
    smartpitch_dqn_final.zip        — 학습 완료 후 최종 모델
"""
import numpy as np
import wandb
from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


# ─────────────────────────────────────────────────────────────────────────────
# W&B 로깅 콜백
# ─────────────────────────────────────────────────────────────────────────────

class WandbDQNCallback(BaseCallback):
    """
    에피소드가 끝날 때마다 누적 보상과 에피소드 길이를 W&B에 로깅합니다.
    stable-baselines3의 Monitor 래퍼가 info['episode']에 통계를 채워줍니다.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if not (wandb.run and self.locals.get("dones")):
            return True
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep:
                wandb.log({
                    "dqn/episode_reward":  ep["r"],
                    "dqn/episode_length":  ep["l"],
                    "dqn/timestep":        self.num_timesteps,
                    "dqn/exploration_rate": self.model.exploration_rate,
                })
        return True


# ─────────────────────────────────────────────────────────────────────────────
# DQN 트레이너
# ─────────────────────────────────────────────────────────────────────────────

class DQNTrainer:
    """
    PitchEnv 위에서 DQN 에이전트를 학습하고 평가하는 클래스.

    주요 하이퍼파라미터 설명:
    - buffer_size       : Replay Buffer 크기. 과거 경험을 얼마나 저장할지 결정합니다.
    - learning_starts   : 이 스텝 수만큼 랜덤 탐색 후 학습을 시작합니다.
    - exploration_fraction : 전체 학습 중 ε이 1.0 → final_eps로 감소하는 구간 비율.
    - target_update_interval : Target Network를 몇 스텝마다 동기화할지.
    - net_arch          : Q-Network의 은닉층 구조 (기존 MLP와 동일하게 [128, 64]).
    """

    def __init__(self, env, eval_env=None):
        """
        :param env      : 학습용 PitchEnv (Monitor 래핑 전 원본 전달)
        :param eval_env : 평가용 PitchEnv (None이면 env 재사용)
        """
        self.train_env = Monitor(env)
        self.eval_env = Monitor(eval_env if eval_env is not None else env)
        self.model: Optional[DQN] = None

    # ── 모델 초기화 ───────────────────────────────────────────────────────────

    def build(
        self,
        learning_rate: float = 1e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        train_freq: int = 4,
        target_update_interval: int = 1_000,
        exploration_fraction: float = 0.30,
        exploration_final_eps: float = 0.05,
    ) -> "DQNTrainer":
        """
        DQN 모델을 빌드합니다. 메서드 체이닝이 가능하도록 self를 반환합니다.
        """
        self.model = DQN(
            policy="MlpPolicy",
            env=self.train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=train_freq,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=dict(net_arch=[128, 64]),  # 기존 MLP 구조와 동일
            verbose=1,
        )
        print(
            f"DQN 모델 빌드 완료\n"
            f"  행동 공간  : {self.train_env.action_space.n}개 (구종 × 존)\n"
            f"  관측 공간  : {self.train_env.observation_space.shape}\n"
            f"  네트워크   : [128, 64]\n"
            f"  Replay Buffer: {buffer_size:,} 스텝\n"
        )
        return self

    # ── 학습 ─────────────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 300_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        save_path: str = "best_dqn_model",
        use_wandb: bool = True,
    ) -> DQN:
        """
        DQN 학습 실행.

        :param total_timesteps  : 총 학습 스텝 수
        :param eval_freq        : 몇 스텝마다 정책을 평가할지
        :param n_eval_episodes  : 평가 시 몇 에피소드를 돌릴지
        :param save_path        : 최고 성능 모델 저장 경로
        :param use_wandb        : W&B 로깅 활성화 여부
        :return: 학습된 DQN 모델
        """
        if self.model is None:
            raise RuntimeError("build()를 먼저 호출하세요.")

        callbacks = []

        # W&B 에피소드 로깅
        if use_wandb and wandb.run:
            callbacks.append(WandbDQNCallback())

        # 주기적 평가 + 최고 모델 저장
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"./{save_path}/",
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_callback)

        print(f"\nDQN 학습 시작 (총 {total_timesteps:,} 타임스텝)")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

        # 최종 모델 저장
        final_path = "smartpitch_dqn_final"
        self.model.save(final_path)
        print(f"최종 모델 저장 완료: {final_path}.zip")

        # 학습된 모델을 W&B Artifact로 업로드
        if use_wandb and wandb.run:
            artifact = wandb.Artifact(name="smartpitch_dqn_model", type="model")
            artifact.add_file(f"{final_path}.zip")
            wandb.log_artifact(artifact)
            print("W&B Artifact에 DQN 모델 업로드 완료!")

        return self.model

    # ── 정책 평가 ─────────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 100) -> dict:
        """
        학습된 정책을 결정론적(deterministic=True)으로 실행하며 성능을 평가합니다.

        :param n_episodes : 평가할 이닝 수
        :return: 평가 결과 딕셔너리
        """
        if self.model is None:
            raise RuntimeError("학습된 모델이 없습니다. train()을 먼저 실행하세요.")

        episode_rewards = []
        pitch_counts = {}
        zone_counts = {}

        obs, _ = self.eval_env.reset()

        for ep in range(n_episodes):
            ep_reward = 0.0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                ep_reward += reward
                done = terminated or truncated

                # 구종 / 존 분포 집계
                pitch = info.get("pitch", "")
                zone = info.get("zone", "")
                pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1
                zone_counts[zone] = zone_counts.get(zone, 0) + 1

            episode_rewards.append(ep_reward)
            obs, _ = self.eval_env.reset()

        mean_r = float(np.mean(episode_rewards))
        std_r = float(np.std(episode_rewards))

        print(f"\n{'='*50}")
        print(f"  [DQN 정책 평가 결과] ({n_episodes} 이닝)")
        print(f"  평균 보상 : {mean_r:.4f} ± {std_r:.4f}")
        print(f"  구종 분포 : {pitch_counts}")
        print(f"  존 분포   : {zone_counts}")
        print(f"{'='*50}\n")

        result = {
            "mean_reward": mean_r,
            "std_reward": std_r,
            "pitch_distribution": pitch_counts,
            "zone_distribution": zone_counts,
        }

        if wandb.run:
            wandb.log({
                "dqn_eval/mean_reward":  mean_r,
                "dqn_eval/std_reward":   std_r,
            })
            # 구종 분포를 W&B Bar Chart로 로깅
            pitch_table = wandb.Table(
                columns=["Pitch", "Count"],
                data=[[k, v] for k, v in sorted(pitch_counts.items(), key=lambda x: -x[1])],
            )
            wandb.log({"dqn_eval/pitch_distribution": wandb.plot.bar(
                pitch_table, "Pitch", "Count", title="DQN 최적 구종 분포"
            )})

        return result

    # ── 정책 시각화 ───────────────────────────────────────────────────────────

    def print_policy_sample(self, env):
        """
        주요 볼카운트 상황에서 학습된 정책이 선택하는 행동을 출력합니다.
        (0아웃 주자 없음 기준)
        """
        if self.model is None:
            return

        print("\n[학습된 정책 샘플 — 0아웃 주자 없음]")
        print(f"{'볼카운트':<12} {'추천 구종':<15} {'추천 존':<8}")
        print("-" * 38)

        for balls in range(4):
            for strikes in range(3):
                # 8차원(balls, strikes, outs, 1b, 2b, 3b, batter_cluster, pitcher_cluster) 더미 배열
                obs = np.array([balls, strikes, 0, 0, 0, 0, 0, 0], dtype=np.float32)
                action, _ = self.model.predict(obs, deterministic=True)
                label = env.action_to_label(int(action))
                pitch, zone = label.split(" / ")
                print(f"{balls}-{strikes:<11} {pitch:<15} {zone}")
