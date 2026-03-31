"""
pitch_env.py — DQN 강화학습을 위한 Gymnasium 커스텀 환경

역할:
    MLP 전이 모델을 시뮬레이터로 사용해 투구 1구씩 진행하는 이닝 환경.
    DQN 에이전트가 이 환경에서 탐색·학습하여 최적 투구 정책을 학습합니다.

관측 공간 (8D):
    [balls(0-3), strikes(0-2), outs(0-2), on_1b(0/1), on_2b(0/1), on_3b(0/1),
     batter_cluster(0-7), pitcher_cluster(0-K-1)]

행동 공간:
    Discrete(n_pitches × n_zones)
    예: 4구종 × 14코스 = 56
    디코딩: pitch_idx = action // n_zones,  zone_idx = action % n_zones

보상 함수:
    RE24(이전 상황) - RE24(이후 상황) - 실점
    ※ RE24는 아웃수+주자 상태만으로 결정 (count는 영향 없음)

에피소드:
    3아웃 = 이닝 종료 = terminated=True
    초기 상태: 0-0카운트, 랜덤 아웃수, 랜덤 주자 (다양한 상황 학습을 위해)

pitcher_cluster 파라미터:
    int  → 단일 투수 모드: 항상 동일한 투수 유형 사용 (main.py 실행 시)
    None → 범용 모드: 에피소드마다 랜덤 투수 유형 샘플링 (미래 범용 학습용)
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class PitchEnv(gym.Env):
    """
    MLB 투구 시뮬레이션 Gym 환경 (DQN 학습용)

    에피소드: 이닝 단위 (3아웃이 되면 종료)
    State  : [balls(0-3), strikes(0-2), outs(0-2), on_1b(0/1), on_2b(0/1), on_3b(0/1)]
    Action : 구종 × 존 조합 (Discrete)
    Reward : RE24 기반 실점 억제 기대값 변화 (투수 관점 → 높을수록 좋음)
    """

    metadata = {"render_modes": []}

    # 2019 MLB 평균 기대 득점(RE24) 매트릭스 (mdp_solver에서 통합)
    RE24_MATRIX = {
        "0_000": 0.481, "0_100": 0.859, "0_010": 1.100, "0_110": 1.437,
        "0_001": 1.350, "0_101": 1.784, "0_011": 1.964, "0_111": 2.292,
        "1_000": 0.254, "1_100": 0.509, "1_010": 0.664, "1_110": 0.884,
        "1_001": 0.939, "1_101": 1.130, "1_011": 1.376, "1_111": 1.541,
        "2_000": 0.098, "2_100": 0.224, "2_010": 0.319, "2_110": 0.429,
        "2_001": 0.353, "2_101": 0.471, "2_011": 0.580, "2_111": 0.736,
    }

    # 볼넷/사구 진루 매핑 (주자 상태 문자열 → (다음 주자 상태, 득점))
    WALK_ADVANCE = {
        "000": ("100", 0), "100": ("110", 0), "010": ("110", 0),
        "110": ("111", 0), "001": ("101", 0), "101": ("111", 0),
        "011": ("111", 0), "111": ("111", 1),
    }

    def __init__(self, transition_model, pitch_names: List[str], zones: List[float],
                 pitcher_cluster: Optional[int] = None):
        """
        :param transition_model  : 학습된 TransitionProbabilityModel 인스턴스
        :param pitch_names       : 클러스터링으로 식별된 구종 이름 리스트
        :param zones             : 투구 존 번호 리스트
        :param pitcher_cluster   : 고정 투수 군집 ID (int) — None이면 에피소드마다 무작위 선택
        """
        super().__init__()
        import os

        self.transition_model = transition_model
        self.pitch_names = pitch_names
        self.zones = [float(z) for z in zones]
        self.n_pitches = len(pitch_names)
        self.n_zones = len(self.zones)
        
        # ── 타자 군집(Cluster) 매핑 데이터 로드 ─────────────────────────────
        self.batter_clusters = {}
        batter_csv = os.path.join(os.path.dirname(__file__), "..", "data", "batter_clusters_2023.csv")
        try:
            if os.path.exists(batter_csv):
                df_b = pd.read_csv(batter_csv)
                self.batter_clusters = dict(zip(df_b['batter_id'], df_b['cluster']))
                print(f"[PitchEnv] 타자 군집 데이터 매핑 완료: {len(self.batter_clusters)}명")
            else:
                print(f"[PitchEnv] Warning: '{batter_csv}' 없음. 모든 타자를 기본 군집(0)으로 간주합니다.")
        except Exception as e:
            print(f"[PitchEnv] Error reading batter cluster csv: {e}. Defaulting to cluster 0.")

        # ── 투수 군집(Cluster) 매핑 데이터 로드 ─────────────────────────────
        # fixed_pitcher_cluster가 지정되면 항상 그 값 사용 (단일 투수 모드)
        # None이면 에피소드마다 랜덤 선택 (범용 모드)
        self.fixed_pitcher_cluster = pitcher_cluster
        self.n_pitcher_clusters = 1  # 기본값; CSV가 있으면 실제 K 값으로 업데이트
        pitcher_csv = os.path.join(os.path.dirname(__file__), "..", "data", "pitcher_clusters_2023.csv")
        try:
            if os.path.exists(pitcher_csv):
                df_p = pd.read_csv(pitcher_csv)
                self.n_pitcher_clusters = int(df_p['cluster'].max()) + 1
                print(f"[PitchEnv] 투수 군집 데이터 로드 완료: K={self.n_pitcher_clusters}")
            else:
                print(f"[PitchEnv] Warning: '{pitcher_csv}' 없음. 투수 군집 K=1로 기본 처리.")
        except Exception as e:
            print(f"[PitchEnv] Error reading pitcher cluster csv: {e}. K=1 fallback.")

        # ── 행동 공간: 구종 × 존 (이산) ─────────────────────────────────────
        self.action_space = gym.spaces.Discrete(self.n_pitches * self.n_zones)

        # ── 관측 공간: [balls, strikes, outs, 1b, 2b, 3b, batter_cluster, pitcher_cluster] ──
        # batter_cluster : 0 ~ 7  (K=8 고정)
        # pitcher_cluster: 0 ~ (n_pitcher_clusters - 1)  (K=4~8, 실루엣 탐색 결과)
        max_pc = max(self.n_pitcher_clusters - 1, 0)
        self.observation_space = gym.spaces.Box(
            low=np.array( [0, 0, 0, 0, 0, 0, 0, 0],       dtype=np.float32),
            high=np.array([3, 2, 2, 1, 1, 1, 7, max_pc],   dtype=np.float32),
            dtype=np.float32,
        )

        # 내부 상태 변수
        self.balls = 0
        self.strikes = 0
        self.outs = 0
        self.runners = [0, 0, 0]  # [1루, 2루, 3루]
        self.current_batter_id = None
        self.current_batter_cluster = 0
        self.current_pitcher_cluster = self.fixed_pitcher_cluster if self.fixed_pitcher_cluster is not None else 0

    # ─────────────────────────────────────────────────────────────────────────
    # Gym 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        새 이닝 시작.
        다양한 상황을 균등하게 경험시키기 위해 아웃카운트와 주자 상태를 무작위로 초기화합니다.
        무작위 타자를 선택하여 현재 이닝의 타자 군집 정보를 세팅합니다.
        """
        super().reset(seed=seed)
        self.balls = 0
        self.strikes = 0
        self.outs = int(self.np_random.integers(0, 3))           # 0, 1, 2 중 하나
        self.runners = list(self.np_random.integers(0, 2, size=3).tolist())  # 각 루 0 or 1
        
        # 무작위 타석 시뮬레이션을 위함
        # 매핑표가 있으면 실제 타자를 픽하고, 아니면 0~7 사이 랜덤값을 부여
        if self.batter_clusters:
            batter_ids = list(self.batter_clusters.keys())
            self.current_batter_id = self.np_random.choice(batter_ids)
            self.current_batter_cluster = self.batter_clusters.get(self.current_batter_id, 0)
        else:
            self.current_batter_id = -1
            self.current_batter_cluster = int(self.np_random.integers(0, 8))

        # 투수 군집: 고정값(단일 투수 모드)이면 그 값 유지, None이면 에피소드마다 무작위 선택
        if self.fixed_pitcher_cluster is not None:
            self.current_pitcher_cluster = self.fixed_pitcher_cluster
        else:
            self.current_pitcher_cluster = int(self.np_random.integers(0, self.n_pitcher_clusters))

        return self._get_obs(), {}

    def step(self, action: int):
        """
        투구 1구 실행.
        :param action: 행동 인덱스 (pitch_idx * n_zones + zone_idx)
        :return: (observation, reward, terminated, truncated, info)
        """
        pitch_idx = action // self.n_zones
        zone_idx = action % self.n_zones
        pitch = self.pitch_names[pitch_idx]
        zone = self.zones[zone_idx]

        # 보상 계산을 위한 사전 RE24
        re24_before = self._get_re24()

        # MLP 확률 예측 → 결과 샘플링
        outcome = self._sample_outcome(pitch, zone)

        # 상태 전이 및 실점 집계
        runs_scored = self._apply_outcome(outcome)

        # 사후 RE24
        re24_after = self._get_re24()

        # 보상 = RE24 감소량 - 실점 (투수 관점: 실점 억제할수록 +)
        reward = float(re24_before - re24_after - runs_scored)

        terminated = self.outs >= 3   # 3아웃 → 이닝 종료

        info = {
            "pitch": pitch,
            "zone": int(zone),
            "outcome": outcome,
            "runs_scored": runs_scored,
        }
        return self._get_obs(), reward, terminated, False, info

    def action_to_label(self, action: int) -> str:
        """행동 인덱스를 '구종 / 존' 문자열로 변환 (디버깅 및 정책 시각화용)"""
        pitch = self.pitch_names[action // self.n_zones]
        zone = int(self.zones[action % self.n_zones])
        return f"{pitch} / Zone {zone}"

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼 메서드
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.balls, self.strikes, self.outs,
                self.runners[0], self.runners[1], self.runners[2],
                self.current_batter_cluster,
                self.current_pitcher_cluster,
            ],
            dtype=np.float32,
        )

    def _get_re24(self) -> float:
        """현재 상태의 RE24 기대 득점 반환 (이닝 종료 시 0)"""
        if self.outs >= 3:
            return 0.0
        runners_str = "".join(map(str, self.runners))
        return self.RE24_MATRIX.get(f"{self.outs}_{runners_str}", 0.0)

    def _runners_str(self) -> str:
        return "".join(map(str, self.runners))

    def _sample_outcome(self, pitch: str, zone: float) -> str:
        """
        TransitionProbabilityModel에 현재 상태 + 행동을 입력해 투구 결과를 확률적으로 샘플링.
        모델 학습 시 사용한 One-Hot Encoding 형식과 동일하게 구성합니다.
        """
        # 모델 입력 DataFrame (Zero-initialized)
        input_df = pd.DataFrame(
            np.zeros((1, len(self.transition_model.feature_columns))),
            columns=self.transition_model.feature_columns,
        )

        # 수치 피처: 볼카운트/아웃/주자 직접 할당 (B안: count_state one-hot 대체)
        for col, val in [
            ('balls',   self.balls),
            ('strikes', self.strikes),
            ('outs',    self.outs),
            ('on_1b',   self.runners[0]),
            ('on_2b',   self.runners[1]),
            ('on_3b',   self.runners[2]),
        ]:
            if col in input_df.columns:
                input_df[col] = float(val)

        # 카테고리 피처: 구종/존/타자군집/투수군집 one-hot
        for col_key, col_val in [
            ("mapped_pitch_name", pitch),
            ("zone", zone),
            ("batter_cluster", str(int(self.current_batter_cluster))),
            ("pitcher_cluster", str(int(self.current_pitcher_cluster))),
        ]:
            col_name = f"{col_key}_{col_val}"
            if col_name in input_df.columns:
                input_df[col_name] = 1.0

        proba = self.transition_model.predict_proba(input_df)[0]

        # 수치 안정성 보정 후 샘플링
        proba = np.clip(proba, 0, None)
        proba /= proba.sum()
        outcome_idx = self.np_random.choice(len(proba), p=proba)
        return self.transition_model.target_classes[outcome_idx]

    def _apply_outcome(self, outcome: str) -> float:
        """
        투구 결과에 따라 볼카운트 / 아웃 / 주자 상태를 갱신하고 실점을 반환합니다.
        mdp_solver._get_next_states_and_rewards()의 전이 로직을 환경 내부에 통합.
        """
        runs = 0.0

        # ── 스트라이크 (strike 그룹) ─────────────────────────────────────────
        # called_strike / swinging_strike / foul_tip / swinging_strike_blocked
        # / missed_bunt / bunt_foul_tip 이 모두 "strike"로 병합됨
        if outcome == "strike":
            self.strikes += 1
            if self.strikes >= 3:          # 삼진
                self.outs += 1
                self.balls, self.strikes = 0, 0

        # ── 파울 (foul 그룹) ─────────────────────────────────────────────────
        # foul / foul_bunt 가 "foul"로 병합됨
        elif outcome == "foul":
            if self.strikes < 2:           # 2스트라이크 이후 파울은 카운트 불변
                self.strikes += 1

        # ── 볼 (ball 그룹) ───────────────────────────────────────────────────
        # ball / blocked_ball 이 "ball"로 병합됨
        elif outcome == "ball":
            self.balls += 1
            if self.balls >= 4:            # 볼넷
                runs = self._apply_walk()
                self.balls, self.strikes = 0, 0

        # ── 사구(HBP) ─────────────────────────────────────────────────────────
        # 범용 모델(4클래스)에서는 HBP가 발생하지 않음. 단일 투수 모드 전용.
        elif outcome == "hit_by_pitch":
            runs = self._apply_walk()
            self.balls, self.strikes = 0, 0

        # ── 인플레이 타구 ─────────────────────────────────────────────────────
        elif outcome == "hit_into_play":
            runs = self._apply_batted_ball()
            self.balls, self.strikes = 0, 0

        # ── 기타 (상태 유지) ──────────────────────────────────────────────────
        # else: 아무것도 변경하지 않음

        return float(runs)

    def _apply_walk(self) -> float:
        """볼넷 / 사구 진루 처리. WALK_ADVANCE 테이블 기반."""
        key = self._runners_str()
        next_runners_str, runs = self.WALK_ADVANCE.get(key, (key, 0))
        self.runners = [int(c) for c in next_runners_str]
        return float(runs)

    def _apply_batted_ball(self) -> float:
        """
        인플레이 타구 확률적 처리 (mdp_solver 동일 가정).
          - 범타 아웃  70%
          - 1루타      15%
          - 2루타      10%
          - 홈런        5%
        """
        p = self.np_random.random()
        r1, r2, r3 = self.runners
        runs = 0

        if p < 0.70:    # 범타 아웃
            self.outs += 1

        elif p < 0.85:  # 1루타: 타자→1루, 주자 1루씩 진루, 3루 주자 득점
            runs = r3
            self.runners = [1, r1, r2]

        elif p < 0.95:  # 2루타: 타자→2루, 1루 주자→3루, 2/3루 주자 득점
            runs = r2 + r3
            self.runners = [0, 1, r1]

        else:           # 홈런: 모든 주자 + 타자 득점
            runs = 1 + r1 + r2 + r3
            self.runners = [0, 0, 0]

        return float(runs)
