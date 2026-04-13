"""
mdp_solver.py — MDP 가치반복(Value Iteration) 기반 최적 투구 전략 계산

역할:
    학습된 MLP(model.py)를 이용해 각 게임 상황에서 기대 실점(RE24)을 최소화하는
    최적 구종+코스 정책(Policy)을 계산합니다.

상태 공간:
    상태 키 형식: "{count}_{outs}_{runners}_{batter_cluster}_{pitcher_cluster}"
    예시: "3-2_2_111_7_0"
      - count     = "3-2"      볼-스트라이크 (볼0~3, 스트라이크0~2)
      - outs      = "2"        아웃 수 (0~2)
      - runners   = "111"      주자 상태 (1루/2루/3루, 각 0 또는 1)
      - batter_cl = "7"        타자 군집 (0~7)
      - pitcher_cl= "0"        투수 군집 (0~K-1, 현재 K=4)

    총 상태 수: 12 × 3 × 8 × 8 × K
      - 단일 투수 모드 (K=1): 2,304개
      - 범용 모드     (K=4): 9,216개

행동 공간:
    (구종 수) × (코스 수 14) 개의 이산 행동
    예: 4구종 × 14코스 = 56가지 행동

보상 함수:
    reward = RE24_before - RE24_after - runs_scored
    양수: 투수가 기대 실점을 줄임 (좋음)
    음수: 기대 실점이 증가하거나 실점 발생 (나쁨)

RE24 매트릭스:
    2019 MLB 평균 기대 득점 (아웃수_주자상태 → 기대득점)
    주의: 현재 2019 기준 하드코딩 → 향후 연도별 갱신 필요

알고리즘:
    Value Iteration 최대 20회 반복 (max|ΔV| < 1e-4 조기 종료)
    γ=0.99 할인율 적용 (foul self-loop 가치 무한 누적 방지)
    각 반복에서 모든 상태에 대해 최적 행동 탐색
    MLP를 매번 호출하므로 상태 수가 많을수록 느림 (GPU 권장)
"""
import os
import pandas as pd
import numpy as np
import wandb
import itertools
from typing import List, Dict, Tuple, Any, Optional

class MDPOptimizer:
    """
    TransitionProbabilityModel의 예측 능력을 활용해,
    벨만 방정식(Value Iteration)을 거꾸로 계산하여 최적의 볼배합을 찾아내는 클래스 (RE24 기반)
    """
    
    def __init__(self, transition_model, feature_columns: List[str], target_classes: List[str], pitch_names: List[str], zones: List[float], pitcher_clusters: List[str] = None, valid_pitches_by_cluster: Dict[str, List[str]] = None):
        """
        초기화 메서드
        :param transition_model: 학습된 TransitionProbabilityModel 객체 (predict_proba 메서드 제공)
        :param feature_columns: 모델 입력에 사용된 One-Hot Encoding 컬럼 리스트
        :param target_classes: 모델 출력의 클래스(결과) 리스트
        :param pitch_names: 클러스터링으로 식별된 구종 이름 리스트
        :param zones: 투구 코스(존) 리스트
        :param pitcher_clusters: 사용할 투수 군집 ID 문자열 리스트 (예: ["0","1","2"]).
                                  None이면 ["0"] (단일 투수 모드 — 상태 수 동일 유지)
        :param valid_pitches_by_cluster: 군집별 유효 구종 딕셔너리 (예: {"0": ["Fastball","Slider",...], ...}).
                                          None이면 모든 군집에서 pitch_names 전체 사용 (기존 동작)
        """
        self.transition_model = transition_model
        self.feature_columns = feature_columns
        self.target_classes = target_classes
        self.pitch_names = pitch_names
        self.zones = zones
        # 단일 투수 모드: pitcher_clusters=["0"] → 2,304개 상태 유지
        # 범용 모드: pitcher_clusters=["0","1",...,"K-1"] → 2,304×K 상태
        self.pitcher_clusters = pitcher_clusters if pitcher_clusters is not None else ["0"]

        # 군집별 유효 구종 필터 (Task 18: 액션 스페이스 최적화)
        # None이면 모든 군집에서 pitch_names 전체 사용 (기존 동작 호환)
        self.valid_pitches_by_cluster = valid_pitches_by_cluster
        
        # 2019 MLB 평균 기대 득점(RE24) 매트릭스 (투수 목표: 이를 낮추는 것)
        # 키 형식: '아웃_주자' (예: '0_000', '2_111')
        self.re24_matrix = {
            '0_000': 0.481, '0_100': 0.859, '0_010': 1.100, '0_110': 1.437, 
            '0_001': 1.350, '0_101': 1.784, '0_011': 1.964, '0_111': 2.292,
            '1_000': 0.254, '1_100': 0.509, '1_010': 0.664, '1_110': 0.884, 
            '1_001': 0.939, '1_101': 1.130, '1_011': 1.376, '1_111': 1.541,
            '2_000': 0.098, '2_100': 0.224, '2_010': 0.319, '2_110': 0.429, 
            '2_001': 0.353, '2_101': 0.471, '2_011': 0.580, '2_111': 0.736
        }
        
        # ── 물리 피처 lookup 테이블 로드 (Task 12 Phase 2) ──────────────────
        # (pitcher_cluster, mapped_pitch_name) → (release_speed_n, pfx_x_n, pfx_z_n)
        self.physical_lookup = {}
        lookup_csv = os.path.join(os.path.dirname(__file__), "..", "data", "physical_feature_lookup.csv")
        try:
            if os.path.exists(lookup_csv):
                df_lk = pd.read_csv(lookup_csv)
                for _, row in df_lk.iterrows():
                    key = (str(int(row['pitcher_cluster'])), row['mapped_pitch_name'])
                    self.physical_lookup[key] = {
                        'release_speed_n': float(row['release_speed_n']),
                        'pfx_x_n': float(row['pfx_x_n']),
                        'pfx_z_n': float(row['pfx_z_n']),
                    }
                print(f"[MDPOptimizer] 물리 피처 lookup 로드 완료: {len(self.physical_lookup)}개 항목")
            else:
                print(f"[MDPOptimizer] Warning: '{lookup_csv}' 없음. 물리 피처는 0으로 채워집니다.")
        except Exception as e:
            print(f"[MDPOptimizer] Error reading physical lookup csv: {e}. 물리 피처 0 fallback.")

        self.state_values = {}
        self.optimal_policy = {}

    def _get_re24(self, outs: int, runners: str) -> float:
        """아웃카운트와 주자 상태에 따른 RE24 값을 반환"""
        if outs >= 3:
            return 0.0 # 이닝 종료 시 기대 득점은 0
        return self.re24_matrix.get(f"{outs}_{runners}", 0.0)

    def _advance_runners_walk(self, runners: str) -> Tuple[str, int]:
        """볼넷/사구 발생 시 진루 로직. 반환값: (새 주자상태, 득점)"""
        if runners == '000': return '100', 0
        elif runners == '100': return '110', 0
        elif runners == '010': return '110', 0
        elif runners == '110': return '111', 0
        elif runners == '001': return '101', 0
        elif runners == '101': return '111', 0
        elif runners == '011': return '111', 0
        elif runners == '111': return '111', 1 # 밀어내기 득점
        return runners, 0

    def _get_next_states_and_rewards(self, current_state: str, outcome: str) -> List[Tuple[str, float, float]]:
        """
        [내부 메서드] 현재 상태와 투구 결과를 바탕으로 다음 상태(볼카운트/아웃/주자)와 확률 분기, 득점을 반환
        return: List of (next_state_key, probability, runs_scored)
        """
        try:
            parts = current_state.split('_')
            count, outs_str, runners, batter_cluster, pitcher_cluster = parts
            b, s = map(int, count.split('-'))
            outs = int(outs_str)
        except Exception:
            return [(current_state, 1.0, 0.0)]

        outcomes = [] # (next_count, next_outs, next_runners, prob, runs)

        # 1. 스트라이크 그룹 (strike)
        # called_strike / swinging_strike / foul_tip / swinging_strike_blocked
        # / missed_bunt / bunt_foul_tip 이 모두 "strike"로 병합됨
        if outcome == 'strike':
            s += 1
            if s >= 3:
                # 삼진 아웃: 카운트 리셋, 아웃 증가
                outcomes.append(("0-0", outs + 1, runners, 1.0, 0))
            else:
                outcomes.append((f"{b}-{s}", outs, runners, 1.0, 0))

        # 2. 파울 그룹 (foul)
        # foul / foul_bunt 가 "foul"로 병합됨
        elif outcome == 'foul':
            if s < 2:
                s += 1
            # 2스트라이크 이후 파울은 카운트 유지
            outcomes.append((f"{b}-{s}", outs, runners, 1.0, 0))

        # 3. 볼 그룹 (ball)
        # ball / blocked_ball 이 "ball"로 병합됨
        elif outcome == 'ball':
            b += 1
            if b >= 4:
                # 볼넷: 카운트 리셋, 주자 진루
                next_runners, runs = self._advance_runners_walk(runners)
                outcomes.append(("0-0", outs, next_runners, 1.0, runs))
            else:
                outcomes.append((f"{b}-{s}", outs, runners, 1.0, 0))

        # 4. 몸에 맞는 볼 (hit_by_pitch)
        elif outcome == 'hit_by_pitch':
            # 사구: 카운트 리셋, 주자 진루
            next_runners, runs = self._advance_runners_walk(runners)
            outcomes.append(("0-0", outs, next_runners, 1.0, runs))

        # 5. 인플레이 타구 (hit_into_play, 단순화된 확률적 분기)
        elif outcome == 'hit_into_play':
            # 인플레이 시 해당 타석 종료이므로 다음 타자 카운트는 무조건 0-0
            
            # 5-1. 범타 아웃 (70% 확률) - 주자 변동 없다고 가정
            outcomes.append(("0-0", outs + 1, runners, 0.70, 0))
            
            # 5-2. 1루타 (15% 확률) - 타자 1루, 주자 1루씩 진루, 3루 주자 득점
            single_runners = {'000':'100', '100':'110', '010':'101', '110':'111', 
                              '001':'100', '101':'110', '011':'101', '111':'111'}
            single_runs = 1 if runners[2] == '1' else 0
            outcomes.append(("0-0", outs, single_runners.get(runners, '100'), 0.15, single_runs))
            
            # 5-3. 2루타 (10% 확률) - 타자 2루, 주자 2루씩 진루, 2/3루 주자 득점
            double_runners = {'000':'010', '100':'011', '010':'010', '110':'011', 
                              '001':'010', '101':'011', '011':'010', '111':'011'}
            double_runs = int(runners[1]) + int(runners[2])
            outcomes.append(("0-0", outs, double_runners.get(runners, '010'), 0.10, double_runs))
            
            # 5-4. 홈런 (5% 확률) - 타자 및 모든 주자 득점, 베이스 초기화
            hr_runs = 1 + int(runners[0]) + int(runners[1]) + int(runners[2])
            outcomes.append(("0-0", outs, '000', 0.05, hr_runs))
            
        else:
            # 기타 알 수 없는 결과 (상태 유지)
            outcomes.append((f"{b}-{s}", outs, runners, 1.0, 0))

        # 반환 포맷 구성
        results = []
        for n_cnt, n_outs, n_run, prob, runs in outcomes:
            if n_outs >= 3:
                results.append(("END", prob, runs))
            else:
                # 투수 군집은 전이 과정에서 변하지 않으므로 그대로 유지
                results.append((f"{n_cnt}_{n_outs}_{n_run}_{batter_cluster}_{pitcher_cluster}", prob, runs))

        return results

    def solve_mdp(self):
        """
        2304개 상태에 대해 가치 반복(Value Iteration)을 수행하여
        RE24 기반 실점 억제 기대치가 가장 높은 최적의 행동(구종, 코스)을 계산
        """
        print("8장: RE24 기반 MDP 최적 투구 전략 역순 계산 중...")
        
        counts = ["3-2", "2-2", "3-1", "1-2", "2-1", "3-0", "0-2", "1-1", "2-0", "0-1", "1-0", "0-0"]
        outs_list = ["2", "1", "0"]
        runners = ["111", "011", "101", "110", "001", "010", "100", "000"]
        batter_clusters = [str(i) for i in range(8)]

        # 상태 공간: 12 × 3 × 8 × 8 × K_pitchers
        # 단일 투수 모드(K=1): 2,304개 / 범용 모드(K=6): 13,824개
        n_total = len(counts) * len(outs_list) * len(runners) * len(batter_clusters) * len(self.pitcher_clusters)
        print(f"총 상태 수: {n_total}개 (투수 군집 K={len(self.pitcher_clusters)})")
        if self.valid_pitches_by_cluster:
            for pc, vp in self.valid_pitches_by_cluster.items():
                print(f"  cluster {pc}: {len(vp)}구종 × {len(self.zones)}존 = {len(vp)*len(self.zones)} actions — {vp}")
        else:
            print(f"  전체: {len(self.pitch_names)}구종 × {len(self.zones)}존 = {len(self.pitch_names)*len(self.zones)} actions")
        states = [
            f"{c}_{o}_{r}_{bc}_{pc}"
            for c, o, r, bc, pc in itertools.product(counts, outs_list, runners, batter_clusters, self.pitcher_clusters)
        ]
        
        # 상태 가치 초기화
        self.state_values = {state: 0.0 for state in states}
        self.optimal_policy = {}
        
        input_df_template = pd.DataFrame(np.zeros((1, len(self.feature_columns))), columns=self.feature_columns)
        
        # Value Iteration: 최대 20회 반복, max|ΔV| < 1e-4이면 조기 종료
        # γ=0.99: foul self-loop에서 가치 무한 누적 방지 (Task 16)
        gamma = 0.99
        max_iterations = 20
        convergence_threshold = 1e-4

        for iteration in range(max_iterations):
            max_delta = 0.0  # 이번 반복의 최대 가치 변화량

            # 역순 탐색 (3-2부터 0-0까지)
            for state in states:
                best_action = None
                best_expected_reward = float('-inf')

                # 5-파트 상태 키 파싱: count_outs_runners_batter_pitcher
                s_parts = state.split('_')
                cur_outs_str = s_parts[1]
                cur_runners = s_parts[2]
                cur_batter_cluster = s_parts[3]
                cur_pitcher_cluster = s_parts[4]
                cur_b, cur_s = map(int, s_parts[0].split('-'))
                cur_outs = int(cur_outs_str)
                current_re24 = self._get_re24(cur_outs, cur_runners)

                # 군집별 유효 구종 필터 적용 (Task 18)
                if self.valid_pitches_by_cluster and cur_pitcher_cluster in self.valid_pitches_by_cluster:
                    pitches_to_try = self.valid_pitches_by_cluster[cur_pitcher_cluster]
                else:
                    pitches_to_try = self.pitch_names

                for pitch in pitches_to_try:
                    for zone in self.zones:
                        input_df = input_df_template.copy()

                        # 수치 피처: 볼카운트/아웃/주자 직접 할당 (B안: count_state one-hot 대체)
                        for col, val in [
                            ('balls',   cur_b),
                            ('strikes', cur_s),
                            ('outs',    cur_outs),
                            ('on_1b',   int(cur_runners[0])),
                            ('on_2b',   int(cur_runners[1])),
                            ('on_3b',   int(cur_runners[2])),
                        ]:
                            if col in input_df.columns: input_df[col] = val

                        # 물리 피처: lookup 테이블에서 (pitcher_cluster, pitch) 기준 채움 (Task 12 Phase 2)
                        phys_key = (cur_pitcher_cluster, pitch)
                        if phys_key in self.physical_lookup:
                            phys = self.physical_lookup[phys_key]
                            for col, val in phys.items():
                                if col in input_df.columns: input_df[col] = val

                        # 카테고리 피처: 구종/존/타자·투수군집 one-hot
                        for col_name, col_val in [
                            (f"mapped_pitch_name_{pitch}",             1),
                            (f"zone_{zone}",                           1),
                            (f"batter_cluster_{cur_batter_cluster}",   1),
                            (f"pitcher_cluster_{cur_pitcher_cluster}", 1),
                        ]:
                            if col_name in input_df.columns: input_df[col_name] = col_val
                        
                        outcome_proba = self.transition_model.predict_proba(input_df)[0]
                        
                        expected_reward = 0.0
                        
                        for outcome_name, model_prob in zip(self.target_classes, outcome_proba):
                            if model_prob == 0: continue
                            
                            next_outcomes = self._get_next_states_and_rewards(state, outcome_name)
                            
                            for next_state_key, transition_prob, runs_scored in next_outcomes:
                                total_prob = model_prob * transition_prob
                                
                                if next_state_key == "END":
                                    next_re24 = 0.0
                                    future_value = 0.0
                                else:
                                    n_parts = next_state_key.split('_')
                                    n_outs_str = n_parts[1]
                                    n_runners = n_parts[2]
                                    next_re24 = self._get_re24(int(n_outs_str), n_runners)
                                    future_value = self.state_values.get(next_state_key, 0.0)
                                    
                                # 보상(Reward) = 현재 RE24 - 다음 RE24 - 실제 실점
                                # 이 값이 양수이고 클수록 투수가 실점을 잘 억제했다는 의미
                                immediate_reward = current_re24 - next_re24 - runs_scored
                                
                                expected_reward += total_prob * (immediate_reward + gamma * future_value)

                        if expected_reward > best_expected_reward:
                            best_expected_reward = expected_reward
                            best_action = (pitch, zone)
                            
                old_value = self.state_values[state]
                self.state_values[state] = best_expected_reward
                max_delta = max(max_delta, abs(best_expected_reward - old_value))
                self.optimal_policy[state] = {
                    'pitch': best_action[0],
                    'zone': best_action[1],
                    'value': best_expected_reward
                }

            print(f"  VI iter {iteration + 1}/{max_iterations}: max|ΔV| = {max_delta:.6f}")
            if max_delta < convergence_threshold:
                print(f"  수렴 완료 (max|ΔV| < {convergence_threshold})")
                break

        print(f"MDP 연산(Value Iteration) {iteration + 1}회 반복 완료! (γ={gamma}, max|ΔV|={max_delta:.6f})")

    def log_policy_to_wandb(self):
        """
        도출된 최적 정책 중 특정 주요 상황을 W&B Table로 로깅
        """
        print("W&B 대시보드에 최적 투구 전략 로깅 중...")
        
        forward_counts = ["0-0", "0-1", "1-0", "0-2", "1-1", "2-0", "1-2", "2-1", "3-0", "2-2", "3-1", "3-2"]
        # 상태 키 형식: count_outs_runners_batter_pitcher
        # 투수 군집은 첫 번째 값(self.pitcher_clusters[0])으로 고정하여 샘플 출력
        pc = self.pitcher_clusters[0]
        situations = [
            ("0아웃 주자 없음 (타자군집 0)", f"0_000_0_{pc}"),
            ("2아웃 만루 (타자군집 2)",      f"2_111_2_{pc}"),
            ("무사 2루 (타자군집 5)",        f"0_010_5_{pc}"),
        ]

        for sit_name, sit_code in situations:
            table = wandb.Table(columns=["Count", "Optimal Pitch", "Optimal Zone", "Run Prevented (Expected)"])

            for count in forward_counts:
                state_key = f"{count}_{sit_code}"
                if state_key in self.optimal_policy:
                    action = self.optimal_policy[state_key]
                    table.add_data(
                        count, 
                        action['pitch'], 
                        int(action['zone']), 
                        round(action['value'], 4)
                    )
            
            if wandb.run:
                wandb.log({f"Optimal_Policy_{sit_name}": table})
                print(f"W&B에 [{sit_name}] 테이블 로깅 완료!")
            else:
                print(f"W&B run이 비활성화됨. [{sit_name}] 전략을 출력합니다.")
                print(table.data)

    def run_optimizer(self):
        """
        MDP 파이프라인(가치 반복 및 로깅) 전체 실행
        """
        self.solve_mdp()
        self.log_policy_to_wandb()
        return self.optimal_policy
