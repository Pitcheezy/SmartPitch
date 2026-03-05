import pandas as pd
import numpy as np
import wandb
import itertools
from typing import List, Dict, Tuple, Any

class MDPOptimizer:
    """
    TransitionProbabilityModel의 예측 능력을 활용해,
    벨만 방정식(Value Iteration)을 거꾸로 계산하여 최적의 볼배합을 찾아내는 클래스 (RE24 기반)
    """
    
    def __init__(self, transition_model, feature_columns: List[str], target_classes: List[str], pitch_names: List[str], zones: List[float]):
        """
        초기화 메서드
        :param transition_model: 학습된 TransitionProbabilityModel 객체 (predict_proba 메서드 제공)
        :param feature_columns: 모델 입력에 사용된 One-Hot Encoding 컬럼 리스트
        :param target_classes: 모델 출력의 클래스(결과) 리스트
        :param pitch_names: 클러스터링으로 식별된 구종 이름 리스트
        :param zones: 투구 코스(존) 리스트
        """
        self.transition_model = transition_model
        self.feature_columns = feature_columns
        self.target_classes = target_classes
        self.pitch_names = pitch_names
        self.zones = zones
        
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
            count, outs_str, runners = current_state.split('_')
            b, s = map(int, count.split('-'))
            outs = int(outs_str)
        except Exception:
            return [(current_state, 1.0, 0.0)]

        outcomes = [] # (next_count, next_outs, next_runners, prob, runs)
        
        # 1. 스트라이크 계열 (스트라이크 판정, 헛스윙 등)
        if outcome in ['called_strike', 'swinging_strike', 'foul_tip', 'swinging_strike_blocked', 'missed_bunt']:
            s += 1
            if s >= 3:
                # 삼진 아웃: 카운트 리셋, 아웃 증가
                outcomes.append(("0-0", outs + 1, runners, 1.0, 0))
            else:
                outcomes.append((f"{b}-{s}", outs, runners, 1.0, 0))
                
        # 2. 파울
        elif outcome in ['foul', 'foul_bunt']:
            if s < 2:
                s += 1
            # 2스트라이크 이후 파울은 카운트 유지
            outcomes.append((f"{b}-{s}", outs, runners, 1.0, 0))
            
        # 3. 볼 계열
        elif outcome in ['ball', 'blocked_dirt', 'pitchout']:
            b += 1
            if b >= 4:
                # 볼넷: 카운트 리셋, 주자 진루
                next_runners, runs = self._advance_runners_walk(runners)
                outcomes.append(("0-0", outs, next_runners, 1.0, runs))
            else:
                outcomes.append((f"{b}-{s}", outs, runners, 1.0, 0))
                
        # 4. 몸에 맞는 볼 (HBP)
        elif outcome == 'hit_by_pitch':
            # 사구: 카운트 리셋, 주자 진루
            next_runners, runs = self._advance_runners_walk(runners)
            outcomes.append(("0-0", outs, next_runners, 1.0, runs))
            
        # 5. 인플레이 타구 (단순화된 확률적 분기)
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
                results.append((f"{n_cnt}_{n_outs}_{n_run}", prob, runs))
                
        return results

    def solve_mdp(self):
        """
        288개 상태에 대해 가치 반복(Value Iteration)을 수행하여
        RE24 기반 실점 억제 기대치가 가장 높은 최적의 행동(구종, 코스)을 계산
        """
        print("8장: RE24 기반 MDP 최적 투구 전략 역순 계산 중...")
        
        counts = ["3-2", "2-2", "3-1", "1-2", "2-1", "3-0", "0-2", "1-1", "2-0", "0-1", "1-0", "0-0"]
        outs = ["2", "1", "0"]
        runners = ["111", "011", "101", "110", "001", "010", "100", "000"]
        
        # 288개 상태 공간 생성
        states = [f"{c}_{o}_{r}" for c, o, r in itertools.product(counts, outs, runners)]
        
        # 상태 가치 초기화
        self.state_values = {state: 0.0 for state in states}
        self.optimal_policy = {}
        
        input_df_template = pd.DataFrame(np.zeros((1, len(self.feature_columns))), columns=self.feature_columns)
        
        # 파울 시 카운트가 유지되는 사이클 구조를 해결하기 위해 Value Iteration을 5회 반복하여 수렴시킴
        for iteration in range(5):
            # 역순 탐색 (3-2부터 0-0까지)
            for state in states:
                best_action = None
                best_expected_reward = float('-inf')
                
                _, cur_outs_str, cur_runners = state.split('_')
                current_re24 = self._get_re24(int(cur_outs_str), cur_runners)
                
                for pitch in self.pitch_names:
                    for zone in self.zones:
                        input_df = input_df_template.copy()
                        
                        state_col = f"count_state_{state}"
                        pitch_col = f"mapped_pitch_name_{pitch}"
                        zone_col = f"zone_{zone}"
                        
                        if state_col in input_df.columns: input_df[state_col] = 1
                        if pitch_col in input_df.columns: input_df[pitch_col] = 1
                        if zone_col in input_df.columns: input_df[zone_col] = 1
                        
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
                                    _, n_outs_str, n_runners = next_state_key.split('_')
                                    next_re24 = self._get_re24(int(n_outs_str), n_runners)
                                    future_value = self.state_values.get(next_state_key, 0.0)
                                    
                                # 보상(Reward) = 현재 RE24 - 다음 RE24 - 실제 실점
                                # 이 값이 양수이고 클수록 투수가 실점을 잘 억제했다는 의미
                                immediate_reward = current_re24 - next_re24 - runs_scored
                                
                                expected_reward += total_prob * (immediate_reward + future_value)
                                
                        if expected_reward > best_expected_reward:
                            best_expected_reward = expected_reward
                            best_action = (pitch, zone)
                            
                self.state_values[state] = best_expected_reward
                self.optimal_policy[state] = {
                    'pitch': best_action[0],
                    'zone': best_action[1],
                    'value': best_expected_reward
                }
                
        print("MDP 연산(Value Iteration) 5회 반복 완료 및 최적 정책 수렴 완료!")

    def log_policy_to_wandb(self):
        """
        도출된 최적 정책 중 특정 주요 상황을 W&B Table로 로깅
        """
        print("W&B 대시보드에 최적 투구 전략 로깅 중...")
        
        forward_counts = ["0-0", "0-1", "1-0", "0-2", "1-1", "2-0", "1-2", "2-1", "3-0", "2-2", "3-1", "3-2"]
        situations = [
            ("0아웃 주자 없음", "0_000"),
            ("2아웃 만루", "2_111")
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
