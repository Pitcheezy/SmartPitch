import pandas as pd
import numpy as np
import wandb
import itertools
from typing import List, Dict, Tuple, Any

class MDPOptimizer:
    """
    TransitionProbabilityModel의 예측 능력을 활용해,
    벨만 방정식(Value Iteration)을 거꾸로 계산하여 최적의 볼배합을 찾아내는 클래스
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
        
        # 야구 통계 기반 보상 체계 (기대 득점 변화를 투수 관점에서 단순화)
        self.reward_system = {
            'strike': 0.1,    # 스트라이크는 유리함
            'ball': -0.1,     # 볼은 불리함
            'foul': 0.05,     # 파울은 약간 유리함 (스트라이크 카운트를 늘리므로)
            'out': 0.5,       # 아웃을 잡으면 매우 유리함
            'hit': -0.5,      # 안타나 사구 등은 매우 불리함
            'walk': -0.3      # 볼넷
        }
        
        self.state_values = {}
        self.optimal_policy = {}

    def _get_next_state(self, current_state: str, outcome: str) -> str:
        """
        [내부 메서드] 현재 상태와 투구 결과를 바탕으로 다음 상태(볼카운트)를 반환
        """
        try:
            count, outs, runners = current_state.split('_')
            b, s = map(int, count.split('-'))
            o = int(outs)
            
            if outcome in ['called_strike', 'swinging_strike', 'foul_tip', 'swinging_strike_blocked', 'missed_bunt']:
                s += 1
                if s >= 3:
                    return 'TERMINAL_OUT'
            elif outcome in ['foul', 'foul_bunt']:
                if s < 2:
                    s += 1
            elif outcome in ['ball', 'blocked_dirt', 'pitchout']:
                b += 1
                if b >= 4:
                    return 'TERMINAL_WALK'
            elif outcome in ['hit_into_play', 'hit_by_pitch']:
                return 'TERMINAL_HIT'
                
            return f"{b}-{s}_{o}_{runners}"
        except Exception:
            # 파싱 실패 시 현재 상태 유지
            return current_state

    def _get_reward(self, outcome: str, next_state: str) -> float:
        """
        [내부 메서드] 투구 결과와 다음 상태에 따른 즉각적인 보상(Reward)을 반환
        """
        # 1. 상태 변화(아웃, 볼넷)로 인한 큰 보상
        if next_state == 'TERMINAL_OUT':
            return self.reward_system['out']
        elif next_state == 'TERMINAL_WALK':
            return self.reward_system['walk']
            
        # 2. 개별 투구 결과에 따른 보상
        if outcome in ['called_strike', 'swinging_strike', 'foul_tip', 'swinging_strike_blocked', 'missed_bunt']:
            return self.reward_system['strike']
        elif outcome in ['ball', 'blocked_dirt', 'pitchout']:
            return self.reward_system['ball']
        elif outcome in ['foul', 'foul_bunt']:
            return self.reward_system['foul']
        elif outcome in ['hit_into_play', 'hit_by_pitch']:
            return self.reward_system['hit']
            
        return 0.0

    def solve_mdp(self):
        """
        288개 상태에 대해 '3-2'부터 '0-0'까지 역순 가치 반복(Value Iteration)을 수행하여
        최적의 행동(구종, 코스)과 가치를 계산
        """
        print("8장: MDP 기반 최적 투구 전략(Optimal Policy) 역순 계산 중...")
        
        # 거꾸로 탐색하기 위해 가장 뒷부분 카운트부터 정렬
        counts = ["3-2", "2-2", "3-1", "1-2", "2-1", "3-0", "0-2", "1-1", "2-0", "0-1", "1-0", "0-0"]
        outs = ["2", "1", "0"]
        runners = ["111", "011", "101", "110", "001", "010", "100", "000"]
        
        # 288개 상태 공간 생성
        states = [f"{c}_{o}_{r}" for c, o, r in itertools.product(counts, outs, runners)]
        
        # 상태 가치 초기화
        self.state_values = {state: 0.0 for state in states}
        self.optimal_policy = {}
        
        # 모델에 던질 1줄짜리 빈 데이터프레임 템플릿 생성
        input_df_template = pd.DataFrame(np.zeros((1, len(self.feature_columns))), columns=self.feature_columns)
        
        for state in states:
            best_action = None
            best_expected_value = float('-inf')
            
            for pitch in self.pitch_names:
                for zone in self.zones:
                    input_df = input_df_template.copy()
                    
                    # 현재 조건에 해당하는 컬럼 찾아서 1로 세팅 (One-Hot)
                    state_col = f"count_state_{state}"
                    pitch_col = f"mapped_pitch_name_{pitch}"
                    zone_col = f"zone_{zone}"
                    
                    if state_col in input_df.columns: input_df[state_col] = 1
                    if pitch_col in input_df.columns: input_df[pitch_col] = 1
                    if zone_col in input_df.columns: input_df[zone_col] = 1
                    
                    # 모델 예측 (추론)
                    outcome_proba = self.transition_model.predict_proba(input_df)[0]
                    
                    # 벨만 방정식: 가치 누적
                    aggregated_value = 0.0
                    for outcome_name, prob in zip(self.target_classes, outcome_proba):
                        if prob == 0:
                            continue
                            
                        next_state = self._get_next_state(state, outcome_name)
                        immediate_reward = self._get_reward(outcome_name, next_state)
                        future_value = self.state_values.get(next_state, 0.0)
                        
                        aggregated_value += prob * (immediate_reward + future_value)
                        
                    # 최고 가치를 주는 행동 갱신
                    if aggregated_value > best_expected_value:
                        best_expected_value = aggregated_value
                        best_action = (pitch, zone)
                        
            # 탐색 완료 후 현재 상태의 가치 및 최적 정책 저장
            self.state_values[state] = best_expected_value
            self.optimal_policy[state] = {
                'pitch': best_action[0],
                'zone': best_action[1],
                'value': best_expected_value
            }
            
        print("MDP 역순 계산 완료! 모든 볼카운트에서의 최적 전략이 도출되었습니다.")

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
            table = wandb.Table(columns=["Count", "Optimal Pitch", "Optimal Zone", "Expected Value"])
            
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
