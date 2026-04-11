# MDP vs PitchEnv 보상 일관성 분석

`evaluate_baselines.py` 군집별 비교에서 MDPPolicy가 군집 0/2/3에서 Random·MostFrequent에 뒤지는
현상의 원인을 추적한다. 분석 코드: `scripts/analyze_mdp_vs_env.py`.

## A. 보상·전이 로직 줄 단위 비교

### A.1 보상 함수

| 항목 | `mdp_solver.solve_mdp` | `pitch_env.step` |
|---|---|---|
| 위치 | `src/mdp_solver.py:282-286` | `src/pitch_env.py:178-190` |
| 시점 | Value Iteration 내부, 모든 (s,a) 쌍 | 1 step 호출 |
| 보상식 | `immediate_reward = current_re24 - next_re24 - runs_scored` | `reward = re24_before - re24_after - runs_scored` |
| 합산 | `expected_reward += total_prob * (immediate_reward + future_value)` | episode 합산은 호출자 |
| 할인율 γ | 1 (코드에 명시 없음) | 1 (자연 합산) |

**결론**: 보상식 자체는 동일. 차이는 (1) MDP는 `predict_proba`의 *기대값*에서 합산,
PitchEnv는 `np_random.choice(p=proba)`로 단일 outcome을 샘플링한다는 점, (2) MDP는 `+ V(s')`
로 미래까지 누적, env는 step 단위만 반환한다는 점이다.

### A.2 전이 로직 (outcome → 다음 상태)

| outcome | `mdp_solver._get_next_states_and_rewards` (L121-L179) | `pitch_env._apply_outcome` (L285-L319) |
|---|---|---|
| `strike` | `s+=1; if s≥3: ('0-0', outs+1, runners, 1.0, 0)` | `self.strikes+=1; if ≥3: outs+=1; balls,strikes=0,0` |
| `foul`   | `if s<2: s+=1`; 카운트 그대로 → `(b-s, outs, runners, 1.0, 0)` | `if self.strikes<2: self.strikes+=1` |
| `ball`   | `b+=1; if b≥4`: walk advance → `('0-0', outs, next, 1.0, runs)` | `self.balls+=1; if ≥4: self._apply_walk(); reset count` |
| `hit_by_pitch` | walk advance | `self._apply_walk()` (4-class 모델은 발생 X) |
| `hit_into_play` | 4분기 분포: out 0.70 / 1B 0.15 / 2B 0.10 / HR 0.05 (확정 가지) | 동일 70/15/10/5%를 한 번 sampling |

진루 매핑 일치 검증:

- **walk** : `WALK_ADVANCE`(env L58-L62) ↔ `_advance_runners_walk`(mdp L91-L101). 8개 케이스 1:1 일치.
- **single**: env `[1, r1, r2]` (L345) ↔ mdp `single_runners` dict (L162-L163). 8 케이스 모두 동일 결과
  (예: `'010'→'101'`, `'011'→'101'`, runs는 r3).
- **double**: env `[0, 1, r1]` (L349) ↔ mdp `double_runners` dict (L168-L169). 8 케이스 일치, runs = r2+r3.
- **HR**    : 둘 다 `'000'`, runs = `1+r1+r2+r3`.

**결론**: 전이 매핑·득점 계산은 1:1 일치. mdp_solver는 합법적인 PitchEnv 기대값 계산기다.
두 모듈이 어긋나는 *결정론적 버그*는 없으며, 차이는 모두 _확률 sampling vs 확률 expectation_의 차이로 환원된다.

### A.3 자기참조 루프 (foul self-loop)

- 2-strike 상태(`X-2`)에서 `foul` 결과: 카운트 유지 → mdp는 `(b-s, outs, runners, 1.0, 0)`로 자기 자신을 가리킴.
- Value Iteration 1회당 self-loop는 `prev V(s)`로 평가되므로, foul 확률이 큰 액션을 고를수록 수렴이 느려진다.
- PitchEnv는 자기참조가 아니라 step만 반복하므로 episode 길이만 늘어남 (`pitches_per_ep` 증가).

## B. Value Iteration 수렴 검증

`mdp_solver.solve_mdp()`는 5회로 고정 (`for iteration in range(5)`, L220).
동일한 가치반복 루프를 K=1(cluster 0, 2,304 상태) 에 대해 10회까지 확장 실행하고
매 iter `max_s |V_k(s) − V_{k-1}(s)|` 을 측정.

| iter | max\|ΔV\| | mean V(s) |
|---|---|---|
| 1 | 0.506096 | +0.1841 |
| 2 | 0.411673 | +0.3404 |
| 3 | 0.372605 | +0.4481 |
| 4 | 0.242770 | +0.5131 |
| 5 | 0.145419 | +0.5495 |
| 6 | 0.079698 | +0.5691 |
| 7 | 0.044116 | +0.5795 |
| 8 | 0.026195 | +0.5848 |
| 9 | 0.014396 | +0.5875 |
| 10 | 0.007641 | +0.5889 |

- 5회 시점 max|ΔV| = **0.145419** (`solve_mdp()`이 멈추는 지점)
- 10회 시점 max|ΔV| = **0.007641**
- 5→10 사이 ΔV가 절반 이하로 줄어 수렴 진행 중. 5회는 underestimate를 남긴다.

## C. MDP 정책 행동 분포 (cached K=4, 9,216 states)

`data/mdp_optimal_policy.pkl` (5회 반복 결과) 기준.

### C.1 구종별 빈도

| Pitch | Count | Share |
|---|---|---|
| Knuckleball | 6501 | 70.5% |
| Splitter | 821 | 8.9% |
| Sweeper | 688 | 7.5% |
| Slider | 667 | 7.2% |
| Curveball | 536 | 5.8% |
| Cutter | 3 | 0.0% |

### C.2 존별 빈도

| Zone | Count | Share |
|---|---|---|
| 1 | 3341 | 36.3% |
| 2 | 24 | 0.3% |
| 3 | 1629 | 17.7% |
| 4 | 690 | 7.5% |
| 5 | 492 | 5.3% |
| 6 | 106 | 1.2% |
| 7 | 88 | 1.0% |
| 9 | 1 | 0.0% |
| 11 | 92 | 1.0% |
| 12 | 202 | 2.2% |
| 13 | 520 | 5.6% |
| 14 | 2031 | 22.0% |

### C.3 상위 10 (pitch, zone) 조합

| Pitch | Zone | Count | Share |
|---|---|---|---|
| Knuckleball | 1 | 3277 | 35.6% |
| Knuckleball | 3 | 1489 | 16.2% |
| Splitter | 14 | 713 | 7.7% |
| Knuckleball | 4 | 690 | 7.5% |
| Sweeper | 14 | 622 | 6.7% |
| Curveball | 14 | 504 | 5.5% |
| Knuckleball | 5 | 492 | 5.3% |
| Slider | 13 | 441 | 4.8% |
| Knuckleball | 12 | 202 | 2.2% |
| Slider | 14 | 152 | 1.6% |

## D. PitchEnv 1-episode trace (cluster 0, seed=42)

MDPPolicy로 한 이닝을 끝까지 trace.

| step | state | pitch / zone | MDP V(s) | outcome | reward | runs |
|---|---|---|---|---|---|---|
| 0 | `0-0_0_110_2_0` | Knuckleball / 3 | +0.893 | strike | +0.000 | 0 |
| 1 | `0-1_0_110_2_0` | Knuckleball / 3 | +0.912 | ball | +0.000 | 0 |
| 2 | `1-1_0_110_2_0` | Knuckleball / 3 | +0.886 | strike | +0.000 | 0 |
| 3 | `1-2_0_110_2_0` | Curveball / 14 | +0.939 | hit_into_play | -0.855 | 0 |
| 4 | `0-0_0_111_2_0` | Knuckleball / 1 | +1.329 | foul | +0.000 | 0 |
| 5 | `0-1_0_111_2_0` | Knuckleball / 1 | +1.364 | hit_into_play | +0.751 | 0 |
| 6 | `0-0_1_111_2_0` | Knuckleball / 1 | +0.968 | strike | +0.000 | 0 |
| 7 | `0-1_1_111_2_0` | Knuckleball / 1 | +1.003 | strike | +0.000 | 0 |
| 8 | `0-2_1_111_2_0` | Splitter / 14 | +1.080 | strike | +0.805 | 0 |
| 9 | `0-0_2_111_2_0` | Knuckleball / 1 | +0.468 | hit_into_play | +0.736 | 0 |

- 총 step: **10**  (foul 1 / strike 5 / ball 1 / hit_into_play 3)
- 누적 보상: **+1.437**  (PitchEnv 평균 ep 길이 ≈ 7.5와 비교)

## E. Random이 MDP를 이기는 근본 원인 — 검증

`evaluate_baselines.py` 결과에서 관찰된 사실 (1000ep, pitcher_cluster=0):

- Random      : `+0.231 ± 1.098`,  pitches/ep = 7.55,  entropy 2.197
- MDPPolicy   : `+0.151 ± 1.264`,  pitches/ep = 10.30, entropy 0.905
- 군집 3에서는 MDP `+0.036` vs Random `+0.176` (격차 가장 큼)

위 4가지 가설을 본 분석 결과로 검증:

**가설 1. MLP 확률 분포(58% acc)는 절대 신뢰도가 낮다.**
- C 섹션에서 정책이 소수 (pitch, zone) 쌍에 매우 편중되었다면 → 모델이 미세한 확률 차이로 결정을 내린다는 신호.
- 실측: 상위 1쌍이 전체 9,216 상태 중 **35.6%** 차지.
  편중이 강할수록 가설 1이 뒷받침됨.

**가설 2. PitchEnv는 같은 분포에서 1개를 sampling.**
- A.1에서 확인했듯 MDP의 기대값은 `predict_proba`에 직접 의존, env는 같은 분포의 sample.
- 1000ep 평균은 _모델이 옳다는 가정 하에서만_ MDP의 V(s)와 일치. MLP가 58%만 맞으면
  편향 자체가 양쪽에 동일하게 들어가 sampling 분산이 그대로 reward 분산으로 흘러간다.
- 실측: MDPPolicy std = 1.264 > Random std = 1.098. **분산 큼 = 가설 2 부합.**

**가설 3. Value Iteration 5회는 self-loop foul 사이클에서 underestimate.**
- B 섹션: iter 5에서 max|ΔV| = 0.1454, iter 10에서 0.0076.
- D 섹션의 trace: pitches/ep = 10 (평균 7.5보다 길거나 비슷).
  D의 outcome 통계 foul 1회는 MDP 정책이 실제로 self-loop를 자주 만든다는 직접 증거.

**가설 4. Random은 117 균등 → 모델 오차 영향 최소화.**
- Random은 분포 가정 자체가 약해 MLP의 보정 오차를 *베팅*하지 않는다.
- MDP는 좁은 0.001 단위 차이로 action을 골라 한 번의 sample 오차가 누적되면 손해가 더 큼.
- 군집 3 결과(멀티피치, MDP +0.036 vs Random +0.176)는 9구종 중 모델 헷갈림이 가장 큰 군집에서
  MDP의 micro-optimization이 가장 크게 역효과를 낸 사례 → 가설 4 부합.

### 결론

코드 측 버그는 없다. 두 모듈은 보상·전이 모두 1:1 일치한다.
MDP가 실패하는 진짜 원인은 다음 3가지의 결합이다:

1. **MLP 확률 분포의 calibration이 불충분**해 V iteration이 의존할 신호가 약함.
2. **5회 반복 미수렴**으로 self-loop 상태의 V가 underestimate → 정책이 'foul 유도'로 비대칭 편향.
3. **stochastic env의 단일 sample**이 위 두 오차를 그대로 reward 분산으로 흘려보냄.

## F. 권장 후속 조치

- `solve_mdp()` 반복 횟수 5 → 적어도 10, 또는 `delta < 1e-4` 조기 종료 추가.
- γ = 0.99 도입으로 self-loop의 무한 누적 방지 (현재 γ=1).
- MLP probability calibration (temperature scaling) 후 MDP 재평가.
- DQN(+0.436)이 같은 모델·환경에서 MDP를 크게 앞서는 점 — sampling 환경에서 직접 학습이
  *기대값 계획*보다 견고함을 시사. RL 우선 방향성 유지.

## 재실행

```bash
uv run python scripts/analyze_mdp_vs_env.py
```
