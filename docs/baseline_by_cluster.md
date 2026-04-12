# Per-Pitcher-Cluster Baseline Comparison

- 환경: `PitchEnv(pitcher_cluster=K)` + 범용 전이 모델
- 평가: 각 (군집 × 에이전트) **1000** 에피소드, 동일 seed
- Action space: 9 구종 × 13 존 = 117

## 군집 정보

| Cluster | 투구 수 (2023) | 최빈 (pitch / zone) | 특성 (avg 구속 · top3 구종) |
|---|---|---|---|
| 0 | 202,058 | Fastball / Zone 11 | avg 89.9 mph · Fastball 46%, Slider 25%, Changeup 10% |
| 1 | 132,511 | Changeup / Zone 14 | avg 87.8 mph · Fastball 29%, Sinker 20%, Changeup 17% |
| 2 | 117,946 | Slider / Zone 14 | avg 89.5 mph · Sinker 37%, Slider 18%, Fastball 15% |
| 3 | 193,208 | Fastball / Zone 11 | avg 88.8 mph · Fastball 30%, Cutter 17%, Curveball 16% |

## 평가 결과

| Cluster | Agent | Mean ± Std | Entropy | Pitches/Ep |
|---|---|---|---|---|
| 0 | Random | +0.204 ± 1.156 | 2.197 | 7.48 |
| 0 | MostFrequent | +0.220 ± 1.177 | -0.000 | 8.48 |
| 0 | Frequency | +0.175 ± 1.123 | 1.539 | 7.88 |
| 0 | MDPPolicy | +0.250 ± 1.093 | 1.291 | 7.46 |
| 0 | DQN (Cole 2019 ref) | +0.436 ± 1.255 | — | — |
| 1 | Random | +0.197 ± 1.162 | 2.197 | 7.57 |
| 1 | MostFrequent | +0.140 ± 1.188 | -0.000 | 6.78 |
| 1 | Frequency | +0.141 ± 1.175 | 1.813 | 7.57 |
| 1 | MDPPolicy | +0.230 ± 1.178 | 0.418 | 10.77 |
| 1 | DQN (117 actions) | +0.255 ± 1.154 | — | 8.96 |
| 2 | Random | +0.209 ± 1.190 | 2.197 | 7.91 |
| 2 | MostFrequent | +0.201 ± 1.123 | -0.000 | 7.38 |
| 2 | Frequency | +0.211 ± 1.101 | 1.761 | 7.67 |
| 2 | MDPPolicy | +0.260 ± 1.101 | 1.369 | 8.00 |
| 2 | DQN (117 actions) | +0.184 ± 1.214 | — | 8.80 |
| 3 | Random | +0.203 ± 1.145 | 2.197 | 7.86 |
| 3 | MostFrequent | +0.251 ± 1.128 | -0.000 | 8.74 |
| 3 | Frequency | +0.202 ± 1.158 | 1.882 | 7.70 |
| 3 | MDPPolicy | +0.048 ± 1.439 | 0.611 | 12.41 |
| 3 | DQN (117 actions) | +0.242 ± 1.134 | — | 10.53 |

## 비고

- 군집 정의는 `data/pitcher_clusters_2023.csv` (K=4) — `pitcher_clustering.py` 산출물
- MostFrequent의 최빈 조합은 해당 군집 내 2023 시즌 (pitch_type → mapped_pitch_name, zone) value_counts 1위
- Frequency는 동일 군집 내 (pitch, zone) 빈도 분포로 매 step 샘플링
- MDPPolicy는 `MDPOptimizer.solve_mdp()`로 9,216개 상태 전체에 대해 한 번만 풀고
  `data/mdp_optimal_policy.pkl`로 캐시. obs(8D) → state key `"b-s_outs_runners_bc_pc"` 변환 후 lookup.
  모든 군집에서 동일한 정책을 공유하지만 PitchEnv 시뮬레이션 시 `pitcher_cluster`만 달라짐.
- DQN 행: 군집 0(Cole 2019)은 W&B run `h4n3o0di`의 평가값(+0.436 ± 1.255, action space ~52).
  군집 1~3은 범용 모델 기반 9구종×13존=117 action space로 학습(300K 타임스텝).
  **action space 차이 주의**: 군집 0 DQN은 Cole 본인 4구종만 사용(~52), 군집 1~3은 전체 9구종(117).
- 군집 1~3 DQN 모두 Knuckleball 편중 (36~58%) — MLP calibration 이슈.
  실제 MLB에서 Knuckleball 투수는 극소수이나, MLP가 Knuckleball에 대해 strike/foul 확률을
  과대추정하여 DQN이 이를 최적으로 학습함. 향후 MLP 개선 또는 action masking 필요.
- 동일 seed로 환경 reset(`seed=0..999`) → 군집/에이전트 간 공정 비교

## 재실행

```bash
uv run src/evaluate_baselines.py
```
