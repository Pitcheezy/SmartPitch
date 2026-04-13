# Per-Pitcher-Cluster Baseline Comparison

- 환경: `PitchEnv(pitcher_cluster=K)` + 범용 전이 모델
- 평가: 각 (군집 × 에이전트) **1000** 에피소드, 동일 seed
- Action space: 군집별 유효 구종 × 13 존 (1% 미만 구종 제외)

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
| 0 | Random | +0.185 ± 1.177 | 2.079 | 7.36 |
| 0 | MostFrequent | +0.220 ± 1.177 | -0.000 | 8.48 |
| 0 | Frequency | +0.175 ± 1.123 | 1.539 | 7.88 |
| 0 | MDPPolicy | +0.258 ± 1.091 | 1.206 | 7.37 |
| 0 | DQN (Cole 2019 ref) | +0.436 ± 1.255 | — | — |
| 1 | Random | +0.136 ± 1.201 | 1.945 | 7.54 |
| 1 | MostFrequent | +0.140 ± 1.188 | -0.000 | 6.78 |
| 1 | Frequency | +0.169 ± 1.174 | 1.799 | 7.57 |
| 1 | MDPPolicy | +0.247 ± 1.096 | 1.027 | 7.46 |
| 1 | DQN (91 actions) | +0.188 ± 1.127 | — | 7.78 |
| 2 | Random | +0.229 ± 1.130 | 2.079 | 7.66 |
| 2 | MostFrequent | +0.201 ± 1.123 | -0.000 | 7.38 |
| 2 | Frequency | +0.203 ± 1.111 | 1.753 | 7.67 |
| 2 | MDPPolicy | +0.262 ± 1.092 | 1.260 | 7.93 |
| 2 | DQN (104 actions) | +0.242 ± 1.130 | — | 8.17 |
| 3 | Random | +0.171 ± 1.194 | 2.079 | 7.52 |
| 3 | MostFrequent | +0.251 ± 1.128 | -0.000 | 8.74 |
| 3 | Frequency | +0.202 ± 1.158 | 1.882 | 7.70 |
| 3 | MDPPolicy | +0.256 ± 1.072 | 1.395 | 7.22 |
| 3 | DQN (104 actions) | +0.215 ± 1.157 | — | 7.87 |

## 비고

- 군집 정의는 `data/pitcher_clusters_2023.csv` (K=4) — `pitcher_clustering.py` 산출물
- MostFrequent의 최빈 조합은 해당 군집 내 2023 시즌 (pitch_type → mapped_pitch_name, zone) value_counts 1위
- Frequency는 동일 군집 내 (pitch, zone) 빈도 분포로 매 step 샘플링
- MDPPolicy는 `MDPOptimizer.solve_mdp()`로 9,216개 상태 전체에 대해 한 번만 풀고
  `data/mdp_optimal_policy.pkl`로 캐시. obs(8D) → state key `"b-s_outs_runners_bc_pc"` 변환 후 lookup.
  모든 군집에서 동일한 정책을 공유하지만 PitchEnv 시뮬레이션 시 `pitcher_cluster`만 달라짐.
- DQN 행: 군집 0(Cole 2019)은 W&B run `h4n3o0di`의 평가값(+0.436 ± 1.255, ~52 actions).
  군집 1~3은 `scripts/train_dqn_all_clusters.py`로 300K 스텝 학습, 1000 에피소드 평가.
  군집별 유효 구종만 사용: 군집 1=91 actions (7구종), 군집 2~3=104 actions (8구종).
  Knuckleball(< 1%) 및 Splitter(군집 1에서 0.3%) 제외.
- 동일 seed로 환경 reset(`seed=0..999`) → 군집/에이전트 간 공정 비교

## 재실행

```bash
uv run src/evaluate_baselines.py
```
