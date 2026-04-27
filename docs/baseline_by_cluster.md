# Per-Pitcher-Cluster Baseline Comparison

- 환경: `PitchEnv(pitcher_cluster=K)` + 범용 전이 모델
- 평가: 각 (군집 × 에이전트) **1000** 에피소드, 동일 seed
- Action space: 군집별 유효 구종 × 13 존 (1% 미만 구종 제외)

## 군집 정보

| Cluster | 투구 수 (2023) | 최빈 (pitch / zone) | 특성 (avg 구속 · top3 구종) |
|---|---|---|---|
| 0 | 202,058 | Fastball / Zone 11 | avg 89.9 mph · Fastball 46%, Slider 24%, Changeup 10% |
| 1 | 132,511 | Changeup / Zone 14 | avg 87.8 mph · Fastball 29%, Sinker 20%, Changeup 17% |
| 2 | 117,946 | Slider / Zone 14 | avg 89.5 mph · Sinker 37%, Slider 18%, Fastball 15% |
| 3 | 193,208 | Fastball / Zone 11 | avg 88.8 mph · Fastball 30%, Cutter 17%, Curveball 16% |

## 평가 결과

| Cluster | Agent | Mean ± Std | Entropy | Pitches/Ep |
|---|---|---|---|---|
| 0 | Random | +0.199 ± 1.229 | 2.079 | 7.64 |
| 0 | MostFrequent | +0.275 ± 1.182 | -0.000 | 8.71 |
| 0 | Frequency | +0.241 ± 1.176 | 1.541 | 8.10 |
| 0 | MDPPolicy | +0.300 ± 1.065 | 1.181 | 7.58 |
| 0 | DQN (Cole 2019 ref) | +0.436 ± 1.255 | — | — |
| 1 | Random | +0.189 ± 1.219 | 1.945 | 7.80 |
| 1 | MostFrequent | +0.162 ± 1.231 | -0.000 | 7.03 |
| 1 | Frequency | +0.222 ± 1.228 | 1.798 | 7.83 |
| 1 | MDPPolicy | +0.292 ± 1.089 | 1.031 | 7.66 |
| 1 | DQN | 미학습 | — | — |
| 2 | Random | +0.207 ± 1.213 | 2.079 | 7.98 |
| 2 | MostFrequent | +0.225 ± 1.141 | -0.000 | 7.66 |
| 2 | Frequency | +0.231 ± 1.182 | 1.753 | 8.03 |
| 2 | MDPPolicy | +0.282 ± 1.100 | 1.264 | 8.19 |
| 2 | DQN | 미학습 | — | — |
| 3 | Random | +0.199 ± 1.196 | 2.079 | 7.70 |
| 3 | MostFrequent | +0.288 ± 1.157 | -0.000 | 9.02 |
| 3 | Frequency | +0.242 ± 1.175 | 1.882 | 8.04 |
| 3 | MDPPolicy | +0.282 ± 1.119 | 1.399 | 7.46 |
| 3 | DQN | 미학습 | — | — |

## 비고

- 군집 정의는 `data/pitcher_clusters_2023.csv` (K=4) — `pitcher_clustering.py` 산출물
- MostFrequent의 최빈 조합은 해당 군집 내 2023 시즌 (pitch_type → mapped_pitch_name, zone) value_counts 1위
- Frequency는 동일 군집 내 (pitch, zone) 빈도 분포로 매 step 샘플링
- MDPPolicy는 `MDPOptimizer.solve_mdp()`로 9,216개 상태 전체에 대해 한 번만 풀고
  `data/mdp_optimal_policy.pkl`로 캐시. obs(8D) → state key `"b-s_outs_runners_bc_pc"` 변환 후 lookup.
  모든 군집에서 동일한 정책을 공유하지만 PitchEnv 시뮬레이션 시 `pitcher_cluster`만 달라짐.
- DQN 행: 군집 0(Cole 2019)만 W&B run `h4n3o0di`의 평가값(+0.436 ± 1.255). 군집 1~3은 학습된 적 없음.
- 동일 seed로 환경 reset(`seed=0..999`) → 군집/에이전트 간 공정 비교

## 재실행

```bash
uv run src/evaluate_baselines.py
```
