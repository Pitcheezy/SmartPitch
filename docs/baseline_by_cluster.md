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
| 0 | Random | +0.225 ± 1.179 | 2.079 | 7.36 |
| 0 | MostFrequent | +0.260 ± 1.180 | -0.000 | 8.48 |
| 0 | Frequency | +0.257 ± 1.170 | 1.541 | 7.69 |
| 0 | MDPPolicy | +0.298 ± 1.094 | 1.183 | 7.37 |
| 0 | DQN (Cole 2019 ref) | +0.436 ± 1.255 | — | — |
| 1 | Random | +0.176 ± 1.204 | 1.945 | 7.54 |
| 1 | MostFrequent | +0.180 ± 1.190 | -0.000 | 6.78 |
| 1 | Frequency | +0.209 ± 1.176 | 1.799 | 7.57 |
| 1 | MDPPolicy | +0.286 ± 1.099 | 1.038 | 7.47 |
| 1 | DQN | 미학습 | — | — |
| 2 | Random | +0.269 ± 1.134 | 2.079 | 7.66 |
| 2 | MostFrequent | +0.241 ± 1.126 | -0.000 | 7.38 |
| 2 | Frequency | +0.243 ± 1.114 | 1.753 | 7.67 |
| 2 | MDPPolicy | +0.300 ± 1.095 | 1.262 | 7.94 |
| 2 | DQN | 미학습 | — | — |
| 3 | Random | +0.211 ± 1.196 | 2.079 | 7.52 |
| 3 | MostFrequent | +0.291 ± 1.130 | -0.000 | 8.74 |
| 3 | Frequency | +0.242 ± 1.161 | 1.882 | 7.70 |
| 3 | MDPPolicy | +0.296 ± 1.075 | 1.395 | 7.22 |
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
