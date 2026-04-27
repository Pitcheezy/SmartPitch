# Baseline Comparison — SmartPitch

- 환경: `PitchEnv(pitcher_cluster=0)` + 범용 전이 모델
  (`best_transition_model_universal.pth`, 4클래스: ball/foul/hit_into_play/strike)
- 평가: 각 에이전트 **1000** 에피소드, 동일 seed (`0~999`)
- Baseline action space: **104** = 8 구종 × 13 존
  - 구종: Changeup, Curveball, Cutter, Fastball, Sinker, Slider, Splitter, Sweeper
  - 존:   1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14

## Results

| Agent | Mean Reward ± Std | Pitch Entropy | Mean Pitches/Ep | Action Space | Notes |
|---|---|---|---|---|---|
| Random | +0.225 ± 1.179 | 2.079 | 7.36 | 104 |  |
| MostFrequent (Cole 2019) | +0.260 ± 1.180 | -0.000 | 8.48 | 104 |  |
| Frequency (2023 League) | +0.202 ± 1.205 | 1.853 | 7.75 | 104 |  |
| Frequency (Cole 2019) | +0.264 ± 1.140 | 1.243 | 7.78 | 104 |  |
| DQN (Cole 2019 ref) | +0.436 ± 1.255 | — | — | ~52 (Cole 식별 4구종 × 13존) | W&B run h4n3o0di |

## 비고

- **DQN (Cole 2019 ref)**: W&B run `h4n3o0di`의 100 에피소드 평가 결과(평균 보상 +0.436 ± 1.255)이며
  본 스크립트가 직접 재실행한 값이 아닙니다. DQN은 `clustering.PitchClustering`로
  Cole 본인이 던진 구종(보통 4종: Fastball/Slider/Curveball/Changeup)만으로 학습되었으므로
  action space ≈ 4 × 13 = 52로, 본 베이스라인의 117과 다릅니다.
- 베이스라인은 universal 모델의 9개 구종 전체를 후보로 가지므로 탐색 공간이 더 큽니다.
  → DQN과의 직접 비교 시 "DQN은 더 작은 action space에서 학습됨"을 감안해야 합니다.
- `release_speed_n / pfx_x_n / pfx_z_n` 물리 피처는 `PitchEnv._sample_outcome`에서 0으로 입력
  되며, 이는 DQN 평가 시점과 동일한 조건입니다 (공정 비교).

## 재실행

```bash
uv run src/evaluate_baselines.py
```
