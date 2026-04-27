# Baseline Comparison — Dylan Cease

- 투수: **Dylan Cease** (MLBAM 656302, 군집 0)
- 학습 데이터: 2024+2025 (5,400 투구)
- 환경: `PitchEnv(pitcher_cluster=0)` + 범용 전이 모델
- 평가: 각 에이전트 **1000** 에피소드, 동일 seed (`0~999`)
- Action space: **39** = 3구종 × 13존
  - 구종: Fastball, Slider, Changeup

## Results

| Agent | Mean Reward | Std | SEM | 95% CI | Pitch Entropy | Pitches/Ep |
|-------|------------|-----|-----|--------|---------------|------------|
| Random | +0.2355 | 1.1558 | 0.0365 | [+0.1639, +0.3072] | 1.098 | 7.71 |
| MostFrequent | +0.2595 | 1.1796 | 0.0373 | [+0.1864, +0.3326] | -0.000 | 8.48 |
| Frequency | +0.2725 | 1.1325 | 0.0358 | [+0.2023, +0.3427] | 0.748 | 7.76 |
| MDPPolicy | +0.2905 | 1.0970 | 0.0347 | [+0.2225, +0.3585] | 1.053 | 7.47 |
| DQN (Dylan Cease) | +0.2375 | 1.1802 | 0.0373 | [+0.1644, +0.3107] | 0.567 | 8.29 |

## 구종 분포

| Agent | Fastball | Slider | Changeup |
|-------|------|------|------|
| Random | 33.3% | 34.3% | 32.4% |
| MostFrequent | 100.0% | 0.0% | 0.0% |
| Frequency | 49.3% | 49.5% | 1.2% |
| MDPPolicy | 19.8% | 40.6% | 39.6% |
| DQN (Dylan Cease) | 83.3% | 7.6% | 9.2% |

## 통계적 유의성 검정

Welch's t-test (양측검정, 불등분산), Cohen's d 효과크기

| 비교 | 차이 | t-통계량 | p-value | 유의(α=0.05) | Cohen's d | 효과크기 |
|------|------|---------|---------|-------------|-----------|---------|
| DQN vs Random | +0.0020 | 0.0383 | 0.9695 | No | +0.0017 | negligible |
| DQN vs MostFrequent | -0.0220 | -0.4167 | 0.6769 | No | -0.0186 | negligible |
| DQN vs Frequency | -0.0350 | -0.6763 | 0.4989 | No | -0.0303 | negligible |
| DQN vs MDPPolicy | -0.0530 | -1.0396 | 0.2986 | No | -0.0465 | negligible |

## 재실행

```bash
uv run scripts/evaluate_personal_dqn.py
```
