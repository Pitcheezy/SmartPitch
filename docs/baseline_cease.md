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
| Random | +0.1959 | 1.1534 | 0.0365 | [+0.1245, +0.2674] | 1.098 | 7.71 |
| MostFrequent | +0.2199 | 1.1767 | 0.0372 | [+0.1470, +0.2929] | -0.000 | 8.48 |
| Frequency | +0.2329 | 1.1291 | 0.0357 | [+0.1630, +0.3029] | 0.748 | 7.76 |
| MDPPolicy | +0.2509 | 1.0948 | 0.0346 | [+0.1831, +0.3188] | 1.048 | 7.46 |
| DQN (Dylan Cease) | +0.1979 | 1.1769 | 0.0372 | [+0.1250, +0.2709] | 0.567 | 8.29 |

## 구종 분포

| Agent | Fastball | Slider | Changeup |
|-------|------|------|------|
| Random | 33.3% | 34.3% | 32.4% |
| MostFrequent | 100.0% | 0.0% | 0.0% |
| Frequency | 49.3% | 49.5% | 1.2% |
| MDPPolicy | 19.1% | 41.5% | 39.4% |
| DQN (Dylan Cease) | 83.3% | 7.6% | 9.2% |

## 통계적 유의성 검정

Welch's t-test (양측검정, 불등분산), Cohen's d 효과크기

| 비교 | 차이 | t-통계량 | p-value | 유의(α=0.05) | Cohen's d | 효과크기 |
|------|------|---------|---------|-------------|-----------|---------|
| DQN vs Random | +0.0020 | 0.0384 | 0.9694 | No | +0.0017 | negligible |
| DQN vs MostFrequent | -0.0220 | -0.4178 | 0.6761 | No | -0.0187 | negligible |
| DQN vs Frequency | -0.0350 | -0.6783 | 0.4977 | No | -0.0303 | negligible |
| DQN vs MDPPolicy | -0.0530 | -1.0422 | 0.2975 | No | -0.0466 | negligible |

## 재실행

```bash
uv run scripts/evaluate_personal_dqn.py
```
