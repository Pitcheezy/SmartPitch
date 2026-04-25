# Baseline Comparison — Zac Gallen

- 투수: **Zac Gallen** (MLBAM 668678, 군집 0)
- 학습 데이터: 2024+2025 (4,572 투구)
- 환경: `PitchEnv(pitcher_cluster=0)` + 범용 전이 모델
- 평가: 각 에이전트 **1000** 에피소드, 동일 seed (`0~999`)
- Action space: **52** = 4구종 × 13존
  - 구종: Fastball, Slider, Changeup, Curveball

## Results

| Agent | Mean Reward | Std | SEM | 95% CI | Pitch Entropy | Pitches/Ep |
|-------|------------|-----|-----|--------|---------------|------------|
| Random | +0.2239 | 1.1319 | 0.0358 | [+0.1538, +0.2941] | 1.386 | 7.44 |
| MostFrequent | +0.2199 | 1.1767 | 0.0372 | [+0.1470, +0.2929] | -0.000 | 8.48 |
| Frequency | +0.2039 | 1.1715 | 0.0370 | [+0.1313, +0.2766] | 1.224 | 7.85 |
| MDPPolicy | +0.2359 | 1.1345 | 0.0359 | [+0.1656, +0.3063] | 1.292 | 7.66 |
| DQN (Zac Gallen) | +0.2389 | 1.1344 | 0.0359 | [+0.1686, +0.3093] | 1.304 | 7.75 |

## 구종 분포

| Agent | Fastball | Slider | Changeup | Curveball |
|-------|------|------|------|------|
| Random | 25.1% | 25.3% | 25.3% | 24.3% |
| MostFrequent | 100.0% | 0.0% | 0.0% | 0.0% |
| Frequency | 48.3% | 11.3% | 13.8% | 26.5% |
| MDPPolicy | 15.9% | 41.1% | 13.9% | 29.0% |
| DQN (Zac Gallen) | 35.7% | 17.4% | 13.0% | 33.9% |

## 통계적 유의성 검정

Welch's t-test (양측검정, 불등분산), Cohen's d 효과크기

| 비교 | 차이 | t-통계량 | p-value | 유의(α=0.05) | Cohen's d | 효과크기 |
|------|------|---------|---------|-------------|-----------|---------|
| DQN vs Random | +0.0150 | 0.2959 | 0.7674 | No | +0.0132 | negligible |
| DQN vs MostFrequent | +0.0190 | 0.3674 | 0.7134 | No | +0.0164 | negligible |
| DQN vs Frequency | +0.0350 | 0.6784 | 0.4976 | No | +0.0304 | negligible |
| DQN vs MDPPolicy | +0.0030 | 0.0591 | 0.9529 | No | +0.0026 | negligible |

## 재실행

```bash
uv run scripts/evaluate_personal_dqn.py
```
