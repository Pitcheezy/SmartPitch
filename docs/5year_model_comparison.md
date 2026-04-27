# 5년 통합 모델 비교 결과 (Task 21+23+25)

> 생성일: 2026-04-28
> 데이터: 2021~2025 MLB 5시즌 정규시즌 (3,551,921건)

## 모델 비교

| 모델 | val_acc | macro F1 | Brier | ECE | Brier(cal) | ECE(cal) | T | 피처수 |
|------|---------|----------|-------|-----|------------|----------|---|--------|
| Exp5 (2023, baseline) | 57.5% | 0.495 | — | — | — | — | — | 43 |
| 5Year_MLP_Base | 57.4% | 0.497 | 0.1411 | 0.0496 | 0.1467 | 0.0714 | 1.45 | 43 |
| 5Year_MLP_Extended | **57.9%** | **0.509** | 0.1392 | 0.0480 | 0.1448 | 0.0707 | 1.45 | 50 |
| LightGBM_5Year | **58.7%** | 0.497 | **0.1363** | **0.0025** | — | — | 1.00 | 18 |

## 확장 피처 (MLP Extended, +7개)

| 피처 | 정규화 | 설명 |
|------|--------|------|
| `spin_rate_n` | (release_spin_rate - 2200) / 500 | 회전수 |
| `spin_axis_n` | spin_axis / 360 | 회전축 |
| `release_pos_x_n` | (release_pos_x + 2.0) / 2.0 | 릴리스 좌우 위치 |
| `release_pos_z_n` | (release_pos_z - 6.0) / 1.0 | 릴리스 높이 |
| `platoon_advantage` | 0/1 (좌투-우타 or 우투-좌타) | 플래툰 이점 |
| `p_throws_L` | 0/1 | 좌투 여부 |
| `stand_L` | 0/1 | 좌타 여부 |

## 분석

### MLP Base vs Extended
- 확장 피처 추가로 val_acc +0.4%p, macro F1 +0.012
- **foul recall: 0.242 → 0.280** (+3.8%p) — 확장 피처의 주 효과
- Top-2 Accuracy: 80.1% → 80.9%, Top-3: 94.9% → 95.2%

### LightGBM
- 18개 피처로 가장 높은 val_acc (58.7%) 달성
- **ECE 0.0025**: MLP(0.048) 대비 ~20배 나은 calibration → 확률 예측이 현실에 더 가까움
- Temperature Scaling 불필요 (T=1.0)
- 단, macro F1은 MLP Extended(0.509)보다 낮음(0.497) — 소수 클래스 recall은 MLP Extended가 우위

### Temperature Scaling 결과
- MLP에서 T=1.45로 수렴 — 모델이 과신(overconfident)하지 않으나 calibration 개선 효과 미미
- Brier/ECE가 오히려 소폭 악화 → MLP의 경우 Temperature Scaling 비추천
- LightGBM은 이미 잘 calibrated되어 T=1.0 유지

## 결론

1. **LightGBM이 accuracy/calibration 모두 최고** — 실전 배포 시 우선 고려
2. **MLP Extended가 macro F1 최고** (0.509) — 소수 클래스 균형 중시 시 선택
3. 5년 확장은 MLP에서 미미한 개선, LightGBM에서 의미있는 개선
4. Temperature Scaling은 이 데이터셋에서 비효과적
5. 기존 발표 데모 모델(Cease/Gallen)은 영향 없음 — 별도 파이프라인
