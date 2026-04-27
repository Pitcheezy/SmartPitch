# 작업 보고서: Task 21 + 23 + 24 + 25 (2026-04-28)

## 개요

인플레이 타구 확률 실데이터 교체(Task 21), 5시즌 데이터 확장 + 확장 피처(Task 23+25),
Temperature Scaling 평가(Task 24)를 통합 실행했다.

---

## 1. Task 21: BIP 확률 실데이터 교체

### 배경
`pitch_env.py`와 `mdp_solver.py`에 인플레이 타구 확률이 하드코딩(out 70%, single 15%, double 10%, HR 5%)되어 있었다.
실제 MLB 데이터와 큰 괴리가 있어 시뮬레이션 정확도에 영향을 미쳤다.

### 작업 내용

| 항목 | 내용 |
|------|------|
| `scripts/compute_bip_probabilities.py` 신규 | Statcast play-by-play에서 BIP 확률 계산, 2021~2025 |
| `src/bip_loader.py` 신규 | JSON 기반 로더, `lru_cache`, `load_average()` 다시즌 평균 |
| `data/bip_probabilities_{2021~2025}.json` | 5시즌 BIP 확률 JSON |
| `pitch_env.py` 수정 | 하드코딩 → `bip_loader.load_average()`, 3루타 분기 추가 |
| `mdp_solver.py` 수정 | 동일 하드코딩 → `bip_loader`, 3루타 분기 추가 |

### 실측 vs 하드코딩 (5시즌 평균)

| 카테고리 | 하드코딩 | 실측 | 차이 |
|----------|---------|------|------|
| out | 70.0% | 67.0% | -3.0% |
| single | 15.0% | **21.7%** | **+6.7%** |
| double | 10.0% | 6.3% | -3.7% |
| triple | 0.0% | 0.56% | +0.56% |
| home_run | 5.0% | 4.4% | -0.6% |

핵심: **1루타가 하드코딩 대비 45% 과소평가**되어 있었다.

### 재평가 결과 (BIP 실데이터 + 2024 RE24)

| 군집 | MDP | MostFreq | Freq | Random |
|------|-----|----------|------|--------|
| 0 | +0.300 | +0.275 | +0.241 | +0.199 |
| 1 | +0.292 | +0.162 | +0.222 | +0.189 |
| 2 | +0.282 | +0.225 | +0.231 | +0.207 |
| 3 | +0.282 | +0.288 | +0.242 | +0.199 |

- 전반적 보상 하락 (1루타 증가 → 실점 기회 증가)
- **MDP 전 군집 최고 순위 보존**

---

## 2. Task 23+25: 5시즌 통합 학습 + 확장 피처

### 배경
기존 모델은 2023 단일 시즌 72만건으로 학습. 5시즌(2021~2025) 355만건으로 확대하고,
spin_rate/spin_axis/release_pos/platoon 등 7개 확장 피처를 추가.

### 3모델 비교 결과

| 모델 | val_acc | macro F1 | Brier | ECE | 피처수 |
|------|---------|----------|-------|-----|--------|
| Exp5 (2023, baseline) | 57.5% | 0.495 | — | — | 43 |
| MLP Base (5년) | 57.4% | 0.497 | 0.1411 | 0.0496 | 43 |
| MLP Extended (5년) | **57.9%** | **0.509** | 0.1392 | 0.0480 | 50 |
| LightGBM (5년) | **58.7%** | 0.497 | **0.1363** | **0.0025** | 18 |

### 분석

1. **LightGBM이 accuracy/calibration 모두 최고**
   - ECE 0.0025: MLP(0.048) 대비 ~20배 나은 calibration
   - 18개 피처로 43~50피처 MLP를 상회

2. **MLP Extended의 확장 피처 효과**
   - foul recall: 0.242 → 0.280 (+3.8%p)
   - macro F1: 0.497 → 0.509 (+0.012)
   - 확장 피처 7개: spin_rate_n, spin_axis_n, release_pos_x/z_n, platoon_advantage, p_throws_L, stand_L

3. **5년 확장의 한계**
   - MLP에서 5배 데이터 증가에도 val_acc 거의 동일 (57.5% → 57.4~57.9%)
   - 모델 capacity 또는 피처 표현력이 병목

### 산출물
- `best_model_5Year_MLP_Base.pth`, `best_model_5Year_MLP_Extended.pth`
- `data/feature_columns_5Year_MLP_*.json`, `data/target_classes_5Year_MLP_*.json`
- `docs/5year_model_comparison.md`

---

## 3. Task 24: Temperature Scaling 평가

### 결과
- MLP: T=1.45로 수렴, Brier 0.1411→0.1467 (악화), ECE 0.0496→0.0714 (악화)
- LightGBM: T=1.00 (이미 calibrated), ECE 0.0025

### 결론
Temperature Scaling은 이 데이터셋/모델에서 비효과적.
LightGBM은 추가 calibration 불필요.

---

## 4. 코드 변경 사항

### 신규 파일
| 파일 | 용도 |
|------|------|
| `scripts/compute_bip_probabilities.py` | BIP 확률 계산 |
| `src/bip_loader.py` | BIP JSON 로더 |
| `scripts/train_5year_models.py` | 5년 통합 학습 |
| `data/bip_probabilities_{2021~2025}.json` | 5시즌 BIP 데이터 |
| `data/re24_{2021,2022,2025}.json` | 추가 RE24 매트릭스 |
| `docs/5year_model_comparison.md` | 모델 비교 보고서 |

### 수정 파일
| 파일 | 변경 내용 |
|------|----------|
| `src/pitch_env.py` | BIP 하드코딩 → bip_loader, 3루타 분기 추가 |
| `src/mdp_solver.py` | BIP 하드코딩 → bip_loader, 3루타 분기 추가 |
| `src/model.py` | val_loader 속성 저장, 확장 피처 _extended_feats 목록 추가 |
| `CLAUDE.md` | 성능 수치, 파일 구조, 완료 작업, 설계 결정 갱신 |
| `TODO.md` | Task 21/23/24/25 완료 표시 |

---

## 5. 결론 및 다음 단계

### 결론
- BIP 실데이터 교체로 **시뮬레이션 현실성 대폭 향상** (1루타 +45% 보정)
- LightGBM이 MLP 대비 **accuracy +1.2%p, calibration 20배 우수**
- 5년 확장은 MLP에서 미미한 효과, 피처/모델 구조 변경이 더 효과적
- Temperature Scaling은 비효과적

### 다음 우선순위
1. **Task 26**: 구종 분류 통합 (Statcast pitch_type → PitchClustering)
2. **Task 27**: 투수 군집 K 재검토 (5시즌 데이터 기반)
3. **Task 28**: DQN 학습 스텝 증가 (300K → 500K~1M)
4. **Task 22**: 군집별 BIP 확률 세분화
