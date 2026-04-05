# 범용 MLP 전이 모델 실험 비교 (2023 MLB 전체 72만 건)

학습 대상: 2023 시즌 MLB 전체 Statcast 투구 (~752,739건, 4클래스)
출력 클래스: `ball` / `strike` / `foul` / `hit_into_play`
학습 환경: 20 epochs, batch_size=1024, lr=0.001, EarlyStopping(patience=5)
관리: W&B 프로젝트 `pitcheezy/SmartPitch-Portfolio`

---

## 4개 실험 요약

| | **Exp1** Baseline | **Exp3** CW | **Exp4** Physical | **Exp5** CW+Physical |
|---|:-:|:-:|:-:|:-:|
| Architecture | [256,128,64] | [128,64] | [256,128,64] | [256,128,64] |
| `use_class_weights` | ❌ | ✅ | ❌ | ✅ |
| `use_physical_features` | ❌ | ❌ | ✅ | ✅ |
| 입력 차원 | 40 | 40 | **43** | **43** |
| **Val Acc (Top-1)** | 58.1% | ~57.3% | **58.3%** 🏆 | 57.5% |
| Val Acc (Top-2) | — | — | **80.8%** | 79.7% |
| Val Acc (Top-3) | — | — | **95.1%** | 94.8% |
| Val Loss | 1.0077 | — | **0.9995** 🏆 | 1.0886† |
| Macro F1 | — | — | 0.481 | **0.495** 🏆 |

_† class_weights로 인해 loss 스케일이 다름 (직접 비교 불가)_

---

## Per-class F1 비교 (Exp4 vs Exp5)

| 클래스 | support | **Exp4** P / R / F1 | **Exp5** P / R / F1 | F1 Δ |
|---|:-:|:-:|:-:|:-:|
| ball | 54,078 | 0.673 / 0.952 / **0.789** | 0.678 / 0.941 / 0.788 | -0.001 |
| foul | 27,152 | 0.408 / 0.143 / 0.212 | 0.369 / 0.232 / **0.285** | **+0.073** ✅ |
| hit_into_play | 26,181 | 0.427 / 0.351 / 0.385 | 0.400 / 0.433 / **0.415** | **+0.030** ✅ |
| strike | 43,137 | 0.539 / 0.538 / **0.539** | 0.599 / 0.417 / 0.492 | -0.047 |

> Exp5는 class_weights로 **소수 클래스(foul, hit_into_play) recall을 대폭 개선**하고,
> 대신 strike recall을 일부 희생했다.

---

## Confusion Matrix

### Exp4 (Physical features only)
```
              ball      foul  hit_into    strike
ball         51476       260       332      2010
foul          7232      3892      7170      8858
hit_into      5177      2843      9187      8974
strike       12575      2534      4815     23213
```
- foul recall **14.3%** (대부분 strike 또는 hit로 분류)
- hit_into_play recall 35.1%

### Exp5 (CW + Physical)
```
              ball      foul  hit_into    strike
ball         50896       948       674      1560
foul          6886      6298      8707      5261
hit_into      4920      4691     11325      5245
strike       12365      5144      7638     17990
```
- foul recall **23.2%** (+8.9pp)
- hit_into_play recall **43.3%** (+8.2pp)
- strike recall 41.7% (-12.1pp, 대가)

---

## 선정 결정

**Canonical 모델: Exp5 (CW + Physical)**

### 이유
1. **MDP/DQN은 `predict_proba()` 확률 분포 전체를 사용** → Top-1 accuracy보다 분포 품질(F1)이 중요
2. Exp4 대비 **macro F1 +0.014**, 소수 클래스 recall 대폭 개선
3. ball 클래스(35.9% 비율)에 치우치는 예측 경향 완화

### 저장 파일 (현재 상태)
- `best_transition_model_universal.pth` → **Exp5** 가중치
- `data/feature_columns_universal.json` → 43차원 (물리 피처 포함)
- `data/model_config_universal.json` → `{hidden_dims: [256,128,64], dropout_rate: 0.2}`
- W&B Artifact `universal_transition_mlp:latest` → Exp5
- W&B Artifact `universal_transition_mlp:v0` → Exp4 (이전 버전)

---

## Task 12 — 투구 물리 피처 추가 결과

### 추가된 피처 (정규화 수식 하드코딩)
```python
release_speed_n = (release_speed - 90.0) / 5.0    # 구속 (mph)
pfx_x_n         = pfx_x                            # 수평 무브먼트 (ft)
pfx_z_n         = pfx_z                            # 수직 무브먼트 (ft)
```

### 해석
- **Top-1 개선폭이 작다 (+0.2pp)**: pitcher_cluster(K=4)와 mapped_pitch_name(9종)이
  이미 물리 정보를 간접적으로 담고 있어 중복 신호로 작용
- **val_loss 감소 (-0.008)**: 확률 분포는 유의미하게 개선
- **Phase 2 필요**: PitchEnv/MDPOptimizer에서 해당 컬럼이 0으로 채워지면 정확도 저하

### Phase 2 (다음 작업)
학습 데이터에서 `(pitcher_cluster, mapped_pitch_name)`별 평균
`release_speed_n/pfx_x_n/pfx_z_n` lookup 테이블 CSV 생성 →
PitchEnv._sample_outcome() / MDPOptimizer.predict_proba() 호출 시 주입.

---

## 재현 방법

```bash
# 특정 실험만 실행 (src/universal_model_trainer.py 하단 주석 해제)
# EXPERIMENTS = [EXPERIMENTS[-1]]

# 전체 재현
uv run src/universal_model_trainer.py
```
