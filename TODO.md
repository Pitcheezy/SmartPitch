# SmartPitch TODO — 작업 리스트

코드를 직접 읽고 현재 상태를 확인한 후 작성했습니다.
마지막 업데이트: 2026-04-11

---

## 완료된 작업

- [x] **Task 1** — 범용 MLP 전환: `universal_model_trainer.py` 완성, `load_from_checkpoint()` 구현, `USE_UNIVERSAL_MODEL=True` 연동
- [x] **Task 2** — MLP 하이퍼파라미터: epochs=20, batch_size=256 적용
- [x] **Task 3** — 버그 수정: artifact_name 동적 처리, hidden_dims 전달 누락 수정
- [x] **Task 4** — EarlyStopping: `train_model(patience=5)`, val_loss 기준, `early_stop_epoch` W&B 로깅
- [x] **Task 추가** — 4클래스 전환: hit_by_pitch 제거, 출력 ball/strike/foul/hit_into_play
- [x] **Task 추가** — model_config_universal.json: 아키텍처 불일치 방지, hidden_dims 영속화
- [x] **Task 추가** — 실험 비교 기준: `min(val_loss)` → `max(val_acc)` (class-weighted loss 혼재 대응)
- [x] **Task 11** — Top-K Accuracy 측정 추가 (Top-1/2/3, 커밋 89d64d1)
- [x] **Task 12** — 투구 물리 피처 추가 (Phase 1): release_speed/pfx_x/pfx_z, Exp4 val_acc=58.3%, Exp5 macro F1=0.495 (커밋 a160553)
- [x] **Task 13** — 베이스라인 5종 비교 평가 (커밋 3fd2352)
  - `src/evaluate_baselines.py`: Random / MostFrequent / Frequency / MDPPolicy / DQN (Cole 2019 ref)
  - 전체(cluster 0) + 군집별(K=4) 각 1000 에피소드, 동일 seed
  - `data/mdp_optimal_policy.pkl` 캐시 (9,216 상태, 5회 VI 결과)
  - `docs/baseline_comparison.md`, `docs/baseline_by_cluster.md`
  - 결과: DQN +0.436 > Random +0.231 > Frequency +0.223 > MDP +0.151 > MostFrequent +0.151 (cluster 0)
- [x] **Task 14** — MDP vs PitchEnv 보상 일관성 분석 (커밋 3fd2352)
  - `scripts/analyze_mdp_vs_env.py`: 보상식·전이 로직 줄 단위 비교 + VI 수렴·정책 분포·episode trace
  - `docs/mdp_vs_env_reward_analysis.md`: 6섹션 리포트
  - 결론: 코드 버그 없음. MDP 열위 원인 = (1) VI 5회 미수렴(max|ΔV|=0.145), (2) MLP 58% 보정 부족으로 정책이 Knuckleball 70.5%에 편중, (3) stochastic env에서 단일 sample이 오차를 증폭
- [x] **Task 15** — 발표용 시각화 자료 생성 (`scripts/generate_baseline_presentation.py`)
  - `docs/presentation_baseline_overall.png` (Cole 2019 5-agent bar)
  - `docs/presentation_baseline_by_cluster.png` (4군집 × 5에이전트 grouped bar)
  - `docs/presentation_summary_table.png` (행별 최고값 강조 요약표)

---

## 우선순위 판단 기준

각 작업은 아래 3가지 기준으로 평가됩니다:
- **[병목]** 이 작업을 안 하면 다른 작업이 막히는가?
- **[임팩트]** 전체 모델 성능에 얼마나 영향을 주는가?
- **[신뢰도]** 이 작업을 안 하면 현재 결과를 신뢰할 수 없는가?

---

## 전체 작업 목록 (우선순위 순)

---

### ~~1순위: Task 12 — 투구 물리 피처 추가~~ (Phase 1 완료, Phase 2 남음)

**왜 필요한가**
현재 MLP는 `mapped_pitch_name`(one-hot 9차원) + `zone`만 본다.
즉 "어떤 구종을 어느 존에" 던졌는지는 알지만, 그 투구의 **구속**(95mph vs 88mph)과
**무브먼트**(pfx_x, pfx_z)는 모른다. 같은 "Fastball" 라벨 안에서도 구속/무브먼트
편차가 크기 때문에 결과(strike/foul/hit) 확률이 달라지는데 모델은 이를 구분 못한다.

**왜 1순위인가**
- 현재 val_acc 58.1% 천장의 가장 큰 원인 중 하나 (정보 손실)
- Exp3의 `is_two_strike`/`zone_row/col` 파생 피처와 같은 철학이지만 **훨씬 강한 신호**
- Task 10(구종 군집화)과 중첩: 구종 군집화는 물리 정보를 간접적으로 넣는 것,
  물리 피처를 직접 넣는 편이 단순하고 효과도 더 크다 → Task 10보다 우선
- 업계 표준: Stuff+/Pitching+ 모델은 구속·무브먼트를 핵심 피처로 사용

**평가**
- [병목] ❌
- [임팩트] ✅✅ val_acc 천장 돌파 기대 (58.1% → 60%+ 가능성)
- [신뢰도] ✅ 동일 라벨 내 분산을 모델에 알려줌

**수정 파일**
- `src/universal_model_trainer.py`
  - `required` 리스트에 `release_speed`, `pfx_x`, `pfx_z` 추가 (L181-186)
- `src/model.py`
  - `_prepare_data()` X_num에 3개 피처 추가 (L228-235)
  - StandardScaler 필요 (스케일 차이 큼: 구속 80~100, pfx는 -2~2)

**Phase 1 (완료, 커밋 a160553)**
- Exp4 PhysicalFeatures val_acc **58.3%** (+0.2pp)
- Exp5 CW+Physical macro F1 **0.495**

**Phase 2 (우선순위 1로 상승)** — PitchEnv/MDPOptimizer 추론 연결
- 현재 `PitchEnv._sample_outcome`/`MDPOptimizer.predict_proba`에서 물리 피처가 0으로 주입
- Task 14 분석 결과 이것이 MDP 정책 편중(Knuckleball 70.5%)의 한 원인일 가능성
- 학습 데이터에서 (pitcher_cluster, mapped_pitch_name)별 평균 CSV 생성 후 lookup

**예상 난이도**: 쉬움 (Phase 1), 보통 (Phase 2)
**선행 작업**: 없음

---

### 2순위: Task 10 — 범용 모델 구종 군집화 통합 (학습-추론 구종 분류 일치)

**왜 필요한가**
현재 범용 모델(`universal_model_trainer.py`)은 Statcast의 `pitch_type`(MLB 공식 라벨: FF, SL 등)을
그대로 사용하지만, 실제 추론 시(`main.py` → `clustering.py`)에는 투수별 물리적 군집화 결과를 사용한다.
학습과 추론의 구종 분류 기준이 **불일치**하므로, 범용 모델 학습에도 투수별 물리적 군집화를 적용해야 한다.

**왜 1순위인가**
학습-추론 불일치는 파이프라인에서 가장 근본적인 신뢰도 문제다.
이 불일치가 해소되지 않으면 Task 5, 6, 7의 개선 효과도 정확히 측정할 수 없다.
또한 MLP 정확도 58.1%의 천장이 이 불일치 때문일 수 있다.

**평가**
- [병목] ❌
- [임팩트] ✅ 학습-추론 구종 분류 일치로 MLP 예측 정확도 개선 기대
- [신뢰도] ✅✅ 현재 학습(MLB 라벨)과 추론(물리적 군집)이 다른 기준 사용 중

**파이프라인 설계**
```
Statcast 72만 건
    ↓
투수별 그룹핑 (479명)
    ↓
각 투수마다 clustering.py 로직 적용
  (6개 물리 피처 → UMAP → KMeans K=3~6 실루엣 탐색)
    ↓
군집별 구속 내림차순 구종명 매핑 (Fastball, Slider, Changeup, ...)
    ↓
pitch_type 컬럼을 mapped_pitch_name으로 교체
    ↓
나머지 동일하게 MLP 학습
```

**수정 파일**
- `src/universal_model_trainer.py` (전처리 단계에 투수별 군집화 추가)
- `src/clustering.py` (배치 처리용 함수 추출 또는 리팩토링)

**고려사항**
- 투수마다 최적 K가 다름 (3~6종) → 전체 구종 카테고리 수 변동
- 479명 × UMAP+KMeans → 전처리 시간 약 30~60분 추가
- 군집화 실패 투수 (투구 수 부족 등) fallback 처리 필요

**예상 난이도**: 어려움
**선행 작업**: 없음 (독립적)

---

### 3순위: Task 5 — RE24 매트릭스 연도별 갱신

**왜 필요한가**
pitch_env.py와 mdp_solver.py 두 파일에 2019 MLB 기준값이 하드코딩되어 있다.
분석 대상 시즌과 기대득점 기준이 다르면 보상 함수의 절대값이 틀어진다.

**평가**
- [병목] ❌
- [임팩트] ✅ 보상 함수의 기준값이 바뀌므로 MDP/DQN 정책에 직접 영향
- [신뢰도] ✅ 2024 시즌 분석에 2019 기준 사용 중

**수정 파일**
- `src/pitch_env.py` (`RE24_MATRIX` 딕셔너리)
- `src/mdp_solver.py` (`re24_matrix` 딕셔너리)
- 두 파일을 **반드시 동시에** 수정해야 함

**구현 방향**
```python
# 방법 1: pybaseball의 run_expectancies() 활용해 연도별 자동 계산
# 방법 2: 외부 JSON 파일로 분리해 main.py에서 시즌에 맞게 로드
# 방법 3: 수동 업데이트 (분석 대상 시즌 RE24 값 조사 후 교체)
```

**예상 난이도**: 보통
**선행 작업**: 없음 (독립적)

---

### 4순위: Task 6 — 인플레이 타구 전이 확률 실데이터 기반 교체

**왜 필요한가**
인플레이 타구 결과를 아웃 70%, 1루타 15%, 2루타 10%, 홈런 5%로 임의 설정했다.
두 파일에 동일하게 하드코딩되어 있으며, 실제 MLB 통계와 다를 수 있다.

**평가**
- [병목] ❌
- [임팩트] ✅ MDP 상태 전이 확률에 직접 영향, DQN 보상 계산에도 영향
- [신뢰도] ✅ 임의 설정값으로 보상 계산 중

**수정 파일**
- `src/pitch_env.py` (`_apply_batted_ball()`)
- `src/mdp_solver.py` (`_get_next_states_and_rewards()`)
- 두 파일을 **반드시 동시에** 수정해야 함

**구현 방향**
```python
# 2023 MLB Statcast 데이터에서 hit_into_play 결과 중 out/single/double/triple/hr 비율 집계
# 이상적으로는 타자 군집별로 다른 비율 적용
```

**예상 난이도**: 보통
**선행 작업**: 없음 (독립적), batter_clustering.py의 raw_df 재활용 가능

---

### ~~Task 11 — Top-K Accuracy 측정 추가~~ (완료, 커밋 89d64d1)

**왜 필요한가**
현재 MLP 평가는 Top-1 accuracy(58.1%)만 측정한다.
투구 결과는 본질적으로 확률적이므로 (같은 상황에서도 strike/foul/ball 모두 가능),
Top-1만으로는 모델의 실제 성능을 과소평가할 수 있다.

Top-K accuracy를 함께 보고하면 모델이 확률 분포를 얼마나 잘 학습했는지 보여줄 수 있다.

```
예시:
  MLP 예측: foul 35%, strike 30%, ball 25%, hit 10%
  정답: strike

  Top-1: ✗ (1등 foul ≠ strike)
  Top-2: ✓ (2등 strike = 정답)
```

**발표 활용**
```
"Top-1 accuracy 58%이지만, Top-2 accuracy는 78%입니다.
 이는 모델이 실제 발생 가능한 결과를 상위 2개 안에 포착하고 있다는 의미이며,
 MDP/DQN은 확률 분포 전체를 사용하므로 Top-1보다 높은 성능으로 작동합니다."
```

**평가**
- [병목] ❌
- [임팩트] ❌ 모델 자체는 변하지 않음 (측정 방식 추가)
- [신뢰도] ✅ 성능 보고의 다각적 분석

**수정 파일**
- `src/universal_model_trainer.py` (평가 단계에 Top-2, Top-3 accuracy 계산 추가)
- `src/model.py` (선택: `evaluate()` 메서드에 Top-K 옵션 추가)

**구현 방향**
```python
# PyTorch에서 Top-K accuracy 계산 (1줄)
top_k_correct = torch.topk(probs, k=2).indices  # 상위 2개 인덱스
top_k_acc = (top_k_correct == labels.unsqueeze(1)).any(dim=1).float().mean()
```

**예상 난이도**: 쉬움
**선행 작업**: 없음 (피처 추가 실험 시 함께 측정하면 효율적)

---

### 5순위 (변경 없음): Task 7 — DQN 학습 강화

**왜 필요한가**
현재 total_timesteps=300,000, exploration_fraction=0.30으로 학습량이 부족하다.
Task 10, 5, 6으로 모델과 보상을 개선한 뒤 학습량을 늘려야 의미 있는 비교가 가능하다.

**평가**
- [병목] ❌
- [임팩트] ✅ 평균 보상 추가 개선 여지
- [신뢰도] ❌

**수정 파일**
- `src/main.py` (wandb config)

**구체적 변경**
```python
"dqn_total_timesteps": 500_000,       # 300K → 500K
"dqn_exploration_fraction": 0.40,     # 0.30 → 0.40
```

**예상 난이도**: 쉬움
**선행 작업**: Task 10, 5, 6 이후 권장 (개선된 모델/보상 기반으로 학습해야 의미 있음)

---

### 6순위: Task 8 — W&B Artifact를 파이프라인에 통합

**왜 필요한가**
파이프라인은 로컬 CSV를 직접 읽는다. 팀원이 다른 환경에서 실행할 때 로컬 CSV가 없으면 fallback 동작.

**평가**
- [병목] ❌
- [임팩트] ❌ (로컬 파일이 있으면 동일하게 작동)
- [신뢰도] ✅ 팀 협업 환경에서 재현성 보장 불가

**수정 파일**
- `src/main.py` (Artifact 다운로드 로직 추가)

**구현 방향**
```python
api = wandb.Api()
artifact = api.artifact("pitcheezy/SmartPitch-Portfolio/batter_cluster_mapping:latest")
artifact.download(root="data/")
```

**예상 난이도**: 보통
**선행 작업**: 없음 (독립적)

---

### 7순위: Task 9 — FastAPI 실시간 추천 API 구현

**왜 필요한가**
현재 결과를 실제로 쓰려면 main.py를 직접 실행해야 한다.

**평가**
- [병목] ❌
- [임팩트] ✅ 실사용 가능한 서비스로 전환
- [신뢰도] ❌

**수정 파일**
- 신규 파일: `src/api.py`
- `pyproject.toml` (`fastapi`, `uvicorn` 의존성 추가)

**구현 방향**
```
POST /recommend
입력: {balls, strikes, outs, on_1b, on_2b, on_3b, batter_id, pitcher_id}
처리: batter/pitcher CSV 조회 → 8D obs 구성 → DQN.predict()
출력: {pitch_type, zone, expected_run_prevention}
```

**예상 난이도**: 어려움
**선행 작업**: Task 7 (잘 학습된 DQN 권장)

---

## 요약 테이블

| 순위 | # | 작업 | 난이도 | [병목] | [임팩트] | [신뢰도] | 선행 작업 |
|------|---|------|--------|--------|---------|---------|---------|
| ~~—~~ | ~~1~~ | ~~범용 MLP 전환~~ | ~~어려움~~ | ~~✅~~ | ~~✅~~ | ~~✅~~ | ~~완료~~ |
| ~~—~~ | ~~2~~ | ~~epochs 20, batch 256~~ | ~~쉬움~~ | ~~—~~ | ~~✅~~ | ~~✅~~ | ~~완료~~ |
| ~~—~~ | ~~3~~ | ~~버그 수정 2건~~ | ~~쉬움~~ | ~~—~~ | ~~—~~ | ~~✅~~ | ~~완료~~ |
| ~~—~~ | ~~4~~ | ~~EarlyStopping 추가~~ | ~~쉬움~~ | ~~—~~ | ~~✅~~ | ~~—~~ | ~~완료~~ |
| ~~—~~ | ~~12P1~~ | ~~투구 물리 피처 추가 Phase 1~~ | ~~쉬움~~ | ~~—~~ | ~~✅✅~~ | ~~✅~~ | ~~완료~~ |
| ~~—~~ | ~~13~~ | ~~베이스라인 5종 비교 평가~~ | ~~보통~~ | ~~—~~ | ~~—~~ | ~~✅~~ | ~~완료~~ |
| ~~—~~ | ~~14~~ | ~~MDP vs PitchEnv 보상 분석~~ | ~~보통~~ | ~~—~~ | ~~—~~ | ~~✅✅~~ | ~~완료~~ |
| ~~—~~ | ~~15~~ | ~~발표용 시각화 자료~~ | ~~쉬움~~ | ~~—~~ | ~~—~~ | ~~—~~ | ~~완료~~ |
| **1** | 12P2 | 물리 피처 Phase 2 (lookup 테이블) | 보통 | — | ✅✅ | ✅ | Task 13/14 결과 반영 |
| **2** | 16 | MDP solve_mdp 수렴 개선 (5→10회 또는 δ<1e-4, γ=0.99) | 쉬움 | — | ✅ | ✅ | Task 14 권고 |
| **3** | 10 | 범용 모델 구종 군집화 통합 | 어려움 | — | ✅ | ✅✅ | — |
| **4** | 5 | RE24 매트릭스 연도별 갱신 | 보통 | — | ✅ | ✅ | — |
| **5** | 6 | 인플레이 타구 확률 실데이터 교체 | 보통 | — | ✅ | ✅ | — |
| **6** | 7 | DQN 학습 강화 (300K→500K) + 군집 1~3 DQN 학습 | 쉬움 | — | ✅ | — | Task 12P2,10,5,6 이후 |
| **7** | 8 | W&B Artifact 파이프라인 통합 | 보통 | — | — | ✅ | — |
| **8** | 9 | FastAPI 실시간 추천 API | 어려움 | — | ✅ | — | Task 7 이후 |
