# SmartPitch TODO — 작업 리스트

코드를 직접 읽고 현재 상태를 확인한 후 작성했습니다.
마지막 업데이트: 2026-03-26

---

## 완료된 작업

- [x] **Task 1** — 범용 MLP 전환: `universal_model_trainer.py` 완성, `model.py`에 `load_from_checkpoint()` 구현, `main.py`에 `USE_UNIVERSAL_MODEL=True` 플래그 연동 완료
- [x] **Task 2** — MLP epochs 5→20, batch_size 64→256 (`main.py:98`, `main.py:152`)
- [x] **Task 4** — EarlyStopping 추가: `train_model(patience=5)`, val_loss 기준, W&B `early_stop_epoch` 로깅
- [x] **Task 3** — 버그 수정 2건: artifact_name 하드코딩 (`main.py:125`), hidden_dims 미전달 (`main.py:157~160`, `model.py:289~293`)

---

## 우선순위 판단 기준

각 작업은 아래 3가지 기준으로 평가됩니다:
- **[병목]** 이 작업을 안 하면 다른 작업이 막히는가?
- **[임팩트]** 전체 모델 성능에 얼마나 영향을 주는가?
- **[신뢰도]** 이 작업을 안 하면 현재 결과를 신뢰할 수 없는가?

---

## 상위 3개 작업 우선순위 근거

### 왜 Task 1이 가장 먼저인가?
MLP가 단일 투수 데이터(~2,000건)로만 학습되어 val_acc 47% 수준이다.
MLP는 MDP(Task 5)와 DQN(Task 8) 양쪽의 시뮬레이터로 사용되므로
MLP가 틀리면 모든 다운스트림 결과가 함께 신뢰를 잃는다.
72만 건 데이터는 이미 batter_clustering.py가 받아놓은 상태이므로 진입 장벽이 낮다.

### 왜 Task 2가 두 번째인가?
main.py:98에 epochs=5가 하드코딩되어 있고,
model.py 독스트링에도 "epochs=5는 너무 적음"이라고 명시되어 있다.
Task 1과 코드 위치가 겹치므로 함께 처리하면 main.py를 두 번 건드릴 필요가 없다.
코드 변경 2줄로 즉시 효과를 볼 수 있는 가장 쉬운 개선이다.

### 왜 Task 3이 세 번째인가?
main.py:125에 artifact_name="gerrit_cole_raw_pitches"가 하드코딩되어 있어
Kershaw나 다른 투수를 분석해도 W&B Artifact가 Gerrit Cole 이름으로 올라간다.
또한 wandb.config에 hidden_dims=[128,64]를 정의해놓고
실제 train_model()에 전달하지 않아 설정값이 무시되는 문제도 있다.
두 버그 모두 현재 결과의 재현성과 신뢰도를 해친다.

---

## 전체 작업 목록 (우선순위 순)

---

### ~~Task 1 — 범용 MLP 전환 (전체 MLB 데이터로 재학습)~~ ✅ 완료

`universal_model_trainer.py` 생성, `model.py`에 `load_from_checkpoint()` classmethod 구현,
`main.py`에 `USE_UNIVERSAL_MODEL=True` 플래그 및 분기 로직 완성.
`uv run src/universal_model_trainer.py` 실행 후 `best_transition_model_universal.pth` 및
`data/feature_columns_universal.json`, `data/target_classes_universal.json` 생성됨.

---

### Task 2 — MLP 하이퍼파라미터 즉시 개선

**왜 필요한가**
main.py:98에 epochs=5, main.py:152에 batch_size=64가 하드코딩되어 있다.
model.py 독스트링에도 "epochs=5는 너무 적음"이라 명시된 상태로, 가장 빠른 성능 개선 수단이다.

**평가**
- [병목] ❌
- [임팩트] ✅ 학습량 부족이 val_acc 47%의 직접적 원인 중 하나
- [신뢰도] ✅ 5 epoch는 수렴 전에 학습을 끊는 것과 같음

**수정 파일**
- `src/main.py`

**구체적 변경**
```python
# main.py:98 변경
"epochs": 5,  →  "epochs": 20,

# main.py:152 변경
batch_size=64,  →  batch_size=256,  # 범용 모델 시 1024 권장
```

**예상 난이도**: 쉬움
**선행 작업**: Task 1과 동시 처리 권장 (같은 파일)

---

### Task 3 — 코드 버그 2건 수정

**왜 필요한가**
현재 두 가지 버그가 존재하며 실험 재현성과 신뢰도를 해친다.

**버그 1**: main.py:125 — artifact_name 하드코딩
```python
# 현재 (버그)
artifact_name="gerrit_cole_raw_pitches"  # 어떤 투수를 분석해도 이 이름으로 올라감

# 수정
artifact_name=f"{player_first_name}_{player_last_name}_raw_pitches"
```

**버그 2**: main.py:99 + model.py — hidden_dims 미전달
```python
# wandb.config에 정의되어 있으나 train_model()에 전달되지 않음
"hidden_dims": [128, 64],  # 이 값이 실제로는 무시됨

# run_modeling_pipeline()와 train_model() 시그니처에 hidden_dims 전달 추가 필요
model_module.run_modeling_pipeline(
    epochs=config.epochs,
    hidden_dims=config.hidden_dims,  # 추가
    upload_artifact=True
)
```

**평가**
- [병목] ❌
- [임팩트] ❌ (현재 기본값이 wandb config와 동일하여 결과는 같음)
- [신뢰도] ✅ W&B 기록이 실제 실험과 불일치 → 재현 불가

**수정 파일**
- `src/main.py` (lines 99, 125, 157~160)
- `src/model.py` (`run_modeling_pipeline` 시그니처)

**예상 난이도**: 쉬움
**선행 작업**: 없음 (독립적)

---

### Task 4 — EarlyStopping 추가

**왜 필요한가**
epochs를 20으로 늘리면 과적합 위험이 높아진다.
현재 model.py에 EarlyStopping이 없어 val_loss가 다시 올라가도 학습을 멈추지 않는다.
현재도 train/val 갭이 0.43으로 과적합 징후가 있다.

**평가**
- [병목] ❌
- [임팩트] ✅ epochs 증가 시 과적합 방지 → 실제 최적 성능 달성
- [신뢰도] ❌

**수정 파일**
- `src/model.py` (`train_model()` 내부에 patience 기반 early stopping 추가)

**구현 방향**
```python
patience = 5
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1, epochs + 1):
    ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(...)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

**예상 난이도**: 쉬움
**선행 작업**: Task 2 (epochs 늘린 후 의미가 생김)

---

### Task 5 — RE24 매트릭스 연도별 갱신

**왜 필요한가**
현재 pitch_env.py와 mdp_solver.py 두 파일에 2019 MLB 기준값이 하드코딩되어 있다.
분석 대상 시즌(예: 2024)과 5년 차이가 나는 기대득점 기준을 사용하는 것은 신뢰도 문제다.

**평가**
- [병목] ❌
- [임팩트] ✅ 보상 함수의 기준값이 바뀌므로 MDP/DQN 정책에 직접 영향
- [신뢰도] ✅ 2024 시즌 분석에 2019 기준 사용 중

**수정 파일**
- `src/pitch_env.py` (`RE24_MATRIX` 딕셔너리, line 48~55)
- `src/mdp_solver.py` (`re24_matrix` 딕셔너리, line 73~80)
- 두 파일을 **반드시 동시에** 수정해야 함 (같은 값을 두 곳에서 사용)

**구현 방향**
```python
# 방법 1: pybaseball의 run_expectancies() 활용해 연도별 자동 계산
# 방법 2: 외부 JSON 파일로 분리해 main.py에서 시즌에 맞게 로드
# 방법 3: 수동 업데이트 (2024 MLB RE24 값 조사 후 교체)
```

**예상 난이도**: 보통
**선행 작업**: 없음 (독립적)

---

### Task 6 — 인플레이 타구 전이 확률 실데이터 기반 교체

**왜 필요한가**
인플레이 타구 결과를 아웃 70%, 1루타 15%, 2루타 10%, 홈런 5%로 임의 설정했다.
이 값도 두 파일에 동일하게 하드코딩되어 있으며, 실제 MLB 통계와 다를 수 있다.

**평가**
- [병목] ❌
- [임팩트] ✅ MDP 상태 전이 확률에 직접 영향, DQN 보상 계산에도 영향
- [신뢰도] ✅ 임의 설정값으로 보상 계산 중

**수정 파일**
- `src/pitch_env.py` (`_apply_batted_ball()`, lines 328~342)
- `src/mdp_solver.py` (`_get_next_states_and_rewards()`, lines 155~171)
- 두 파일을 **반드시 동시에** 수정해야 함

**구현 방향**
```python
# 2023 MLB Statcast 데이터에서 실제 비율 계산
# hit_into_play 결과 중 out/single/double/triple/hr 비율 집계
# 이상적으로는 타자 군집별로 다른 비율 적용
```

**예상 난이도**: 보통
**선행 작업**: 없음 (독립적), Task 1과 함께 처리하면 같은 raw_df 재활용 가능

---

### Task 7 — DQN 학습 강화

**왜 필요한가**
현재 total_timesteps=300,000, exploration_fraction=0.30으로 학습량이 부족하다.
30만 스텝은 이닝 약 30,000개 수준이며, 더 많은 탐색이 필요하다.

**평가**
- [병목] ❌
- [임팩트] ✅ 평균 보상 0.235에서 추가 개선 여지
- [신뢰도] ❌

**수정 파일**
- `src/main.py` (lines 103, 106)

**구체적 변경**
```python
# main.py:103
"dqn_total_timesteps": 300_000,  →  "dqn_total_timesteps": 500_000,

# main.py:106
"dqn_exploration_fraction": 0.30,  →  "dqn_exploration_fraction": 0.40,
```

**예상 난이도**: 쉬움
**선행 작업**: Task 1 권장 (좋은 MLP 시뮬레이터 위에서 DQN을 강화해야 의미 있음)

---

### Task 8 — W&B Artifact를 파이프라인에 통합

**왜 필요한가**
현재 군집화 CSV가 W&B Artifact로 업로드되지만, 파이프라인은 이를 다운로드하지 않고
로컬 파일을 직접 읽는다. 팀원이 다른 환경에서 실행할 때 로컬 CSV가 없으면 fallback 작동.

**평가**
- [병목] ❌
- [임팩트] ❌ (로컬 파일이 있으면 동일하게 작동)
- [신뢰도] ✅ 팀 협업 환경에서 재현성 보장 불가

**수정 파일**
- `src/main.py` (Artifact 다운로드 로직 추가)

**구현 방향**
```python
# main.py에서 군집화 CSV를 W&B Artifact에서 다운로드
api = wandb.Api()
artifact = api.artifact("pitcheezy/SmartPitch-Portfolio/batter_cluster_mapping:latest")
artifact.download(root="data/")
```

**예상 난이도**: 보통
**선행 작업**: 없음 (독립적)

---

### Task 9 — FastAPI 실시간 추천 API 구현

**왜 필요한가**
현재 결과를 실제로 쓰려면 main.py를 직접 실행해야 한다.
API 서버가 있으면 타석 상황을 입력하면 즉시 추천 구종/코스를 받을 수 있다.

**평가**
- [병목] ❌
- [임팩트] ✅ 실사용 가능한 서비스로 전환
- [신뢰도] ❌

**수정 파일**
- 신규 파일: `api.py` (또는 `src/api.py`)
- `pyproject.toml` (`fastapi`, `uvicorn` 의존성 추가)

**구현 방향**
```
POST /recommend
입력: {balls, strikes, outs, on_1b, on_2b, on_3b, batter_id, pitcher_id}
처리: batter/pitcher CSV 조회 → 8D obs 구성 → DQN.predict()
출력: {pitch_type, zone, expected_run_prevention}
```

**예상 난이도**: 어려움
**선행 작업**: Task 1 (좋은 MLP 필요), Task 7 (잘 학습된 DQN 필요)

---

## 요약 테이블

| # | 작업 | 난이도 | [병목] | [임팩트] | [신뢰도] | 선행 작업 |
|---|------|--------|--------|---------|---------|---------|
| ~~1~~ | ~~범용 MLP 전환 (72만 건)~~ | ~~어려움~~ | ~~✅~~ | ~~✅~~ | ~~✅~~ | ~~완료~~ |
| 2 | epochs 5→20, batch 64→256 | 쉬움 | — | ✅ | ✅ | Task 1과 동시 |
| 3 | 버그 수정 2건 (artifact명, hidden_dims) | 쉬움 | — | — | ✅ | — |
| ~~4~~ | ~~EarlyStopping 추가~~ | ~~쉬움~~ | ~~—~~ | ~~✅~~ | ~~—~~ | ~~완료~~ |
| 5 | RE24 매트릭스 연도별 갱신 | 보통 | — | ✅ | ✅ | — |
| 6 | 인플레이 타구 확률 실데이터 교체 | 보통 | — | ✅ | ✅ | — |
| 7 | DQN 학습 강화 (300K→500K) | 쉬움 | — | ✅ | — | Task 1 이후 |
| 8 | W&B Artifact 파이프라인 통합 | 보통 | — | — | ✅ | — |
| 9 | FastAPI 실시간 추천 API | 어려움 | — | ✅ | — | Task 1, 7 이후 |

---

## 즉시 실행 가능한 작업 (선행 작업 없음)

코드 변경 2줄 이하의 빠른 개선:

```bash
# Task 3 — 버그 수정
# main.py:125: "gerrit_cole_raw_pitches" → f"{player_first_name}_{player_last_name}_raw_pitches"
# main.py:157: epochs=config.epochs에 hidden_dims=config.hidden_dims 추가

# Task 2 — 하이퍼파라미터 즉시 개선
# main.py:98: epochs: 5 → 20
# main.py:152: batch_size=64 → batch_size=256
```
