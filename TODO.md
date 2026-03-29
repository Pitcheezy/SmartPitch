# SmartPitch TODO — 작업 리스트

코드를 직접 읽고 현재 상태를 확인한 후 작성했습니다.
마지막 업데이트: 2026-03-30

---

## 완료된 작업

- [x] **Task 1** — 범용 MLP 전환: `universal_model_trainer.py` 완성, `load_from_checkpoint()` 구현, `USE_UNIVERSAL_MODEL=True` 연동
- [x] **Task 2** — MLP 하이퍼파라미터: epochs=20, batch_size=256 적용
- [x] **Task 3** — 버그 수정: artifact_name 동적 처리, hidden_dims 전달 누락 수정
- [x] **Task 4** — EarlyStopping: `train_model(patience=5)`, val_loss 기준, `early_stop_epoch` W&B 로깅
- [x] **Task 추가** — 4클래스 전환: hit_by_pitch 제거, 출력 ball/strike/foul/hit_into_play
- [x] **Task 추가** — model_config_universal.json: 아키텍처 불일치 방지, hidden_dims 영속화
- [x] **Task 추가** — 실험 비교 기준: `min(val_loss)` → `max(val_acc)` (class-weighted loss 혼재 대응)

---

## 우선순위 판단 기준

각 작업은 아래 3가지 기준으로 평가됩니다:
- **[병목]** 이 작업을 안 하면 다른 작업이 막히는가?
- **[임팩트]** 전체 모델 성능에 얼마나 영향을 주는가?
- **[신뢰도]** 이 작업을 안 하면 현재 결과를 신뢰할 수 없는가?

---

## 전체 작업 목록 (우선순위 순)

---

### Task 5 — RE24 매트릭스 연도별 갱신

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

### Task 6 — 인플레이 타구 전이 확률 실데이터 기반 교체

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

### Task 7 — DQN 학습 강화

**왜 필요한가**
현재 total_timesteps=300,000, exploration_fraction=0.30으로 학습량이 부족하다.

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
**선행 작업**: 없음 (범용 모델 이미 완료됨)

---

### Task 8 — W&B Artifact를 파이프라인에 통합

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

### Task 9 — FastAPI 실시간 추천 API 구현

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

| # | 작업 | 난이도 | [병목] | [임팩트] | [신뢰도] | 선행 작업 |
|---|------|--------|--------|---------|---------|---------|
| ~~1~~ | ~~범용 MLP 전환~~ | ~~어려움~~ | ~~✅~~ | ~~✅~~ | ~~✅~~ | ~~완료~~ |
| ~~2~~ | ~~epochs 20, batch 256~~ | ~~쉬움~~ | ~~—~~ | ~~✅~~ | ~~✅~~ | ~~완료~~ |
| ~~3~~ | ~~버그 수정 2건~~ | ~~쉬움~~ | ~~—~~ | ~~—~~ | ~~✅~~ | ~~완료~~ |
| ~~4~~ | ~~EarlyStopping 추가~~ | ~~쉬움~~ | ~~—~~ | ~~✅~~ | ~~—~~ | ~~완료~~ |
| 5 | RE24 매트릭스 연도별 갱신 | 보통 | — | ✅ | ✅ | — |
| 6 | 인플레이 타구 확률 실데이터 교체 | 보통 | — | ✅ | ✅ | — |
| 7 | DQN 학습 강화 (300K→500K) | 쉬움 | — | ✅ | — | — |
| 8 | W&B Artifact 파이프라인 통합 | 보통 | — | — | ✅ | — |
| 9 | FastAPI 실시간 추천 API | 어려움 | — | ✅ | — | Task 7 이후 |
