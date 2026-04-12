# SmartPitch Claude Code 프롬프트 모음 — v4 (2026-04-12 최신)

> 최신 상태: Task 12~15 완료, 커밋 c604272까지 push 완료
> AI_CONTEXT.md, README.md, CLAUDE.md, TODO.md 모두 동기화됨
> 
> **사용법**: 새 Claude Code 세션을 열고, 0단계부터 순서대로 넣어.
> 한 단계 끝나면 결과 확인 → 이해 안 되면 추가 질문 → 다음 단계.

---

## 실행 순서 요약

```
0단계: 컨텍스트 세팅 (매번 새 세션마다)
  ↓
A. 학습용 (프로젝트 이해)     B. 작업용 (다음 Task 진행)
1단계: 전체 구조               B-1: Task 12P2 (물리피처 lookup)
2단계: 데이터 수집/전처리       B-2: Task 16 (MDP 수렴 개선)
3단계: 클러스터링               B-3: 군집 1~3 DQN 학습
4단계: MLP 모델                B-4: 차트 수정
5단계: MDP                     B-5: 예상 질문 답변
6단계: 강화학습 환경            B-6: 커밋/푸시
7단계: DQN 학습
8단계: 오케스트레이터
9단계: 발표용 종합 정리

A는 프로젝트 이해용 (공부), B는 실제 작업용 (개발)
둘 중 필요한 것만 골라서 진행해도 돼.
```

---

## 0단계 — 컨텍스트 세팅 (새 세션마다 매번 넣기)

```
이 프로젝트는 SmartPitch라는 졸업 캡스톤 프로젝트야.
MLB Statcast 데이터 기반 투구 전략 최적화 AI 파이프라인이고,
MLP(전이확률 예측) → MDP(가치반복) → DQN(강화학습) 3단계 구조야.

핵심 정보:
- 2023 MLB 전체 시즌 72만건 데이터
- MLP 출력: 4클래스 (ball, foul, hit_into_play, strike)
- 투수 군집 K=4, 타자 군집 K=8
- canonical 모델: Exp5 (val_acc 57.5%, macro F1 0.495, 43차원 — 물리 피처 3개 포함)
- ⚠️ 현재 PitchEnv/MDPOptimizer에서 물리 피처 3개가 0으로 입력됨 (Phase 2 미완)
- DQN: Cole 2019, 평균 보상 +0.436 ± 1.255
- 베이스라인 5종 비교 완료: DQN > Random > Frequency > MDP ≈ MostFrequent

최근 완료된 작업:
- Task 12: 물리 피처 추가 (Phase 1 완료, Phase 2 남음)
- Task 13: 베이스라인 5종 비교 평가 (src/evaluate_baselines.py)
- Task 14: MDP vs PitchEnv 보상 분석 → MDP 열위 원인 3가지 발견
  (1) VI 5회 미수렴 max|ΔV|=0.145
  (2) MLP 58% 보정 부족 → 정책 Knuckleball 70.5% 편중
  (3) stochastic env 단일 sample 오차 증폭
- Task 15: 발표용 시각화 자료 3종

한국어로, 쉽고 자세하게, 설계 의도까지 설명해줘.
발표 때 교수님 예상 질문도 준비해줘.
```

---

## A. 파이프라인 학습용 프롬프트

### 1단계 — 전체 구조 파악

```
src/ 디렉토리의 모든 파일을 확인하고, AI_CONTEXT.md와 README.md를 읽어서
SmartPitch 프로젝트의 전체 구조를 파이프라인 흐름 순서대로 설명해줘.

특히:
1. 각 src/ 파일의 역할 (한 줄 요약)
2. 데이터가 어떤 순서로 흘러가는지 (입력 → 출력 체인)
3. MLP, MDP, DQN 세 모델의 관계
4. main.py의 호출 순서
5. universal_model_trainer.py와 main.py의 역할 차이

파이썬 초보도 이해할 수 있게 쉬운 비유를 써서 설명해줘.
```

### 2단계 — 데이터 수집 + 전처리

```
src/data_loader.py와 src/universal_model_trainer.py를 읽고
데이터 수집과 전처리 과정을 설명해줘.

두 파일 모두 데이터를 가져오는데, 각각 언제 쓰이는지 차이점부터 설명하고:

특히 이것들을 코드 보면서 자세히:
1. PITCH_TYPE_MAP — 왜 세분화된 코드를 합치는지, 최종 몇 종인지
2. DESCRIPTION_MAP — 12가지 투구 결과를 어떻게 4클래스로 줄이는지
3. _preprocess_raw() 함수의 각 단계
4. _add_physical_features() — 왜 StandardScaler 대신 수식 하드코딩인지
5. dropna로 제거되는 데이터량

발표 예상 질문: "왜 12클래스를 4클래스로 줄였나요?"
코드의 각 라인이 왜 필요한지 주석 수준으로 설명해줘.
```

### 3단계 — 클러스터링

```
src/clustering.py, src/pitcher_clustering.py, src/batter_clustering.py를 읽고
투수/타자 군집화 과정을 설명해줘.

1. 왜 군집화가 필요한지 — 안 하면 어떤 문제가 생기는지
2. 군집화에 사용한 피처들 — 투수/타자 각각 뭔지와 왜 골랐는지
3. UMAP이 뭔지 — 차원 축소를 왜 해야 하는지
4. KMeans가 뭔지 — 어떻게 K개 그룹으로 나누는지
5. K 선정 기준 — 투수 K=4, 타자 K=8을 어떻게 정했는지
6. 실루엣 점수 0.45의 의미
7. 결과 CSV가 model.py _prepare_data()에서 어떻게 merge되는지

발표 예상 질문: "왜 K=4인가요?", "실루엣 점수가 낮지 않나요?"
각 군집이 실제 야구에서 어떤 유형에 대응하는지도 설명해줘.
```

### 4단계 — MLP 모델 (핵심)

```
src/model.py와 src/universal_model_trainer.py를 읽고
MLP 모델의 전체 과정을 코드 기반으로 설명해줘.

1. _prepare_data() 함수 — 입력 구성의 핵심
   - 군집 merge 과정 (left join, fillna(0))
   - hit_by_pitch 제거 시점과 이유
   - X_num(수치)과 X_cat(one-hot) 각각의 구체적 구성
   - feature_columns 저장이 왜 중요한지
   - LabelEncoder의 알파벳순 정렬 결과
   - stratify 옵션

2. MLP 클래스 — 신경망 구조
   - Linear, BatchNorm, ReLU, Dropout 각각이 왜 필요한지
   - 출력층에 softmax가 없는 이유

3. train_model() — 학습 루프
   - class_weights 계산 과정 (실제 수치 예시 포함)
   - EarlyStopping 동작 원리
   - CrossEntropyLoss가 뭔지

4. predict_proba() — MDP/DQN이 호출하는 추론 메서드

5. EXPERIMENTS 리스트 — 5개 실험의 ablation study 설계 논리

발표 예상 질문:
- "58%밖에 안 되는데 의미 있나요?"
- "왜 4클래스인가요?"
- "class_weights를 쓰면 왜 accuracy가 내려가나요?"
```

### 5단계 — MDP

```
src/mdp_solver.py를 읽고 MDP 단계를 설명해줘.

MDP가 뭔지 개념부터 시작해서:
1. 상태(State) — 9,216개가 어떻게 나오는지
2. 행동(Action) — 52개가 어떻게 나오는지
3. 전이확률 — MLP predict_proba()를 어떻게 호출하는지
4. 보상(Reward) — RE24 매트릭스가 뭔지
5. 가치반복(Value Iteration) — 벨만 방정식 구현
6. 최적 정책 — 각 상태의 최선의 구종+존
7. hit_into_play 세부 처리

발표 예상 질문:
- "MDP가 있는데 왜 DQN도 필요한가요?"
- "RE24가 뭔가요?"
```

### 6단계 — 강화학습 환경

```
src/pitch_env.py를 읽고 PitchEnv 클래스를 설명해줘.

1. 왜 Gym 환경을 만들었는지
2. observation_space와 action_space 정의
3. step() 함수 — 한 투구 시뮬레이션 과정
4. reset() — 이닝 초기화
5. done 조건 — 3아웃 시 종료
6. MDP의 수학적 풀이 vs PitchEnv의 시뮬레이션 차이
```

### 7단계 — DQN 학습

```
src/rl_trainer.py를 읽고 DQN 학습 과정을 설명해줘.

1. DQN이 뭔지 — Q-Learning과 뭐가 다른지
2. Stable-Baselines3 DQN 사용법
3. 학습 설정 — timesteps, 하이퍼파라미터들
4. 평균 보상 +0.436의 의미
5. W&B 로깅

발표 예상 질문:
- "DQN 보상 0.436이 좋은 건가요?"
- "왜 PPO 대신 DQN인가요?"
```

### 8단계 — 오케스트레이터

```
src/main.py를 읽고 전체 실행 흐름을 설명해줘.

1. USE_UNIVERSAL_MODEL 플래그의 역할
2. 실행 순서: 데이터 → MLP → MDP → DQN
3. universal_model_trainer.py main()과의 관계
4. 결과 파일 저장 경로
```

### 9단계 — 발표용 종합 정리

```
지금까지 분석한 모든 소스코드를 바탕으로 정리해줘:

1. 한 페이지 요약 (비전공자 교수님용)
2. 기술 요약 (전공 교수님용)
3. 예상 질문 Top 10과 모범 답변
4. 약점 인정 + 향후 계획
```

---

## B. 다음 작업용 프롬프트 (실행 순서대로)

### B-1: Task 12 Phase 2 — 물리 피처 lookup 테이블 (우선순위 1)

```
AI_CONTEXT.md와 TODO.md를 읽고 현재 상태를 확인한 후,
Task 12 Phase 2 (물리 피처 lookup 테이블)를 구현해줘.

현재 문제:
- Exp5 모델은 release_speed_n, pfx_x_n, pfx_z_n 3개 피처로 학습됨
- data/feature_columns_universal.json에 이미 이 3개가 포함 (43차원)
- data/model_config_universal.json: {"hidden_dims": [256,128,64], "dropout_rate": 0.2}
- 그런데 PitchEnv._sample_outcome()과 MDPOptimizer.predict_proba()에서
  이 3개 피처가 0으로 채워져 있음
- Task 14 분석에서 이것이 MDP 정책 Knuckleball 70.5% 편중의 한 원인

해결 방법:
1. 학습 데이터(2023 MLB 72만건)에서 
   (pitcher_cluster, mapped_pitch_name)별 평균 
   release_speed, pfx_x, pfx_z를 계산
2. CSV로 저장: data/physical_feature_lookup.csv
3. PitchEnv과 MDPOptimizer에서 추론 시 이 lookup 테이블을 참조해서
   물리 피처 3개를 채움

수정할 파일:
- src/pitch_env.py: _sample_outcome()에서 input_df 구성 시 lookup 적용
- src/mdp_solver.py: predict_proba 호출 시 lookup 적용
- src/universal_model_trainer.py 또는 scripts/generate_physical_lookup.py:
  (pitcher_cluster, mapped_pitch_name)별 평균 물리 피처 CSV 생성
  pybaseball cache + _preprocess_raw() 재사용

수정 후:
- data/mdp_optimal_policy.pkl 삭제하고 evaluate_baselines.py 재실행
- 결과가 개선됐는지 확인 (특히 MDP 정책의 Knuckleball 편중이 줄었는지)
```

### B-2: Task 16 — MDP 수렴 개선 (우선순위 2)

```
AI_CONTEXT.md의 Task 14 분석 결과를 읽고,
src/mdp_solver.py의 solve_mdp()를 개선해줘.

Task 14에서 발견된 문제:
- 현재: for iteration in range(5) 고정
- iter 5에서 max|ΔV| = 0.145 (미수렴)
- iter 10에서 max|ΔV| = 0.0076 (수렴에 근접)

수정 사항:
1. 반복 횟수를 5 → 최대 20으로 변경
2. 조기 종료 조건 추가: max|ΔV| < 1e-4이면 수렴으로 판단하고 종료
3. γ(감마, 할인율)를 1.0 → 0.99로 변경
   → foul self-loop에서 가치가 무한 누적되는 것 방지
4. 각 iteration마다 max|ΔV|를 출력해서 수렴 과정 확인

수정 후:
- data/mdp_optimal_policy.pkl 삭제
- src/evaluate_baselines.py 재실행해서 MDP 성능 변화 확인
- 특히 MDP의 정책 분포(구종 편중)가 개선됐는지 확인
- docs/baseline_by_cluster.md 결과표 업데이트
```

### B-3: 군집 1~3 DQN 학습 (시간 오래 걸림 — overnight 추천)

```
AI_CONTEXT.md와 TODO.md를 읽고 현재 상태를 확인한 후,
군집 1, 2, 3에 대해 DQN을 학습시키고 평가해줘.

## 현재 상태
- 군집 0(Cole 2019): DQN 학습 완료, 평균 보상 +0.436 ± 1.255
- 군집 1, 2, 3: DQN 미학습

## 학습 방법
src/rl_trainer.py와 src/main.py의 DQN 학습 로직을 참고해서,
각 군집별로 독립적인 DQN을 학습시켜줘:

각 군집(pitcher_cluster=1, 2, 3)에 대해:
1. 범용 모델(best_transition_model_universal.pth) 로드
2. feature_columns_universal.json에서 pitch_names, zones 파싱
3. PitchEnv(transition_model, pitch_names, zones, pitcher_cluster=cid) 생성
4. Stable-Baselines3 DQN 학습:
   - total_timesteps=300_000 (군집 0과 동일)
   - 나머지 하이퍼파라미터도 main.py의 기존 설정과 동일하게
5. 학습된 모델 저장: data/dqn_cluster_{cid}.zip
6. 학습 후 1000 에피소드 평가해서 평균 보상 ± 표준편차 측정

## 평가 및 시각화
학습 완료 후:
1. docs/baseline_by_cluster.md의 DQN 열을 실제 값으로 업데이트
   (현재 "미학습"으로 표시된 군집 1~3)
2. docs/presentation_baseline_by_cluster.png 차트를 재생성
   - 군집 1~3에도 DQN 빨간 막대가 추가된 완성된 차트
   - 모든 군집에서 DQN 수치 라벨 표시
3. docs/presentation_summary_table.png도 재생성
   - DQN 열의 "—"를 실제 값으로 교체

## 스크립트
- 학습 + 평가를 한 번에 실행하는 스크립트를
  scripts/train_dqn_all_clusters.py로 저장
- 이미 학습된 군집(data/dqn_cluster_{cid}.zip 존재)은 스킵하는 옵션 포함
- W&B에 각 군집별 별도 run으로 기록
  (project="SmartPitch-Portfolio", name="DQN_cluster_{cid}")

## 주의사항
- 군집당 30만 timesteps × 3군집 = 90만 timesteps, 1.5~3시간 소요
- 기존 군집 0 DQN 모델(smartpitch_dqn_final.zip)은 건드리지 마
- 학습 중 진행 상황을 터미널에 출력 (몇 만 step마다 평균 보상)
- 학습 완료 후 AI_CONTEXT.md, TODO.md도 업데이트
- action space: feature_columns 파싱 기준 9구종 × 13존 = 117
  기존 군집 0 DQN(Cole)은 PitchClustering으로 4구종만 사용해서 ~52
  비교표에 action space 크기를 반드시 명시
```

### B-4: 차트 수정 (B-3 완료 후)

```
docs/ 폴더의 발표용 차트 3개를 수정해줘.

차트 1: docs/presentation_baseline_overall.png
- 가로 막대그래프(horizontal bar)로 변경
- DQN 막대: 파란색(#2563EB), 나머지: 회색(#9CA3AF)
- 에러바 제거, 막대 끝에 평균값만 표시
- figsize=(10, 5), dpi=200

차트 2: docs/presentation_baseline_by_cluster.png
- 군집 0~3 전부에 DQN 막대 표시 (B-3에서 학습 완료된 값 반영)
- 막대 위에 수치 라벨, figsize=(12, 6), dpi=200

차트 3: docs/presentation_summary_table.png
- DQN 열의 모든 군집에 실제 값 표시
- 첫 번째 행 텍스트 잘리지 않게 열 너비 확대
- figsize=(14, 6), dpi=200
```

### B-5: 예상 질문 답변 정리

```
SmartPitch 발표에서 나올 수 있는 교수님 질문 Top 15와
모범 답변을 정리해줘.

지금까지 완료된 작업 기반으로:
- MLP 58%, 4클래스, macro F1 0.495
- 베이스라인 5종 비교: DQN +0.436 > Random +0.231 > MDP +0.151
- MDP가 Random보다 못한 이유 (VI 미수렴 + MLP calibration + sample gap)
- 투수 군집별 최적 전략이 다름
- 물리 피처 Phase 2 미완

특히 이런 질문 대비:
1. "58%면 낮지 않나요?"
2. "베이스라인이 뭔가요?"
3. "MDP가 있는데 왜 DQN이 필요하죠?"
4. "MDP가 랜덤보다 못하다는 게 말이 되나요?"
5. "실제 야구에서 쓸 수 있나요?"
6. "왜 4클래스인가요?"
7. "DQN 보상 0.436이 좋은 건가요?"
8. "구종 분포가 균형 잡혔다는 기준이 뭔가요?"
9. "Knuckleball 70.5% 편중은 뭔가요?"
10. "물리 피처가 왜 중요한가요?"

각 답변은 30초 이내로 말할 수 있는 분량으로,
기술적 근거 + 쉬운 비유를 포함해줘.
AI_CONTEXT.md와 docs/ 폴더의 분석 문서를 참고해서 답해줘.
```

### B-6: 커밋/푸시 (작업 완료 후 마지막에)

```
현재까지 작업 내용을 AI_CONTEXT.md, README.md,
CLAUDE.md, TODO.md에 반영하고 푸시해줘.

모든 코드를 확인해서:
1. 새로 추가/변경된 파일 목록 확인
2. 4개 문서 동기화 (완료 작업, 우선순위 변경, 성능 수치)
3. git add + commit + push
4. untracked 파일 중 추적해야 할 것이 있는지 확인
```

---

## 팁

- **0단계는 매번 넣어.** 새 세션마다 안 넣으면 Claude Code가 이전 상태를 모름.
- **A(학습)와 B(작업)는 독립적.** 둘 다 할 필요 없고, 필요한 것만 골라.
- **B는 순서대로:**
  B-1(물리피처 lookup) → B-2(MDP 수렴 개선) →
  data/mdp_optimal_policy.pkl 삭제 + evaluate_baselines.py 재실행으로 개선 확인 →
  B-3(DQN 학습) → B-4(차트) → B-5(질문) → B-6(커밋)
- **B-3는 시간이 오래 걸려.** plan mode로 계획 확인 후, overnight으로 실행 추천.
- **B-6은 항상 마지막에.** 작업 끝날 때마다 문서 동기화 + push.
- 이해 안 되면 바로 "○○ 부분 더 쉽게 설명해줘"라고 추가 질문.
- Cursor로 프롬프트 검증받고 넣으면 한 번에 깔끔하게 돼.