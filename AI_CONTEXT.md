# AI_CONTEXT.md
# SmartPitch 프로젝트 AI 작업 컨텍스트
# 이 파일은 Claude Code와 협업할 때마다 업데이트합니다.
# 새 대화를 시작할 때 이 파일을 먼저 읽으면 컨텍스트를 빠르게 복구할 수 있습니다.

---

## 프로젝트 한 줄 요약
MLB 투수에게 볼카운트·주자상황·타자유형·투수유형을 고려한 최적 구종+코스를 추천하는 강화학습 파이프라인.

---

## 기술 스택
- **패키지 관리**: `uv` (Python 3.12+), `pyproject.toml` + `uv.lock`
- **딥러닝**: `PyTorch` (CUDA 지원, 로컬에서만 `uv pip install torch --index-url .../cu124`)
- **강화학습**: `Stable-Baselines3` DQN + `Gymnasium` 커스텀 환경
- **데이터**: `pybaseball` (MLB Statcast), `cache.enable()` 전역 적용됨
- **MLOps**: `Weights & Biases` — 프로젝트: `pitcheezy/SmartPitch-Portfolio`
- **군집화**: `UMAP` + `K-Means` + `실루엣 점수`

---

## 완료된 작업 전체 목록

### 코어 파이프라인
- [x] `data_loader.py`: Statcast 수집, 전처리, W&B Artifact 업로드. `cache.enable()`, `self.pitcher_mlbam_id` 저장
- [x] `clustering.py`: 개별 투수 구종 UMAP+KMeans 자동 식별 (K=3~6 실루엣)
- [x] `model.py`: PyTorch MLP 전이 확률 모델
  - 4클래스 출력 (ball/strike/foul/hit_into_play), hit_by_pitch 제거
  - batter_cluster + pitcher_cluster 동시 원-핫 인코딩
  - EarlyStopping(patience=5, val_loss 기준)
  - class-weighted CrossEntropyLoss 옵션 (`use_class_weights`)
  - `load_from_checkpoint(model_config_path)`: `model_config_universal.json`에서 hidden_dims 로드
- [x] `mdp_solver.py`: MDP 가치반복 (Value Iteration 5회), 5-파트 상태 키, 총 상태 9,216개
- [x] `pitch_env.py`: Gymnasium 8D 관측 공간. pitcher_cluster=int(고정)/None(랜덤)
- [x] `rl_trainer.py`: DQN 학습 + W&B 로깅
- [x] `main.py`: USE_UNIVERSAL_MODEL=True 플래그, `_lookup_pitcher_cluster()`, `_get_all_pitcher_clusters()`

### 군집화 (독립 실행 스크립트)
- [x] `batter_clustering.py`: 2023 MLB 타자 K=8 군집화, pull_pct hc_x 실측 계산
- [x] `pitcher_clustering.py`: 2023 MLB 투수 K=4 군집화 (실루엣 0.4502, 479명)

### 범용 전이 모델
- [x] `universal_model_trainer.py`: 2023 MLB 전체 72만 건으로 5가지 실험 실행
  - Exp1 BiggerModel [256,128,64]: val_acc=58.1%
  - Exp2 LRScheduler [128,64]: val_acc=58.0%
  - Exp3 ClassWeights [128,64]: val_acc=57.3% (foul/hit_into_play F1 소폭 개선)
  - **Exp4 PhysicalFeatures** [256,128,64] + release_speed/pfx_x/pfx_z: val_acc=**58.3%** (Top-1 최고)
  - **Exp5 CW+Physical** [256,128,64] + class_weights + 물리 피처: val_acc=57.5%, **macro F1=0.495** (소수 클래스 recall 개선)
  - **현재 canonical**: Exp5 — MDP는 `predict_proba()` 분포를 사용하므로 macro F1 우선
  - 선정 기준: `max(val_acc)` (단일 실험 실행 시) / 수동 선정 (Exp5 유지)
  - `data/model_config_universal.json` 저장: `{"hidden_dims": [256,128,64], "dropout_rate": 0.2}`
  - 산출물: `best_transition_model_universal.pth`, `data/feature_columns_universal.json`, `data/target_classes_universal.json`, `data/model_config_universal.json`

### Task 12 — 투구 물리 피처 추가 (완료, Phase 1)
- [x] `universal_model_trainer.py`: `required`에 `release_speed/pfx_x/pfx_z` 추가
- [x] `_add_physical_features()`: 정규화 수식 하드코딩 `(v-90)/5, pfx_x, pfx_z` 그대로
- [x] `model.py` X_num: 정규화된 컬럼 auto-detect 추가
- [x] 결과: Exp4 val_acc 58.3% (+0.2pp), Exp5 macro F1 0.495
- [ ] Phase 2 (후속): (pitcher_cluster × mapped_pitch_name)별 평균 물리 피처 lookup 테이블 추가 필요.
      현재 PitchEnv/MDPOptimizer 추론 시 해당 컬럼이 0으로 채워져 정확도 저하 위험.

---

## 투수 군집 결과 (2023 시즌, K=4)

```
군집 0 (157명) — 파워 패스트볼/슬라이더
  구속 89.9, FF 44.7%, SL 25.6%, Whiff 25.9%, 구종수 3.4

군집 1 (102명) — 핀세스/커맨드
  구속 87.9, FF 28.4%, SL 9.6%, Whiff 22.9%, zone% 49.5%

군집 2 (103명) — 무브먼트/싱커볼
  구속 89.9, FF 15.1%, SL 18.6%, Whiff 22.0%

군집 3 (117명) — 멀티피치/아스날
  구속 89.0, FF 29.0%, SL 8.5%, Whiff 24.0%, 구종수 4.4
```

---

## 현재 성능 수치

```
[범용 모델 — universal_model_trainer.py, Exp5 CW+Physical [256,128,64] — 현재 canonical]
MLP val_accuracy : 57.5%   (4클래스: ball/strike/foul/hit_into_play)
MLP macro F1     : 0.495   (foul 0.285, hit_into_play 0.415 — 소수 클래스 recall 개선)

[참고: Top-1 Accuracy 최고는 Exp4 PhysicalFeatures]
Exp4 val_accuracy: 58.3%  Top-2: 80.8%  Top-3: 95.1%  val_loss: 0.9995

[DQN — Gerrit Cole 2019 파이프라인 실행, W&B run: h4n3o0di]
DQN 평균 보상    : 0.436   (100이닝 평가)
DQN 주요 구종    : Fastball 51.3%, Slider 24.3%, Curveball 14.9%, Changeup 10.7%
```

---

## 다음 작업 우선순위 (Next Steps)

### [우선순위 1] 물리 피처 Phase 2 — PitchEnv/MDPOptimizer lookup 테이블
Exp5/Exp4 모델을 DQN 파이프라인에서 쓰려면 `release_speed_n/pfx_x_n/pfx_z_n` 입력이
필요하다. 현재는 `input_df`가 0으로 초기화되어 정확도 저하.
**해결안**: 학습 데이터에서 (pitcher_cluster, mapped_pitch_name)별 평균값 CSV 생성 후
PitchEnv `_sample_outcome()` / MDPOptimizer `predict_proba` 호출 시 lookup.

### [우선순위 2] 범용 모델 구종 군집화 통합 (Task 10)
학습-추론 구종 분류 불일치 문제 (학습: Statcast pitch_type, 추론: 물리 군집).

### [우선순위 3] RE24 매트릭스 연도별 갱신
현재 pitch_env.py와 mdp_solver.py 두 파일에 2019 MLB 기준값이 하드코딩되어 있다.
두 파일을 **반드시 동시에** 수정해야 한다.

### [우선순위 4] 인플레이 타구 확률 실데이터 기반 교체
현재 아웃 70%, 1루타 15%, 2루타 10%, 홈런 5%로 임의 설정.
pitch_env.py + mdp_solver.py 두 곳 동시 수정 필요.

### [우선순위 5] DQN 학습 강화
```python
# main.py wandb config 수정:
"dqn_total_timesteps": 500_000      # 300K → 500K
"dqn_exploration_fraction": 0.40    # 0.30 → 0.40
```

### [우선순위 6 — 미래] 실시간 추천 API
```
FastAPI 서버 구현 계획:
  POST /recommend
  입력: {balls, strikes, outs, on_1b, on_2b, on_3b, batter_id, pitcher_id}
  처리: batter_clusters.csv + pitcher_clusters.csv 조회 → 8D obs → DQN.predict()
  출력: {pitch_type, zone, expected_run_prevention}
```

---

## 실행 명령어 치트시트

```bash
# 환경 세팅 (팀원 최초 1회)
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu124  # GPU 환경만

# 군집화 CSV 생성 (최초 1회 or 데이터 갱신 시)
uv run src/batter_clustering.py     # ~10분 (캐시 후 1분)
uv run src/pitcher_clustering.py    # ~3분 (캐시 후 30초)

# 범용 모델 학습 (최초 1회, ~20~40분)
uv run src/universal_model_trainer.py

# 메인 파이프라인 실행
uv run src/main.py

# W&B 결과 로컬 분석
uv run fetch_wandb_run.py           # wandb_export/ 폴더에 CSV 저장
```

---

## 중요한 구현 결정 사항 (Design Decisions)

1. **상태 키 5-파트 형식**: `"count_outs_runners_batter_pitcher"`
   - count는 `{b}-{s}` 형식 (예: `3-2`), 나머지는 `_`로 구분
   - 파싱: `parts = state.split('_')` → 5개 원소

2. **pitcher_cluster 전달 방식**:
   - `MDPOptimizer(pitcher_clusters=["0","1","2","3"])` — 문자열 리스트
   - `PitchEnv(pitcher_cluster=0)` — 단일 int (고정 모드)
   - `PitchEnv(pitcher_cluster=None)` — 랜덤 모드

3. **타자 군집 CSV 컬럼**: `batter_id`, `stand`, `cluster`
4. **투수 군집 CSV 컬럼**: `pitcher_id`, `cluster`

5. **model_config_universal.json**: `load_from_checkpoint()` 호출 시 hidden_dims를 이 파일에서 읽음.
   파일이 없으면 [128,64] fallback. Exp1([256,128,64])이 best이므로 현재 파일에 저장됨.

6. **CUDA vs CPU 전략**:
   - `pyproject.toml`에는 `torch>=2.10.0` (CPU 기본)
   - GPU 환경: `uv pip install torch --index-url .../cu124` (uv.lock 무시)
   - `uv sync` 실행 시 CPU 버전으로 되돌아가므로 GPU 재설치 필요

---

## 기술 부채 / 알려진 문제

```
[High]
- 물리 피처 Phase 2 미완: PitchEnv/MDPOptimizer에서 release_speed_n/pfx_*_n이 0으로 채워짐
  (lookup 테이블 생성 필요)
- RE24 매트릭스: 2019년 기준 하드코딩 → 연도별 갱신 로직 필요
  (pitch_env.py, mdp_solver.py 두 곳 동시 수정)

[Medium]
- batted ball 전이 확률 (70/15/10/5%): 실제 MLB 데이터 기반으로 교체 필요
- pitcher_clustering.py: 좌/우투 분리 없음 (릴리스 포인트가 대리 지표로 작동)

[Low]
- pull_pct에 hc_x=125 임계값: 타자마다 pull 정의가 다를 수 있음 (125 ± 10)
- WandB Sweep (하이퍼파라미터 자동 탐색) 미구현
```
