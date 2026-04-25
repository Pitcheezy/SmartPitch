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
- [x] `mdp_solver.py`: MDP 가치반복 (Value Iteration 최대 20회, γ=0.99), 5-파트 상태 키, 총 상태 9,216개, 군집별 유효 구종 필터링
- [x] `pitch_env.py`: Gymnasium 8D 관측 공간. pitcher_cluster=int(고정)/None(랜덤). `get_valid_pitches()` 군집별 유효 구종 필터링 유틸리티
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

### Task 12 — 투구 물리 피처 추가 (완료, Phase 1 + Phase 2)
- [x] `universal_model_trainer.py`: `required`에 `release_speed/pfx_x/pfx_z` 추가
- [x] `_add_physical_features()`: 정규화 수식 하드코딩 `(v-90)/5, pfx_x, pfx_z` 그대로
- [x] `model.py` X_num: 정규화된 컬럼 auto-detect 추가
- [x] 결과: Exp4 val_acc 58.3% (+0.2pp), Exp5 macro F1 0.495
- [x] Phase 2: `scripts/generate_physical_lookup.py` → `data/physical_feature_lookup.csv` (34행)
  - `pitch_env.py`, `mdp_solver.py`에서 (pitcher_cluster, mapped_pitch_name)별 평균 물리 피처 lookup
  - MDP 정책 Knuckleball 70.5% 편중 해소 → 구종 entropy 1.3으로 다양화
  - MDP 성능: +0.151 → +0.247 (+63%), 4군집 중 3군집에서 MDP가 베이스라인 1위

### Task 13~15 — 베이스라인 평가 + MDP 분석 + 발표 자료 (완료, 커밋 3fd2352)
- [x] `src/evaluate_baselines.py`: Random / MostFrequent / Frequency / MDPPolicy / DQN (ref) 5종 비교
  - 전체 평가: cluster 0(Cole 2019), 각 1000 에피소드, 동일 seed(0~999)
  - 군집별 평가: K=4, 각 (군집 × 에이전트) 1000 에피소드
  - `data/mdp_optimal_policy.pkl` 캐시 (9,216 상태 VI 결과, ~20분 → 재실행 수 초)
  - `MDPPolicyAgent._obs_to_state_key()`: 8D obs → `"b-s_outs_runners_bc_pc"` 5-파트 키
- [x] `scripts/analyze_mdp_vs_env.py`: `mdp_solver.solve_mdp()` vs `pitch_env.step()` 보상·전이 줄 단위 비교
  + VI 수렴 측정(iter 10까지) + 정책 행동 분포 + 1 episode trace
- [x] `scripts/generate_baseline_presentation.py`: 발표용 PNG 3종 (overall bar, grouped bar, summary table)
- [x] 산출 문서: `docs/baseline_comparison.md`, `docs/baseline_by_cluster.md`, `docs/mdp_vs_env_reward_analysis.md`
- **결과 (cluster 0, 1000 ep, 물리피처 lookup 적용 후)**:
  DQN +0.436 > MDP **+0.247** > MostFrequent +0.220 > Random +0.204 > Frequency +0.175
- **Task 12 Phase 2 적용 효과**: MDP가 +0.151 → +0.247 (+63%), Knuckleball 편중 해소(entropy 1.3)

### Task 17 — 군집 1~3 DQN 학습 (완료, Task 18에서 재학습)
- [x] `scripts/train_dqn_all_clusters.py`: 군집별 DQN 학습 + 평가 스크립트
- [x] 초기 결과 (117 action space): Knuckleball 편중 36~58% 발생
- [x] Task 18 적용 후 유효 구종만으로 재학습하여 해결

### Task 18 — Action Space 최적화 (완료)
- [x] `get_valid_pitches()` 함수 추가 (`pitch_env.py`): physical_feature_lookup.csv에서 1% 미만 구종 자동 제거
- [x] `mdp_solver.py`: `valid_pitches_by_cluster` 파라미터 추가, 군집별 유효 구종만으로 MDP 풀이
- [x] `evaluate_baselines.py`: 군집별 유효 구종 필터링 적용, MDP/에이전트 모두 동일 action space
- [x] `train_dqn_all_clusters.py`: `get_valid_pitches()` 연동, 군집별 DQN 재학습
- [x] Knuckleball 편중 0%로 완전 해소, 현실적 구종 분포 달성
- [x] 군집별 action space: 군집 0=104 (8구종), 군집 1=91 (7구종), 군집 2~3=104 (8구종)
- [x] MDP 전 군집 재계산 (VI 18회 수렴), DQN 군집 1~3 재학습 (300K timesteps)
- **결과**: 군집 0에서는 DQN이 최고 (+0.436), 군집 1~3에서는 MDP가 최고

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
DQN 평균 보상    : 0.436   (100이닝 평가, action space ~52)
DQN 주요 구종    : Fastball 51.3%, Slider 24.3%, Curveball 14.9%, Changeup 10.7%

[DQN — 군집별 범용 모델 학습 (Task 18: 유효 구종만 사용, 300K timesteps, 1000 ep 평가)]
Cluster 1: +0.188 ± 1.127  (Slider 30%, Fastball 28%, Curveball 15%) — 91 actions (7구종)
Cluster 2: +0.242 ± 1.130  (Fastball 56%, Sinker 12%, Slider 10%) — 104 actions (8구종)
Cluster 3: +0.215 ± 1.157  (Fastball 45%, Splitter 13%, Curveball 12%) — 104 actions (8구종)
※ Knuckleball 편중 0%로 완전 해소

[베이스라인 비교 — evaluate_baselines.py, 1000 ep, Task 18 action space 최적화 후]
군집 0: DQN +0.436 > MDP +0.258 > MostFreq +0.220 > Random +0.185 > Freq +0.175 (DQN 최고)
군집 1: MDP +0.247 > DQN +0.188 > Freq +0.169 > MostFreq +0.140 > Random +0.136 (MDP 최고)
군집 2: MDP +0.262 > DQN +0.242 > Random +0.229 > Freq +0.203 > MostFreq +0.201 (MDP 최고)
군집 3: MDP +0.256 > MostFreq +0.251 > DQN +0.215 > Freq +0.202 > Random +0.171 (MDP 최고)

[개인 맞춤 DQN — 2024+2025 시즌, 동일 action space에서 5-agent 비교, 1000 ep]
Cease (39 actions, K=3): MDP +0.251 > Freq +0.233 > MostFreq +0.220 > DQN +0.198 > Random +0.196
Gallen (52 actions, K=4): DQN +0.239 ≈ MDP +0.236 > Random +0.224 > MostFreq +0.220 > Freq +0.204
통계적 유의성: 모든 비교에서 p > 0.29, Cohen's d < 0.05 (negligible)
```

---

## 다음 작업 우선순위 (Next Steps)

### ~~[우선순위 1] 물리 피처 Phase 2 — PitchEnv/MDPOptimizer lookup 테이블~~ (완료)
`scripts/generate_physical_lookup.py` → `data/physical_feature_lookup.csv` (34행) 생성.
`pitch_env.py`, `mdp_solver.py`에서 (pitcher_cluster, mapped_pitch_name)별 평균 물리 피처 lookup 적용.
MDP 성능 +0.151 → +0.247 (+63%), Knuckleball 편중 해소(entropy 0→1.3), 3/4 군집에서 MDP 1위.

### ~~[우선순위 2] MDP solve_mdp 수렴 개선 (Task 16)~~ (완료)
- VI 5회 고정 → 최대 20회 + max|ΔV| < 1e-4 조기종료 (17회에서 수렴)
- γ=1.0 → γ=0.99로 foul self-loop 무한 누적 방지
- 효과: cluster 0 MDP +0.247 → +0.250 (미미한 개선, 안정성 확보)
- 군집 3 부진(+0.048)은 VI 미수렴이 아닌 MLP calibration 문제로 확인

### ~~[Task 19] Cease/Gallen 개인 맞춤 DQN 학습 + 평가~~ (완료)
- [x] `scripts/main_cease.py`, `scripts/main_gallen.py`: 2024+2025 시즌 데이터로 개인 DQN 학습
- [x] Cease: 5,400 투구, K=3 (FF/SL/CH), 39 actions, DQN +0.198 ± 1.177
- [x] Gallen: 4,572 투구, K=4 (FF/SL/CH/CB), 52 actions, DQN +0.239 ± 1.134
- [x] `scripts/evaluate_personal_dqn.py`: 5-agent 비교 + Welch t-test + Cohen's d
- [x] 통계 결과: 모든 비교에서 p > 0.29, Cohen's d < 0.05 (negligible)
  → DQN과 베이스라인 간 유의미한 차이 없음 (야구의 본질적 변동성 σ ≈ 1.15)
- [x] `docs/MODEL_USAGE.md`: 백엔드 통합 가이드 (모델 로드, 추론, Action 변환)
- [x] 산출물: `docs/baseline_cease.md`, `docs/baseline_gallen.md`, `docs/personal_dqn_report.md`, `docs/re24_decision.md`
- [x] 모델 파일: `dqn_cease_2024_2025.zip`, `dqn_gallen_2024_2025.zip`

### [우선순위 3] 범용 모델 구종 군집화 통합 (Task 10)
학습-추론 구종 분류 불일치 문제 (학습: Statcast pitch_type, 추론: 물리 군집).

### [우선순위 4] RE24 매트릭스 연도별 갱신
현재 pitch_env.py와 mdp_solver.py 두 파일에 2019 MLB 기준값이 하드코딩되어 있다.
두 파일을 **반드시 동시에** 수정해야 한다.

### [우선순위 5] 인플레이 타구 확률 실데이터 기반 교체
현재 아웃 70%, 1루타 15%, 2루타 10%, 홈런 5%로 임의 설정.
pitch_env.py + mdp_solver.py 두 곳 동시 수정 필요.

### ~~[우선순위 6] 군집 1~3 DQN 학습~~ (완료, Task 18에서 재학습)
- `scripts/train_dqn_all_clusters.py` 작성, 군집 1~3 각 300K 타임스텝 학습
- Task 18 적용 후 유효 구종만 사용하여 재학습 (Knuckleball 편중 완전 해소)
- 결과: cluster 1 +0.188 (91 actions), cluster 2 +0.242 (104 actions), cluster 3 +0.215 (104 actions)
- 군집 0: DQN 최고, 군집 1~3: MDP 최고
- 모델 저장: `data/dqn_cluster_{1,2,3}.zip`

### ~~[우선순위 — Task 18] Action Space 최적화~~ (완료)
- `get_valid_pitches()`: physical_feature_lookup.csv에서 1% 미만 구종 자동 제거
- Knuckleball 편중 0%로 완전 해소, 현실적 구종 분포 달성
- MDP 전 군집 재계산 (VI 18회 수렴, γ=0.99)
- DQN 군집 1~3 재학습 (유효 구종만, 300K timesteps)
- 수정 파일: `pitch_env.py`, `mdp_solver.py`, `evaluate_baselines.py`, `train_dqn_all_clusters.py`

### [우선순위 7 — 미래] 실시간 추천 API
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
- RE24 매트릭스: 2019년 기준 하드코딩 → 연도별 갱신 로직 필요
  (pitch_env.py, mdp_solver.py 두 곳 동시 수정)

[Medium]
- batted ball 전이 확률 (70/15/10/5%): 실제 MLB 데이터 기반으로 교체 필요
- pitcher_clustering.py: 좌/우투 분리 없음 (릴리스 포인트가 대리 지표로 작동)

[Low]
- pull_pct에 hc_x=125 임계값: 타자마다 pull 정의가 다를 수 있음 (125 ± 10)
- WandB Sweep (하이퍼파라미터 자동 탐색) 미구현
```
