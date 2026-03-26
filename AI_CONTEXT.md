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

### 코어 파이프라인 (단일 투수 버전)
- [x] `data_loader.py`: Statcast 수집, 전처리, W&B Artifact 업로드
  - `cache.enable()` 추가 (반복 다운로드 방지)
  - `'pitcher'` 컬럼을 meta에 추가 (투수 군집 merge용)
  - `self.pitcher_mlbam_id` 저장 (main.py에서 군집 조회용)
- [x] `clustering.py`: 개별 투수 구종 UMAP+KMeans 자동 식별 (K=3~6 실루엣)
- [x] `model.py`: PyTorch MLP 전이 확률 모델
  - `pitcher_cluster` 원-핫 인코딩 추가 (pitcher_clusters_2023.csv merge)
  - `batter_cluster` + `pitcher_cluster` 동시 원-핫 인코딩
- [x] `mdp_solver.py`: MDP 가치반복 (Value Iteration 5회)
  - 상태 키 형식: `"count_outs_runners_batter_pitcher"` (5-파트)
  - `pitcher_clusters` 파라미터로 외부 주입
  - 총 상태 수: 12×3×8×8×K (K=4이면 9,216개)
- [x] `pitch_env.py`: Gymnasium 8D 관측 공간
  - `[balls, strikes, outs, 1b, 2b, 3b, batter_cluster, pitcher_cluster]`
  - `pitcher_cluster=int` → 단일 투수 고정 모드
  - `pitcher_cluster=None` → 에피소드마다 랜덤 (범용 모드)
- [x] `rl_trainer.py`: DQN 학습 + W&B 로깅, 8D obs 대응
- [x] `main.py`: 전체 오케스트레이터
  - `_lookup_pitcher_cluster()`: pitcher_clusters_2023.csv 조회
  - `_get_all_pitcher_clusters()`: 전체 군집 ID 반환
  - MDPOptimizer + PitchEnv에 pitcher_clusters/pitcher_cluster 전달

### 군집화 (독립 실행 스크립트)
- [x] `batter_clustering.py`: 2023 MLB 타자 K=8 군집화
  - `pull_pct`: 하드코딩 0.40 → `hc_x` 좌표 기반 실제 계산으로 교체
    - 우타(R): `hc_x < 125` = 좌측/3루 방향 = Pull
    - 좌타(L): `hc_x > 125` = 우측/1루 방향 = Pull
  - `cache.enable()` 추가
  - 출력: `data/batter_clusters_2023.csv`
- [x] `pitcher_clustering.py`: 2023 MLB 투수 K=4 군집화 **[신규 생성 완료]**
  - 479명 (500구 이상), 15개 피처
  - K=4 선택됨 (실루엣 0.4502)
  - 출력: `data/pitcher_clusters_2023.csv` **[생성 완료]**

---

## 투수 군집 결과 (2023 시즌, K=4)

```
군집 0 (157명) — 파워 패스트볼/슬라이더
  구속 89.9, FF 44.7%, SL 25.6%, Whiff 25.9%, 구종수 3.4
  → 현대 파워 불펜형, 직구-슬라이더 2구종 중심

군집 1 (102명) — 핀세스/커맨드
  구속 87.9, FF 28.4%, SL 9.6%, Whiff 22.9%, zone% 49.5%
  → 제구 중심, 변화구 다양, 베테랑 선발형

군집 2 (103명) — 무브먼트/싱커볼
  구속 89.9, FF 15.1%, SL 18.6%, Whiff 22.0%
  → 투심/싱커 중심, 땅볼 유도 스타일

군집 3 (117명) — 멀티피치/아스날
  구속 89.0, FF 29.0%, SL 8.5%, Whiff 24.0%, 구종수 4.4
  → Darvish/Kershaw형 다구종 선발
```

---

## 현재 성능 수치 (Kershaw 2024, W&B: cuafju1e)

```
MLP val_accuracy : 47.1%  ← 낮음, epoch 증가 필요
MLP val/train 갭: 0.43    ← 과적합 징후
DQN 평균 보상   : 0.235   ← 이닝당 기대실점 억제
DQN 주요 구종   : Slider 59.6%, Fastball 18.7%
```

---

## 다음 작업 우선순위 (Next Steps)

### [우선순위 1] 범용 전이 확률 모델 학습 — 핵심 병목
현재 `model.py`는 단일 투수 데이터로만 학습됨.
범용 AI를 위해 전체 MLB 데이터로 재학습 필요.

```python
# 구현 방향: universal_pipeline.py 또는 main.py 확장
# batter_clustering의 raw_df를 재활용 (이미 전 MLB 데이터)
# model.py에 universal=True 모드 추가:
#   - 입력: 전체 2023 시즌 투구 데이터 (~72만 건)
#   - pitcher_cluster: pitcher_clusters_2023.csv로 merge
#   - batter_cluster: batter_clusters_2023.csv로 merge
#   - epochs: 5 → 20, batch_size: 64 → 1024
#   - 목표 val_acc: 65% 이상
```

### [우선순위 2] MLP 하이퍼파라미터 개선
```python
# main.py wandb config 수정:
epochs = 20          # 5 → 20
batch_size = 256     # 64 → 256 (데이터 많아질 때)
hidden_dims = [256, 128, 64]  # 더 깊은 망 고려
# EarlyStopping 추가 고려 (patience=5)
```

### [우선순위 3] DQN 학습 강화
```python
# rl_trainer.py build() 파라미터:
total_timesteps = 500_000      # 300K → 500K
exploration_fraction = 0.40    # 0.30 → 0.40
net_arch = [256, 128]          # [128, 64] → 더 큰 망
```

### [우선순위 4 — 미래] 실시간 추천 API
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

5. **CUDA vs CPU 전략**:
   - `pyproject.toml`에는 `torch>=2.10.0` (CPU 기본)
   - GPU 환경: `uv pip install torch --index-url .../cu124` (uv.lock 무시)
   - `uv sync` 실행 시 CPU 버전으로 되돌아가므로 GPU 재설치 필요

---

## 기술 부채 / 알려진 문제

```
[Critical]
- model.py: 단일 투수 데이터로만 학습됨 → 범용 모델 미완성
- pitcher_clusters_2023.csv: 생성됨, 하지만 main.py 재실행 아직 안 함

[High]
- RE24 매트릭스: 2019년 기준 하드코딩 → 연도별 갱신 로직 필요
- MLP epochs=5: val_acc 47% → 과소학습, 최소 20 필요

[Medium]
- batted ball 전이 확률 (70/15/10/5%): 실제 MLB 데이터 기반으로 교체 필요
- pitcher_clustering.py: 좌/우투 분리 없음 (릴리스 포인트가 대리 지표로 작동)

[Low]
- pull_pct에 hc_x=125 임계값: 타자마다 pull 정의가 다를 수 있음 (125 ± 10)
- WandB Sweep (하이퍼파라미터 자동 탐색) 미구현
```
