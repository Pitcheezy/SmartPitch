# CLAUDE.md — SmartPitch 프로젝트 AI 작업 가이드

이 파일은 Claude Code가 새 대화를 시작할 때 프로젝트 컨텍스트를 즉시 복구하기 위한 문서입니다.
코드를 직접 읽고 작성했으며, 추측 없이 실제 구현 기반으로 기술합니다.

---

## 프로젝트 한 줄 요약

MLB Statcast 투구 데이터를 기반으로 볼카운트·주자상황·타자유형·투수유형을 고려하여
기대 실점(RE24)을 최소화하는 최적 구종+코스를 추천하는 엔드-투-엔드 강화학습 파이프라인.

---

## 기술 스택

| 영역 | 라이브러리 | 버전 | 용도 |
|------|-----------|------|------|
| 패키지 관리 | `uv` | — | `pyproject.toml` + `uv.lock` 기반 팀 동기화 |
| Python | — | 3.12 (`.python-version`으로 고정) | — |
| 데이터 수집 | `pybaseball` | ≥2.2.7 | MLB Statcast API 래퍼 |
| 전처리/분석 | `pandas` ≥3.0, `numpy` ≥2.4 | — | 데이터 조작 |
| 군집화 | `umap-learn` ≥0.5.11, `scikit-learn` ≥1.8 | — | UMAP + K-Means + 실루엣 |
| 딥러닝 | `torch` ≥2.10 | CPU 기본, GPU 별도 설치 | MLP 전이 확률 모델 |
| 강화학습 | `stable-baselines3` ≥2.3, `gymnasium` ≥0.29 | — | DQN 에이전트 |
| MLOps | `wandb` ≥0.25 | — | 실험 추적, Artifact 버전 관리 |
| 시각화 | `plotly` ≥6.6, `seaborn` ≥0.13, `matplotlib` | — | UMAP 산점도, 학습 곡선 |

> **CUDA 설치 (GPU 환경)**: `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`
> 이 명령은 `uv.lock`을 수정하지 않는다. `uv sync` 재실행 시 CPU 버전으로 되돌아감.

---

## 디렉토리 구조 및 파일 역할

```
paper_review/
├── src/
│   ├── __init__.py              빈 파일 (src를 패키지로 인식시키기 위함)
│   ├── main.py                  전체 파이프라인 오케스트레이터 (진입점)
│   ├── data_loader.py           Statcast 투구 데이터 수집 + 전처리 클래스
│   ├── clustering.py            단일 투수 구종 레퍼토리 식별 (UMAP+KMeans)
│   ├── batter_clustering.py     2023 MLB 전체 타자 군집화 (독립 실행 스크립트)
│   ├── pitcher_clustering.py    2023 MLB 전체 투수 군집화 (독립 실행 스크립트)
│   ├── universal_model_trainer.py 2023 MLB 전체 데이터 범용 MLP 학습 (독립 실행 스크립트)
│   ├── model.py                 투구 결과 전이 확률 예측 PyTorch MLP
│   ├── mdp_solver.py            MDP 가치반복(Value Iteration) 최적 정책 계산
│   ├── pitch_env.py             Gymnasium 커스텀 환경 (이닝 단위 시뮬레이션)
│   └── rl_trainer.py            DQN 에이전트 학습/평가 클래스
│
├── data/                        git에서 추적 안 함 (*.csv는 .gitignore)
│   ├── batter_clusters_2023.csv batter_clustering.py 실행 결과
│   └── pitcher_clusters_2023.csv pitcher_clustering.py 실행 결과
│
├── fetch_wandb_run.py                   W&B 런 데이터 로컬 추출 유틸리티
├── best_transition_model.pth            단일 투수 MLP 가중치 (gitignored *.pth)
├── best_transition_model_universal.pth  범용 MLP 가중치 (gitignored *.pth)
│                                        → uv run src/universal_model_trainer.py 로 생성
│                                        → 또는 W&B Artifact "universal_transition_mlp" 에서 다운로드
├── best_dqn_model/              EvalCallback 최고 DQN 체크포인트 (gitignored)
├── smartpitch_dqn_final.zip     최종 DQN 모델 (gitignored *.zip)
│
├── pyproject.toml               의존성 정의
├── uv.lock                      버전 잠금 파일 (팀 동기화 기준)
├── .python-version              "3.12" 고정
├── .gitignore                   *.pth, *.csv, *.zip, *.png, wandb/, best_dqn_model/ 등 제외
├── README.md                    팀원용 문서 (파이프라인 개요, 빠른 시작)
├── AI_CONTEXT.md                AI 작업 컨텍스트 (완료 목록, 다음 우선순위)
└── CLAUDE.md                    이 파일
```

> **gitignored 주의**: `data/*.csv`, `*.pth`, `*.zip`, `wandb/`, `best_dqn_model/`는 git 추적 안 됨.
> 팀원이 클론 후 아래 순서로 실행해야 함:
> 1. `uv run src/batter_clustering.py` / `uv run src/pitcher_clustering.py`
> 2. `uv run src/universal_model_trainer.py` (범용 모델 생성, 약 20~40분)
>    또는 W&B Artifact `universal_transition_mlp`에서 3개 파일 수동 다운로드

---

## 데이터 흐름 (코드 실행 순서)

### 선행 작업 (최초 1회, 독립 실행)

```
[Step 0a] batter_clustering.py
  statcast("2023-03-30", "2023-10-01")  ← 전체 MLB ~72만 투구
       ↓
  좌/우타 분리 → _extract_batter_features() → 타자별 9개 피처 집계
       ↓
  StandardScaler → UMAP(2D) → KMeans(K=8 고정)
       ↓
  data/batter_clusters_2023.csv  [batter_id, stand, cluster(0~7)]

[Step 0b] pitcher_clustering.py
  동일 raw_df 재활용 가능 (pybaseball cache)
       ↓
  _extract_pitcher_features() → 투수별 15개 피처 집계
       ↓
  StandardScaler → UMAP(2D) → KMeans(K=4~8 실루엣 탐색, K=4 선택)
       ↓
  data/pitcher_clusters_2023.csv  [pitcher_id, cluster(0~3)]
```

### 메인 파이프라인 (main.py 실행 시 순서)

```
[Step 1] PitchDataLoader.load_and_prepare_data()
  pybaseball.playerid_lookup() → MLBAM ID 취득
  statcast_pitcher(start, end, player_id) → 단일 투수 raw 데이터
  전처리: on_1b/2b/3b NaN→0/1 문자열, 필요 컬럼 추출(features 6개 + meta 10개)
  self.pitcher_mlbam_id 저장 (이후 군집 조회용)
       ↓  df_processed (단일 투수 데이터)

[Step 2] PitchClustering.run_clustering_pipeline()
  6개 물리 피처 StandardScaler → UMAP(n_components=2) → KMeans(K=3~6 실루엣)
  군집별 평균 구속 내림차순으로 구종 이름 매핑 (Fastball→Slider→Changeup→...)
  df['mapped_pitch_name'] 컬럼 추가
       ↓  df_clustered, identified_pitch_names

[Step 3] TransitionProbabilityModel.run_modeling_pipeline()
  _prepare_data():
    batter_clusters_2023.csv left join → batter_cluster 컬럼
    pitcher_clusters_2023.csv left join → pitcher_cluster 컬럼 (p_cluster로 리네임 후 merge)
    count_state 생성: "{b}-{s}_{outs}_{runners}" 형식
    pd.get_dummies(count_state + mapped_pitch_name + zone + batter_cluster + pitcher_cluster)
    LabelEncoder(description) → y
  PyTorch MLP 학습 (Input→128→64→output_dim)
  best_transition_model.pth 저장
       ↓  feature_columns, target_classes

[Step 4] MDPOptimizer.run_optimizer()
  9,216개 상태 생성: 12카운트 × 3아웃 × 8주자 × 8타자군집 × 4투수군집
  Value Iteration 5회 반복 (파울 사이클 수렴 목적)
  각 상태에서 predict_proba() 호출 → 기대 보상 최대 행동 선택
  W&B Table로 주요 상황별 정책 로깅
       ↓  optimal_policy dict

[Step 5] DQNTrainer.train() + evaluate()
  PitchEnv (Gymnasium): 이닝 단위 시뮬레이션
    reset(): outs/runners 무작위, 타자 군집 무작위, 투수 군집 고정(단일 투수 모드)
    step(action): MLP predict_proba() 호출 → 결과 샘플링 → 상태 전이 → RE24 보상
  Stable-Baselines3 DQN (MlpPolicy, net_arch=[128,64])
  300,000 스텝 학습 + 10,000 스텝마다 EvalCallback
  smartpitch_dqn_final.zip 저장
```

---

## 핵심 데이터 구조

### MDP 상태 키 (문자열, 5-파트)
```
형식: "{count}_{outs}_{runners}_{batter_cluster}_{pitcher_cluster}"
예시: "3-2_2_111_7_0"
파싱: parts = state.split('_')  → 5개 원소
      parts[0]="3-2", parts[1]="2", parts[2]="111", parts[3]="7", parts[4]="0"
      # count_state_val은 model.py 입력에 직접 사용하지 않음 (B안 이후)
      # balls/strikes/outs/runners를 수치 피처로 직접 전달
```

> **주의**: 예전에 4-파트 형식(`count_outs_runners_batter`)이었다가 5-파트로 변경됨.
> `rsplit('_', 1)` 같은 방식으로 파싱하면 틀림. 반드시 `split('_')` + 인덱스로 파싱.

### 관측 벡터 (8D, PitchEnv observation_space)
```python
[balls(0-3), strikes(0-2), outs(0-2), on_1b(0/1), on_2b(0/1), on_3b(0/1),
 batter_cluster(0-7), pitcher_cluster(0-3)]
```

### 모델 입력 피처 구성 (model.py, B안 이후)
```python
# 수치 피처 (6개) + One-Hot 범주형 피처
X_num = pd.DataFrame({'balls':..., 'strikes':..., 'outs':..., 'on_1b':..., 'on_2b':..., 'on_3b':...})
X_cat_encoded = pd.get_dummies(df[['mapped_pitch_name','zone','batter_cluster','pitcher_cluster']])
X_encoded = pd.concat([X_num, X_cat_encoded], axis=1)
# 총 ~40차원: 수치(6) + pitch_name(9) + zone(13) + batter(8) + pitcher(4)
# feature_columns 리스트로 보관 → PitchEnv와 MDPOptimizer에서 동일 순서로 입력 구성
```

### RE24 매트릭스 (2019 MLB, 하드코딩)
```python
# pitch_env.py PitchEnv.RE24_MATRIX 와 mdp_solver.py MDPOptimizer.re24_matrix 에 동일하게 존재
# 키 형식: "{outs}_{runners}"  예) "0_000"=0.481, "2_111"=0.736
```

---

## 주요 설계 결정 사항

| 항목 | 결정 | 이유 |
|------|------|------|
| 타자 군집 K=8 고정 | 실루엣 탐색 없이 고정 | 탐색 결과 K=8이 안정적, MDP 상태 복잡도(×8) 대비 적절 |
| 투수 군집 K=4~8 탐색 | 실루엣 최대 K 선택 | 2023 시즌 결과 K=4 선택됨(0.4502) |
| 좌/우타 분리 군집화 | batter_clustering.py | 동일 수치가 좌우타에서 다른 의미를 가짐 |
| 투수 군집 좌/우투 미분리 | pitcher_clustering.py | 릴리스 포인트(pfx_x, release_pos_x)가 대리 지표로 작동 |
| Value Iteration 5회 | mdp_solver.py | 파울 시 카운트가 유지되는 순환 구조 수렴 필요 |
| 인플레이 타구 확률 하드코딩 | pitch_env.py, mdp_solver.py | 70% 아웃, 15% 1루타, 10% 2루타, 5% 홈런 |
| pitcher_cluster 전달 방식 | main.py → MDPOptimizer/PitchEnv | MDPOptimizer: `pitcher_clusters=["0","1","2","3"]` (문자열 리스트), PitchEnv: `pitcher_cluster=0` (int, 고정 모드) |
| MLP 입력 count_state 형식 | `"3-2_2_111"` (볼-스트라이크_아웃_주자) | mdp_solver와 pitch_env가 동일 형식 사용 |
| batter/pitcher cluster 컬럼 충돌 방지 | pitcher cluster의 `cluster` → `p_cluster`로 리네임 후 merge | 두 CSV 모두 `cluster` 컬럼 보유 |

---

## 실행 방법 및 자주 쓰는 명령어

```bash
# 환경 세팅
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu124  # GPU만

# 선행 작업 (최초 1회 또는 데이터 갱신 시)
uv run src/batter_clustering.py      # ~10분, 캐시 이후 ~1분 → data/batter_clusters_2023.csv
uv run src/pitcher_clustering.py     # ~3분, 캐시 이후 ~30초 → data/pitcher_clusters_2023.csv
uv run src/universal_model_trainer.py # ~20~40분 → best_transition_model_universal.pth
                                      # 또는 W&B Artifact "universal_transition_mlp"에서 다운로드

# 메인 파이프라인 실행
uv run src/main.py                 # 투수 이름, 날짜 입력 프롬프트 표시
# 예) Clayton Kershaw / 2024-03-20 / 2024-09-30

# W&B 로그인 (최초 1회)
uv run wandb login

# W&B 런 데이터 로컬 추출 (fetch_wandb_run.py는 RUN_ID가 하드코딩됨)
uv run fetch_wandb_run.py          # → wandb_export/ 폴더에 CSV/JSON 저장

# Git 작업
git branch --show-current
git log --oneline -5
```

> **주의**: `uv run main.py`는 틀린 명령. 진입점은 `src/` 안에 있으므로 `uv run src/main.py`

---

## 군집화 결과 (2023 시즌)

### 투수 군집 (K=4, 479명)
| 군집 | 특성 | 인원 | 구속 | FF% | SL% | Whiff% | 구종수 |
|------|------|------|------|-----|-----|--------|--------|
| 0 | 파워 패스트볼/슬라이더 | 157 | 89.9 | 44.7% | 25.6% | 25.9% | 3.4 |
| 1 | 핀세스/커맨드 | 102 | 87.9 | 28.4% | 9.6% | 22.9% | 3.7 |
| 2 | 무브먼트/싱커볼 | 103 | 89.9 | 15.1% | 18.6% | 22.0% | 3.9 |
| 3 | 멀티피치/아스날 | 117 | 89.0 | 29.0% | 8.5% | 24.0% | 4.4 |

### 타자 군집 (K=8, 좌/우타 각각 독립 군집화)
피처 9개: `whiff%`, `z_contact%`, `o_swing%`, `o_contact%`, `avg_launch_angle(BBE)`,
`avg_launch_speed(BBE)`, `barrel%`, `pull%`(hc_x 기반), `high_ff_whiff%`

---

## 현재 성능 수치

```
[범용 모델 — Exp1 BiggerModel [256,128,64], 4클래스, 2023 MLB 72만 건]
MLP val_accuracy : 58.1%
MLP val_loss     : 1.0077  (unweighted CrossEntropyLoss)

[DQN — Gerrit Cole 2019, W&B run: h4n3o0di]
DQN 평균 보상    : 0.436   (100이닝 평가)
DQN 주요 구종    : Fastball 51.3%, Slider 24.3%, Curveball 14.9%, Changeup 10.7%
```

---

## 완료된 작업 / 남은 작업

### 완료
- [x] 단일 투수 데이터 수집, 구종 식별, MLP 학습, MDP 정책, DQN 학습 전체 파이프라인
- [x] 타자 군집화 K=8 (pull_pct hc_x 실측 계산, cache.enable())
- [x] 투수 군집화 K=4 (`pitcher_clustering.py` 신규 작성, CSV 생성)
- [x] 7D → 8D 상태공간 (pitch_env, mdp_solver, model, rl_trainer 모두 대응)
- [x] 상태 키 5-파트 형식 통일
- [x] W&B Artifact: 데이터셋, MLP 모델, DQN 모델, 군집 CSV
- [x] 모든 소스 파일 한국어 모듈 독스트링 및 인라인 주석
- [x] 범용 전이 모델: `universal_model_trainer.py` 완성 (Exp1 [256,128,64], val_acc 58.1%)
- [x] MLP epochs 5→20, batch_size 64→256, EarlyStopping(patience=5)
- [x] 4클래스 전환: hit_by_pitch 제거, ball/strike/foul/hit_into_play
- [x] `model_config_universal.json` 도입: load_from_checkpoint 아키텍처 불일치 방지
- [x] class-weighted CrossEntropyLoss 실험 (Exp3, foul/hit_into_play F1 개선 확인)

### 다음 우선순위
1. **[High]** RE24 매트릭스 연도별 갱신 (현재 2019 하드코딩, pitch_env.py + mdp_solver.py 두 곳)
2. **[Medium]** 인플레이 타구 확률 실데이터 기반 교체 (현재 70/15/10/5% 하드코딩)
3. **[Medium]** DQN 강화: total_timesteps 300K→500K, exploration_fraction 0.30→0.40
4. **[Low]** FastAPI 실시간 추천 API

---

## 코딩 컨벤션

- **언어**: 주석·독스트링은 한국어, 변수명·함수명은 영어 스네이크케이스
- **클래스 구조**: 각 파일마다 하나의 핵심 클래스 + `run_*_pipeline()` 퍼블릭 진입 메서드
- **내부 메서드**: `_` 접두사 (예: `_prepare_data()`, `_extract_batter_features()`)
- **W&B 로깅**: 모든 클래스에서 `if wandb.run:` 가드 후 로깅 (run 없어도 동작)
- **Fallback 패턴**: CSV 없으면 cluster="0" 기본값 (pipeline 중단 없이 계속 실행)
- **경로 처리**: `os.path.join(os.path.dirname(__file__), "..", "data", "xxx.csv")` 패턴 통일

---

## 브랜치 전략

```
main                      배포/완성본 브랜치
feature/task5-6-overnight      현재 작업 브랜치 (원격 동기화됨)
```

현재 브랜치 `feature/task5-6-overnight`에서 작업 중. PR 전 main에 머지.

---

## 주의사항 (건드리면 안 되는 것 / 반드시 알아야 할 것)

### 절대 건드리면 안 되는 것
1. **`feature_columns` 순서**: `model.py`의 `pd.get_dummies` 결과 컬럼 순서.
   `pitch_env.py`와 `mdp_solver.py`가 동일 순서로 입력을 구성함.
   컬럼 순서가 바뀌면 MLP 예측이 완전히 틀어짐.

2. **상태 키 파싱**: `state.split('_')` → 반드시 5개 원소.
   `count`에 `-`가 포함되어 있어 `split('_', maxsplit=4)`도 가능하지만,
   `rsplit` 등 다른 방식 사용 금지. parts[0]~parts[4] 직접 인덱싱.

3. **pitcher cluster 컬럼 리네임**: `model.py`에서 pitcher CSV의 `cluster` 컬럼을
   반드시 `p_cluster`로 rename 후 merge. 안 하면 batter의 `cluster`와 충돌해
   `cluster_x`, `cluster_y`로 분리되어 batter_cluster 컬럼 생성 실패.

### 반드시 알아야 할 것
4. **`fetch_wandb_run.py`의 RUN_ID 하드코딩**: `cuafju1e` (Kershaw 2024 런).
   다른 런 분석 시 `RUN_ID` 변수 수동 변경 필요.

5. **`uv sync` 후 CUDA 재설치**: `uv sync`는 `uv.lock` 기준 CPU torch로 되돌림.
   GPU 환경에서 `uv sync` 실행 후 반드시 CUDA torch 재설치.

6. **data/*.csv는 gitignored**: 클론 직후 `data/` 폴더가 비어 있음.
   `main.py` 실행 전에 반드시 두 군집화 스크립트를 먼저 실행해야 함.
   (파일 없으면 cluster="0" fallback으로 실행되지만 정확도 저하)

7. **범용 모델 사용 시 `model_config_universal.json` 필수**: `load_from_checkpoint()`가
   이 파일에서 hidden_dims를 읽어 MLP를 생성함. 파일 없으면 [128,64] fallback이지만,
   현재 best 모델은 [256,128,64]이므로 size mismatch 발생.

8. **RE24 매트릭스 이중 정의**: `pitch_env.py`(PitchEnv.RE24_MATRIX)와
   `mdp_solver.py`(MDPOptimizer.re24_matrix) 두 곳에 동일한 값이 있음.
   수정 시 두 파일 모두 업데이트 필요.

9. **`clustering.py` vs `batter_clustering.py` vs `pitcher_clustering.py` 혼동 주의**:
   - `clustering.py`: 단일 투수 구종 식별 (main.py 파이프라인 내부에서 호출)
   - `batter_clustering.py`: 전체 MLB 타자 유형 분류 (독립 실행 스크립트)
   - `pitcher_clustering.py`: 전체 MLB 투수 유형 분류 (독립 실행 스크립트)

---

## W&B 프로젝트 정보

```
Entity  : pitcheezy
Project : SmartPitch-Portfolio
주요 런 : h4n3o0di  (Gerrit Cole 2019, 범용 모델 + DQN 파이프라인)
```

로깅 항목: MLP 학습 곡선(epoch/train_loss/val_loss/val_accuracy), DQN 에피소드 보상/탐색률,
정책 테이블(3가지 상황 × 12 볼카운트), 구종 분포 막대 차트, UMAP 산점도,
Artifact(전처리 데이터 / MLP 모델 / DQN 모델 / 군집 CSV)
