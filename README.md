# SmartPitch: MLB 투타 매치업 기반 투구 전략 최적화 파이프라인

MLB Statcast 데이터를 기반으로, **특정 투수 유형이 특정 타자 유형을 상대할 때** 기대 실점(RE24)을 최소화하는 최적 구종·코스를 추천하는 엔드-투-엔드 AI 파이프라인입니다. 딥러닝(PyTorch MLP) + 마르코프 결정 과정(MDP) + 강화학습(DQN) 3단계 접근법을 사용합니다.

---

## 전체 파이프라인 흐름

```
Statcast 데이터 수집 (pybaseball)
          │
          ▼
[1단계] 구종 식별         clustering.py
          │  UMAP + K-Means로 투수별 구종 레퍼토리 자동 탐지
          ▼
[2단계] 타자 군집화       batter_clustering.py   → data/batter_clusters_2023.csv
          │  9개 타격 지표로 MLB 타자를 8가지 유형으로 분류
          ▼
[3단계] 투수 군집화       pitcher_clustering.py  → data/pitcher_clusters_2023.csv
          │  15개 투구 지표로 MLB 투수를 4가지 유형으로 분류 (K=4, 실루엣 0.45)
          ▼
[4단계] 전이 확률 모델    model.py / universal_model_trainer.py
          │  (볼카운트 + 구종 + 코스 + 타자유형 + 투수유형) → 투구결과 4클래스 확률
          │  범용 모델: PyTorch MLP [256→128→64], 2023 MLB 전체 72만 건 학습
          │  출력 클래스: ball / strike / foul / hit_into_play
          ▼
[5단계] MDP 최적 정책    mdp_solver.py
          │  벨만 방정식 가치반복(최대 20회, γ=0.99) → 9,216개 상태 최적 구종/코스 정책표
          │  상태: 카운트(12) × 아웃(3) × 주자(8) × 타자군집(8) × 투수군집(4)
          ▼
[6단계] DQN 강화학습      pitch_env.py + rl_trainer.py → smartpitch_dqn_final.zip
          │  8D 관측 [볼,스트라이크,아웃,1루,2루,3루,타자군집,투수군집]
          │  Stable-Baselines3 DQN, 300K 스텝
          ▼
W&B 대시보드 (학습곡선 / UMAP 시각화 / 정책테이블 / 구종분포)
```

---

## 파일 구조

```
SmartPitch/
├── src/
│   ├── main.py                      전체 파이프라인 실행 진입점
│   ├── data_loader.py               Statcast 데이터 수집 + 전처리
│   ├── clustering.py                투수별 구종 식별 (UMAP + K-Means)
│   ├── batter_clustering.py         타자 유형 군집화 (K=8, 독립 실행용)
│   ├── pitcher_clustering.py        투수 유형 군집화 (K=4, 독립 실행용)
│   ├── universal_model_trainer.py   범용 전이 모델 학습 (2023 MLB 전체, 독립 실행용)
│   ├── model.py                     전이 확률 예측 MLP 학습
│   ├── mdp_solver.py                MDP 가치반복 최적 정책 계산
│   ├── pitch_env.py                 Gymnasium RL 환경
│   ├── rl_trainer.py                DQN 에이전트 학습/평가
│   └── evaluate_baselines.py        베이스라인 5종 비교 평가 (Random/MostFreq/Freq/MDP/DQN ref)
├── scripts/
│   ├── analyze_mdp_vs_env.py            MDP vs PitchEnv 보상 일관성 분석 리포트 생성
│   ├── generate_baseline_presentation.py 발표용 PNG 3종 생성 (overall/by_cluster/summary)
│   ├── generate_presentation_charts.py  W&B 실험 결과 발표용 차트
│   ├── generate_pitch_location_heatmaps.py  투구 위치 히트맵
│   ├── single_pitcher_zone_breakdown.py     단일 투수 zone 분포 분석
│   ├── generate_physical_lookup.py          물리 피처 lookup CSV 생성
│   └── train_dqn_all_clusters.py            군집별 DQN 학습 + 평가 (1~3)
├── data/                            (gitignored — 클론 후 생성 필요)
│   ├── batter_clusters_2023.csv         타자 군집 매핑 (batter_id → cluster 0~7)
│   ├── pitcher_clusters_2023.csv        투수 군집 매핑 (pitcher_id → cluster 0~3)
│   ├── feature_columns_universal.json   범용 모델 입력 피처 목록
│   ├── target_classes_universal.json    범용 모델 출력 클래스 목록 (4종)
│   ├── model_config_universal.json      범용 모델 아키텍처 설정 (hidden_dims, dropout_rate)
│   ├── physical_feature_lookup.csv      (pitcher_cluster, mapped_pitch_name)별 평균 물리 피처
│   └── mdp_optimal_policy.pkl           MDP VI 결과 캐시 (9,216 상태)
├── docs/
│   ├── baseline_comparison.md           Cole 2019 5-agent 비교 결과
│   ├── baseline_by_cluster.md           투수 군집별(K=4) 5-agent 비교 결과
│   ├── mdp_vs_env_reward_analysis.md    MDP vs PitchEnv 줄 단위 분석 + 수렴/정책/trace
│   ├── experiment_comparison.md         범용 MLP 실험(Exp1~5) 비교
│   └── work_log_20260329_30.md          작업 로그 (학습용)
├── pyproject.toml               uv 의존성 정의
├── uv.lock                      정확한 버전 잠금 (팀 동기화 기준)
├── .python-version              Python 3.12 고정
├── AI_CONTEXT.md                AI 작업 컨텍스트 (다음 작업 가이드)
├── TODO.md                      작업 리스트 (완료/남은 작업)
└── README.md                    이 파일
```

---

## 팀원 빠른 시작 (Quick Start)

### 환경 세팅

```bash
# 1. 레포 클론
git clone https://github.com/Pitcheezy/SmartPitch
cd SmartPitch

# 2. 의존성 설치 (uv.lock 기반 — 팀 전원 완전히 동일한 버전)
uv sync

# 3. (로컬에 GPU가 있는 경우만) CUDA 버전 torch로 교체
#    이 명령은 uv.lock을 수정하지 않으므로 팀에 영향 없음
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. CUDA 설치 확인
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> **주의**: `uv sync`를 다시 실행하면 CPU 버전 torch로 되돌아갑니다.
> GPU 환경에서는 `uv sync` 후 반드시 위의 CUDA 설치 명령을 다시 실행하세요.

### 실행 순서

```bash
# [선행 작업 1] 군집화 CSV 생성 — 최초 1회만 실행, 이후 캐시 활용
uv run src/batter_clustering.py   # → data/batter_clusters_2023.csv (타자 K=8)
uv run src/pitcher_clustering.py  # → data/pitcher_clusters_2023.csv (투수 K=4)

# [선행 작업 2] 범용 전이 모델 — 아래 두 방법 중 하나 선택

# 방법 A: 직접 학습 (약 20~40분)
uv run src/universal_model_trainer.py

# 방법 B: W&B Artifact에서 다운로드 (1분)
uv run python -c "
import wandb; api = wandb.Api()
artifact = api.artifact('pitcheezy/SmartPitch-Portfolio/universal_transition_mlp:latest')
artifact.download(root='.')
"
# 두 방법 모두 아래 4개 파일을 생성:
# → best_transition_model_universal.pth
# → data/feature_columns_universal.json
# → data/target_classes_universal.json
# → data/model_config_universal.json

# [메인 파이프라인] 특정 투수 분석 + DQN 학습
uv run src/main.py
# 실행 후 투수 이름과 시즌을 입력
# 예) Clayton Kershaw / 2024-03-20 ~ 2024-09-30
# main.py 상단의 USE_UNIVERSAL_MODEL=True 시 범용 모델 사용 (권장)
```

### W&B 로그인 (최초 1회)

```bash
uv run wandb login
# → https://wandb.ai/authorize 에서 API 키 복사 후 붙여넣기
# 프로젝트: pitcheezy/SmartPitch-Portfolio
```

---

## 투수 유형 군집 (K=4, 2023시즌 479명)

| 군집 | 유형명 | 인원 | 평균구속 | FF% | SL% | Whiff% | 구종수 |
|------|--------|------|---------|-----|-----|--------|--------|
| **0** | 파워 패스트볼/슬라이더 | 157명 | 89.9 | 44.7% | 25.6% | 25.9% | 3.4종 |
| **1** | 핀세스/커맨드 | 102명 | 87.9 | 28.4% | 9.6% | 22.9% | 3.7종 |
| **2** | 무브먼트/싱커볼 | 103명 | 89.9 | 15.1% | 18.6% | 22.0% | 3.9종 |
| **3** | 멀티피치/아스날 | 117명 | 89.0 | 29.0% | 8.5% | 24.0% | 4.4종 |

## 타자 유형 군집 (K=8, 2023시즌 좌/우타 분리)

9개 피처: `whiff%`, `z_contact%`, `o_swing%`, `o_contact%`, `avg_launch_angle(BBE)`, `avg_launch_speed(BBE)`, `barrel%`, `pull%`, `high_ff_whiff%`

---

## 상태 공간 설명 (MDP / RL 공통)

```
관측 벡터 (8D):
  [balls(0-3), strikes(0-2), outs(0-2), on_1b(0/1), on_2b(0/1), on_3b(0/1),
   batter_cluster(0-7), pitcher_cluster(0-3)]

MDP 상태 키 형식 (문자열):
  "{count}_{outs}_{runners}_{batter_cluster}_{pitcher_cluster}"
  예시: "3-2_2_111_7_0"  = 3-2카운트, 2아웃, 만루, 타자군집7, 투수군집0

총 상태 수: 12(카운트) × 3(아웃) × 8(주자) × 8(타자) × 4(투수) = 9,216개
```

---

## 현재 성능 지표

| 지표 | 값 | 비고 |
|------|-----|------|
| MLP val_accuracy | 58.3% | **Exp4** 물리 피처(+구속/무브) 추가, [256,128,64], 4클래스 |
| MLP Top-2 / Top-3 | 80.8% / 95.1% | Exp4, 확률 분포 품질 지표 |
| MLP macro F1 | 0.495 | **Exp5** (class_weights + physical), 소수 클래스 recall 개선 |
| 현재 canonical 모델 | **Exp5** | MDP 확률 분포 품질 우선 → 저장된 universal 모델 |
| DQN 평균 보상 | 0.436 | Gerrit Cole 2019, 100이닝 평가 |
| DQN 주요 구종 | Fastball 51.3%, Slider 24.3% | Cole 2019 시즌 실제 구종 비율과 유사 |

### 베이스라인 비교 (4군집 × 5에이전트, 각 1000 에피소드, Task 18 action space 최적화 후)

| 군집 | Random | MostFreq | Frequency | MDP | DQN | 최고 에이전트 |
|------|--------|----------|-----------|-----|-----|-------------|
| **0** (파워 FB/SL) | +0.185 | +0.220 | +0.175 | +0.258 | **+0.436** | DQN |
| **1** (핀세스) | +0.136 | +0.140 | +0.169 | **+0.247** | +0.188 | MDP |
| **2** (싱커볼러) | +0.229 | +0.201 | +0.203 | **+0.262** | +0.242 | MDP |
| **3** (멀티피치) | +0.171 | +0.251 | +0.202 | **+0.256** | +0.215 | MDP |

- 군집 0: DQN이 최고 (Cole 2019 전용 ~52 actions)
- 군집 1~3: MDP가 최고 (VI 18회 수렴, γ=0.99, 군집별 유효 구종 필터링)
- Task 18에서 `get_valid_pitches()`로 1% 미만 구종 제거 → Knuckleball 편중 완전 해소

자세한 비교는 [`docs/baseline_comparison.md`](docs/baseline_comparison.md),
군집별 결과는 [`docs/baseline_by_cluster.md`](docs/baseline_by_cluster.md),
MLP 실험은 [`docs/experiment_comparison.md`](docs/experiment_comparison.md) 참조.

---

## 모델 사용 (백엔드 통합)

3명의 투수(Cole, Cease, Gallen)에 대한 개인 맞춤 DQN 모델이 학습 완료되었습니다.

| 문서 | 내용 |
|------|------|
| [`docs/MODEL_USAGE.md`](docs/MODEL_USAGE.md) | 백엔드 통합 가이드 (모델 로드, 추론, Action 변환) |
| [`docs/demo_api_spec.md`](docs/demo_api_spec.md) | API 입출력 스펙, Mock 응답, 투수별 데이터 |
| [`docs/personal_dqn_report.md`](docs/personal_dqn_report.md) | 3명 투수 종합 비교 + 통계 분석 |

모델 파일(`.zip`)은 gitignored이므로 별도 전달이 필요합니다.

---

## 개선 필요 사항 (기술 부채)

```
우선순위  항목
─────────────────────────────────────────────────────────────
[Done]     물리 피처 Phase 2: PitchEnv/MDPOptimizer lookup 테이블 추가 (MDP +63% 개선)
[Done]     MDP solve_mdp 수렴 개선: VI 최대 20회 + γ=0.99 + δ<1e-4 조기종료 (17회 수렴)
[Done]     Action Space 최적화: get_valid_pitches()로 1% 미만 구종 제거, Knuckleball 편중 해소
[Done]     군집 1~3 DQN 학습 + 전 군집 비교표 완성
[High]     RE24 매트릭스 연도별 갱신 (현재 2019 고정)
           → pitch_env.py와 mdp_solver.py 두 곳 동시 수정 필요
[Medium]   인플레이 타구 확률 실데이터 기반 교체 (현재 70/15/10/5% 하드코딩)
[Medium]   DQN 학습 강화 (300K→500K, exploration 0.30→0.40)
[Low]      FastAPI 실시간 추천 API
```

---

## W&B 대시보드 활용

```
프로젝트: https://wandb.ai/pitcheezy/SmartPitch-Portfolio

로깅 항목:
  - MLP: epoch별 train_loss / val_loss / val_accuracy
  - DQN: 에피소드별 보상 / 탐색률 / 에피소드 길이
  - 정책 테이블: 볼카운트별 최적 구종·코스·기대값 (3가지 상황)
  - 구종 분포: 평가 100이닝 기준 막대 차트
  - UMAP 산점도: 구종 군집 / 타자 군집 / 투수 군집
  - Artifact: 전처리 데이터 / MLP 모델 / DQN 모델 버전 관리
```

---

## 다음 마일스톤

1. ~~**물리 피처 Phase 2**~~ ✅ lookup 테이블 구축 완료 (MDP +0.151→+0.247, +63%)
2. ~~**MDP 수렴 개선**~~ ✅ VI 최대 20회 + γ=0.99 + δ<1e-4 (17회 수렴, MDP +0.250)
3. ~~**군집 1~3 DQN 학습**~~ ✅ 전 군집 DQN 학습 + 비교표 완성
4. ~~**Action Space 최적화**~~ ✅ get_valid_pitches()로 1% 미만 구종 제거, Knuckleball 편중 해소
5. **RE24 갱신**: pitch_env.py + mdp_solver.py의 2019 고정값을 분석 시즌 기준으로 교체
6. **인플레이 타구 확률**: 현재 하드코딩(70/15/10/5%) → 실제 MLB 데이터 기반 교체
7. **실시간 추천 API**: 타석 상황 입력 → 구종·코스 추천 JSON 반환
