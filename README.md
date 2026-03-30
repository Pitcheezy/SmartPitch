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
          │  벨만 방정식 가치반복(5회) → 9,216개 상태 최적 구종/코스 정책표
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
│   └── rl_trainer.py                DQN 에이전트 학습/평가
├── data/                            (gitignored — 클론 후 생성 필요)
│   ├── batter_clusters_2023.csv         타자 군집 매핑 (batter_id → cluster 0~7)
│   ├── pitcher_clusters_2023.csv        투수 군집 매핑 (pitcher_id → cluster 0~3)
│   ├── feature_columns_universal.json   범용 모델 입력 피처 목록
│   ├── target_classes_universal.json    범용 모델 출력 클래스 목록 (4종)
│   └── model_config_universal.json      범용 모델 아키텍처 설정 (hidden_dims, dropout_rate)
├── docs/
│   └── work_log_20260329_30.md      작업 로그 (학습용)
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
| MLP val_accuracy | 58.1% | 범용 모델 Exp1 [256,128,64], 4클래스, 2023 MLB 72만 건 |
| DQN 평균 보상 | 0.436 | Gerrit Cole 2019, 100이닝 평가 |
| DQN 주요 구종 | Fastball 51.3%, Slider 24.3% | Cole 2019 시즌 실제 구종 비율과 유사 |

---

## 개선 필요 사항 (기술 부채)

```
우선순위  항목
─────────────────────────────────────────────────────────────
[High]     RE24 매트릭스 연도별 갱신 (현재 2019 고정)
           → pitch_env.py와 mdp_solver.py 두 곳 동시 수정 필요
[Medium]   batted ball 확률 단순화 (70/15/10/5%) → 실제 데이터 기반
[Medium]   DQN timesteps 300K → 500K+, exploration 0.3 → 0.4
[Low]      실시간 추천 API 서버 (FastAPI) 구현
[Low]      W&B Sweep으로 하이퍼파라미터 자동 탐색
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

1. **RE24 갱신**: pitch_env.py + mdp_solver.py의 2019 고정값을 분석 시즌 기준으로 교체
2. **인플레이 타구 확률**: 현재 하드코딩(70/15/10/5%) → 실제 MLB 데이터 기반 교체
3. **DQN 강화**: total_timesteps 300K → 500K, exploration_fraction 0.30 → 0.40
4. **실시간 추천 API**: 타석 상황 입력 → 구종·코스 추천 JSON 반환
