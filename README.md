# ⚾ SmartPitch: MLB 투구 전략 최적화 파이프라인

Statcast 데이터 기반으로 **투수별 구종 클러스터링** → **전이 확률 MLP** → **MDP 최적 볼배합** → **DQN 강화학습**까지 한 번에 돌리는 통합 파이프라인입니다.  
실험 로그와 아티팩트는 **W&B(Weights & Biases)**로 기록합니다.

---

## 파이프라인 구성

### 1) 데이터 파이프라인 (01~04)

| 단계 | 모듈 | 설명 |
|------|------|------|
| 00 | `src/fetch.py` | pybaseball Statcast 기간별 수집 |
| 01 | `src/preprocess.py` | 컬럼 정리, 파생 변수(description_group 등) 생성 |
| 02 | `src/embedding.py` | 투수별 UMAP + HDBSCAN 구종 클러스터링 |
| 03 | `src/profiles.py` | 투수·타자 프로필 생성 |
| 04 | `src/matchup.py` | 투수-타자 매치업 테이블 생성 |

### 2) 모델 및 최적화

| 단계 | 모듈 | 설명 |
|------|------|------|
| 전이 모델 | `src/model.py` | PyTorch MLP — 볼카운트/구종/존 → 투구 결과 확률 예측 |
| MDP | `src/mdp_solver.py` | RE24 기반 벨만 방정식 역순 가치 반복 → 최적 구종·존 |
| DQN | `src/pitch_env.py`, `src/rl_trainer.py` | Gymnasium 환경 + DQN 에이전트 학습 |

---

## 실행 방법

**입력:** 투수 **MLBAM ID** + 기간(시작일, 종료일).  
데이터는 Statcast의 `pitcher`/`batter` ID 기준으로만 사용하며, W&B에는 ID로 조회한 **선수 이름**만 표시됩니다.

### 환경 준비 (uv 사용)

```bash
cd SmartPitch
uv sync
```

### 파이프라인 실행

```bash
uv run python main.py
```

실행 후 프롬프트에서 다음을 입력합니다.

- **분석할 투수의 MLBAM ID** (예: `543037` = Gerrit Cole)
- **데이터 시작일** (예: `2019-03-28`)
- **데이터 종료일** (예: `2019-09-29`)

### 추천 투수 ID (MLBAM)

| 선수 | ID |
|------|-----|
| Gerrit Cole | 543037 |
| Yu Darvish | 506433 |
| Clayton Kershaw | 477132 |

---

## 디렉터리 구조

```
SmartPitch/
├── main.py              # 진입점 (데이터 파이프라인 + 모델 + MDP + DQN)
├── src/
│   ├── fetch.py         # Statcast 수집
│   ├── preprocess.py    # 전처리
│   ├── embedding.py     # UMAP + HDBSCAN 클러스터링
│   ├── profiles.py      # 투수/타자 프로필
│   ├── matchup.py       # 매치업 테이블
│   ├── io_utils.py      # 경로·Parquet I/O
│   ├── model.py         # 전이 확률 MLP
│   ├── mdp_solver.py    # MDP 최적 정책
│   ├── pitch_env.py     # DQN용 Gym 환경
│   └── rl_trainer.py    # DQN 학습
├── data/
│   ├── raw/             # 수집 CSV (미추적)
│   └── processed/       # parquet 결과물 (미추적)
└── pyproject.toml
```

---

## W&B

- **프로젝트:** `SmartPitch-Portfolio`
- **Run 이름:** `{선수이름}_Pipeline` (ID로 조회한 이름)
- **Config:** `pitcher_id`, `pitcher_name`, 학습 하이퍼파라미터
- **아티팩트:** 전처리 데이터, MLP 가중치 등

실행 전 [wandb 로그인](https://docs.wandb.ai/quickstart)이 되어 있어야 합니다.
