# DQN 모델 사용 가이드 (백엔드 통합용)

SmartPitch DQN 모델을 백엔드 서비스에 통합하기 위한 가이드입니다.

---

## 1. 사용 가능한 모델

| 투수 | 파일명 | Actions | 구종 (구속 내림차순) |
|------|--------|---------|---------------------|
| Gerrit Cole | `smartpitch_dqn_final.zip` | 52 | Fastball, Slider, Curveball, Changeup |
| Dylan Cease | `dqn_cease_2024_2025.zip` | 39 | Fastball, Slider, Changeup |
| Zac Gallen | `dqn_gallen_2024_2025.zip` | 52 | Fastball, Slider, Changeup, Curveball |

> 구종 순서가 투수마다 다릅니다. Action Index 변환 시 반드시 해당 투수의 `pitch_names`를 사용하세요.

---

## 2. 모델 로드

```python
from stable_baselines3 import DQN

# 투수별 모델 로드
model_cole = DQN.load("smartpitch_dqn_final.zip")
model_cease = DQN.load("dqn_cease_2024_2025.zip")
model_gallen = DQN.load("dqn_gallen_2024_2025.zip")
```

의존 패키지: `stable-baselines3>=2.3`, `gymnasium>=0.29`, `torch>=2.10`

---

## 3. Observation (입력: 8차원 벡터)

```python
import numpy as np

obs = np.array([
    1,    # balls:    0~3
    2,    # strikes:  0~2
    1,    # outs:     0~2
    1,    # on_1b:    0 또는 1
    0,    # on_2b:    0 또는 1
    0,    # on_3b:    0 또는 1
    3,    # batter_cluster: 0~7
    0,    # pitcher_cluster: 0 (3명 모두 군집 0)
], dtype=np.float32)
```

| 인덱스 | 피처 | 범위 | 설명 |
|--------|------|------|------|
| 0 | `balls` | 0~3 | 볼 카운트 |
| 1 | `strikes` | 0~2 | 스트라이크 카운트 |
| 2 | `outs` | 0~2 | 현재 아웃 수 |
| 3 | `on_1b` | 0/1 | 1루 주자 유무 |
| 4 | `on_2b` | 0/1 | 2루 주자 유무 |
| 5 | `on_3b` | 0/1 | 3루 주자 유무 |
| 6 | `batter_cluster` | 0~7 | 상대 타자 유형 |
| 7 | `pitcher_cluster` | 0~3 | 투수 유형 (3명 모두 0) |

---

## 4. 추론 호출

```python
action, _ = model.predict(obs, deterministic=True)
action = int(action)
```

응답 시간: ~1ms 미만 (모델 사전 로드 시)

---

## 5. Action Index 변환

```python
# 투수별 구종 리스트 (구속 내림차순, 학습 시 사용된 순서)
PITCHER_CONFIG = {
    "cole": {
        "pitch_names": ["Fastball", "Slider", "Curveball", "Changeup"],
        "model_path": "smartpitch_dqn_final.zip",
    },
    "cease": {
        "pitch_names": ["Fastball", "Slider", "Changeup"],
        "model_path": "dqn_cease_2024_2025.zip",
    },
    "gallen": {
        "pitch_names": ["Fastball", "Slider", "Changeup", "Curveball"],
        "model_path": "dqn_gallen_2024_2025.zip",
    },
}

# 존 목록 (13개, 고정)
ZONES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
N_ZONES = 13

# Action → 구종 + 존 변환
def decode_action(action: int, pitcher_id: str) -> dict:
    config = PITCHER_CONFIG[pitcher_id]
    pitch_names = config["pitch_names"]

    pitch_index = action // N_ZONES
    zone_index = action % N_ZONES

    pitch_name = pitch_names[pitch_index]
    zone = ZONES[zone_index]

    # 존 설명
    ZONE_DESC = {
        1: "좌상", 2: "중상", 3: "우상",
        4: "좌중", 5: "정중앙", 6: "우중",
        7: "좌하", 8: "중하", 9: "우하",
        11: "Up & In", 12: "Up & Away",
        13: "Down & In", 14: "Down & Away",
    }

    return {
        "action_index": action,
        "pitch_name": pitch_name,
        "zone": zone,
        "zone_description": ZONE_DESC.get(zone, ""),
    }
```

### Statcast Zone 맵

```
Zone 1~9: 스트라이크존 3x3 격자 (포수 시점)
    [1] [2] [3]     ← 상단
    [4] [5] [6]     ← 중단
    [7] [8] [9]     ← 하단

Zone 11~14: 볼존 (스트라이크존 바깥)
    [11: Up&In]  [12: Up&Away]
    [13: Down&In] [14: Down&Away]
```

> Zone 1~9의 좌/우는 **포수 시점**입니다. 좌타자/우타자에 따라 In/Away 의미가 반대.

---

## 6. 타자 ID → 군집 변환

```python
import pandas as pd

# 앱 시작 시 1회 로드
batter_df = pd.read_csv("data/batter_clusters_2023.csv")

def get_batter_cluster(batter_id: int) -> int:
    """MLB batter ID → 군집 번호 (0~7). 못 찾으면 랜덤."""
    import random
    matches = batter_df.loc[batter_df['batter'] == batter_id, 'batter_cluster']
    if len(matches) > 0:
        return int(matches.iloc[0])
    return random.randint(0, 7)
```

`data/batter_clusters_2023.csv` 컬럼: `batter` (MLB ID), `stand` (L/R), `batter_cluster` (0~7)

---

## 7. 전체 추론 예시

```python
from stable_baselines3 import DQN
import numpy as np
import pandas as pd

# 초기화 (앱 시작 시 1회)
models = {
    "cole": DQN.load("smartpitch_dqn_final.zip"),
    "cease": DQN.load("dqn_cease_2024_2025.zip"),
    "gallen": DQN.load("dqn_gallen_2024_2025.zip"),
}
batter_df = pd.read_csv("data/batter_clusters_2023.csv")

# 추론 함수
def recommend(pitcher_id: str, balls: int, strikes: int, outs: int,
              on_1b: int, on_2b: int, on_3b: int,
              batter_id: int = None) -> dict:
    # 1. 타자 군집 결정
    if batter_id:
        matches = batter_df.loc[batter_df['batter'] == batter_id, 'batter_cluster']
        batter_cluster = int(matches.iloc[0]) if len(matches) > 0 else 4
    else:
        batter_cluster = 4  # 기본값 (중간)

    # 2. Observation 구성
    obs = np.array([balls, strikes, outs, on_1b, on_2b, on_3b,
                    batter_cluster, 0], dtype=np.float32)

    # 3. 추론
    action, _ = models[pitcher_id].predict(obs, deterministic=True)

    # 4. 변환
    return decode_action(int(action), pitcher_id)


# 사용 예
result = recommend("cease", balls=1, strikes=2, outs=1,
                   on_1b=1, on_2b=0, on_3b=0, batter_id=660271)
# → {"action_index": 2, "pitch_name": "Fastball", "zone": 3, ...}
```

---

## 8. 필요 파일 목록

### 모델 파일 (gitignored, 별도 전달)

| 파일 | 크기 | 설명 |
|------|------|------|
| `smartpitch_dqn_final.zip` | ~200KB | Cole DQN |
| `dqn_cease_2024_2025.zip` | ~217KB | Cease DQN |
| `dqn_gallen_2024_2025.zip` | ~231KB | Gallen DQN |
| `best_transition_model_universal.pth` | ~300KB | 범용 MLP (PitchEnv 내부 사용) |

### 데이터 파일 (git 추적 확인 필요)

| 파일 | 용도 |
|------|------|
| `data/batter_clusters_2023.csv` | 타자 ID → 군집 매핑 |
| `data/feature_columns_universal.json` | MLP 입력 피처 정의 |
| `data/target_classes_universal.json` | MLP 출력 클래스 정의 |
| `data/model_config_universal.json` | MLP 아키텍처 설정 |
| `data/physical_feature_lookup.csv` | 구종별 물리 피처 평균값 |

> `best_transition_model_universal.pth`는 DQN 추론 자체에는 불필요합니다.
> DQN `.predict()`만 호출하면 됩니다.
> `.pth` 파일은 PitchEnv(시뮬레이터)나 MDP 풀이 시에만 필요합니다.

---

## 9. 자주 발생하는 에러

| 에러 | 원인 | 해결 |
|------|------|------|
| `shape mismatch` | Observation이 8차원 아님 또는 dtype이 float32 아님 | `np.array([...], dtype=np.float32)` 확인 |
| `action out of range` | pitch_index가 구종 수 초과 | 해당 투수의 `pitch_names` 길이 확인 |
| `batter not found` | batter_id가 CSV에 없음 | 기본 군집(4) 또는 랜덤 할당 |
| `FileNotFoundError: *.zip` | 모델 파일 누락 | 모델 파일 별도 전달 필요 (gitignored) |

---

## 10. 모델 성능 (참고)

| 투수 | DQN Mean Reward | Action Space | 주력 구종 |
|------|----------------|-------------|----------|
| Cole | +0.436 ± 1.255 | 52 | Fastball 51% |
| Cease | +0.198 ± 1.177 | 39 | Fastball 83% |
| Gallen | +0.239 ± 1.134 | 52 | Fastball 36%, Curveball 34% |

- 상세 평가: [docs/personal_dqn_report.md](personal_dqn_report.md)
- 입출력 JSON 예시: [docs/demo_api_spec.md](demo_api_spec.md)
- 통계적 유의성: 1,000 에피소드 평가에서 DQN과 베이스라인 차이가 통계적으로 유의미하지 않음 (p > 0.29). 야구의 본질적 변동성(σ ≈ 1.15) 때문. 절대값보다 **정책 패턴(구종 분포, 상황별 추천)**이 더 의미 있습니다.

---

## 11. 모델 파일 배포

모델 `.zip` 파일은 `.gitignore`에 포함되어 GitHub에 올라가지 않습니다.

배포 방법:
1. **직접 전달**: Slack/Google Drive 등으로 4개 파일 전달
2. **W&B Artifact**: `wandb.Api().artifact('pitcheezy/SmartPitch-Portfolio/smartpitch_dqn_model:latest').download()`
3. **학습 재실행**: `uv run scripts/main_cease.py` / `uv run scripts/main_gallen.py` (각 ~30분)
