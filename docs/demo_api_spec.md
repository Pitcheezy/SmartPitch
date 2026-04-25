# Demo API Spec — 투수별 DQN 모델 + UI 연동 사양서

UI 팀원에게 전달할 입출력 스펙, Mock 응답, 학습 데이터 요구사항을 정리한 문서입니다.
대상 투수: **Gerrit Cole** (참고용), **Dylan Cease**, **Zac Gallen**

---

## 1. Cole 모델의 입출력 스펙 (참고용)

### 1.1 Observation Space (입력: 8차원 벡터)

DQN 에이전트가 매 스텝 관측하는 8D float 벡터입니다.

| 인덱스 | 피처 | 범위 | 설명 |
|--------|------|------|------|
| 0 | `balls` | 0~3 | 볼 카운트 |
| 1 | `strikes` | 0~2 | 스트라이크 카운트 |
| 2 | `outs` | 0~2 | 현재 아웃 수 |
| 3 | `on_1b` | 0 / 1 | 1루 주자 유무 |
| 4 | `on_2b` | 0 / 1 | 2루 주자 유무 |
| 5 | `on_3b` | 0 / 1 | 3루 주자 유무 |
| 6 | `batter_cluster` | 0~7 | 상대 타자 유형 (K=8 군집) |
| 7 | `pitcher_cluster` | 0~3 | 투수 유형 (K=4 군집) |

> **참고**: `batter_cluster`는 에피소드(타석) 시작 시 랜덤 샘플링되며,
> 실제 서비스에서는 상대 타자 ID → `batter_clusters_2023.csv` 조회로 결정합니다.

### 1.2 Action Space (출력: 정수 → 구종+존 조합)

```
action_index = pitch_index × n_zones + zone_index
```

**Cole 2019**: 4구종 × 13존 = **52 actions**

| Action | pitch_index | zone_index | 구종 | 존 |
|--------|-------------|------------|------|-----|
| 0 | 0 | 0 | Changeup | 1 |
| 1 | 0 | 1 | Changeup | 2 |
| ... | ... | ... | ... | ... |
| 12 | 0 | 12 | Changeup | 14 |
| 13 | 1 | 0 | Curveball | 1 |
| 26 | 2 | 0 | Fastball | 1 |
| 39 | 3 | 0 | Slider | 1 |
| 51 | 3 | 12 | Slider | 14 |

> **구종 순서**: `pitch_names`는 알파벳 정렬 (clustering.py 결과 후 sorted)
> **존 번호**: Statcast 존 (1~9: 스트라이크존 3×3, 11~14: 볼존)

#### Statcast Zone 맵

![Statcast Zone Map](./image.png)
*포수 시점. 우타자 기준 "In"은 좌측, "Away"는 우측.*

- Zone 1~9: 스트라이크존 내부 3×3 (1=좌상, 2=중상, 3=우상, 4=좌중, 5=정중앙, 6=우중, 7=좌하, 8=중하, 9=우하)
- Zone 11: Up & In (상좌 바깥 — 높고 몸쪽)
- Zone 12: Up & Away (상우 바깥 — 높고 바깥쪽)
- Zone 13: Down & In (하좌 바깥 — 낮고 몸쪽)
- Zone 14: Down & Away (하우 바깥 — 낮고 바깥쪽)

> **주의 (UI 팀)**: 좌타자/우타자에 따라 "In(몸쪽)/Away(바깥쪽)" 의미가 반대가 됩니다.
> UI에서 존을 시각화할 때 포수 시점/타자 시점 중 어느 기준인지 명시해야 합니다.
>
> Reference: https://baseballsavant.mlb.com/csv-docs#zone

### 1.3 예시 입력/출력 JSON

```json
// 입력: 상황 정보
{
  "balls": 1,
  "strikes": 2,
  "outs": 1,
  "on_1b": 1,
  "on_2b": 0,
  "on_3b": 0,
  "batter_cluster": 3,
  "pitcher_cluster": 0
}

// 출력: DQN 추천
{
  "action_index": 29,
  "pitch_name": "Fastball",
  "zone": 3,
  "zone_description": "스트라이크존 우측 상단",
  "confidence_note": "DQN deterministic policy"
}
```

---

## 2. 투수 정보 조회 결과

### 2.1 기본 정보

| 투수 | MLBAM ID | 투수 군집 | 군집 유형 |
|------|----------|----------|----------|
| Gerrit Cole | 543037 | **0** | 파워 FB/SL |
| Dylan Cease | 656302 | **0** | 파워 FB/SL |
| Zac Gallen | 668678 | **0** | 파워 FB/SL |

> 세 투수 모두 **군집 0** (파워 패스트볼/슬라이더 유형)에 속합니다.
> `pitcher_cluster` 입력값은 모두 `0`입니다.

### 2.2 시즌별 투구 수

| 투수 | 2019 | 2024 | 2025 | 합계 (학습 대상) |
|------|------|------|------|-----------------|
| Gerrit Cole | **3,362** | — | — | 3,362 (2019) |
| Dylan Cease | — | **3,308** | **2,104** | 5,400 (2024+2025) |
| Zac Gallen | — | **2,507** | **2,068** | 4,572 (2024+2025) |

### 2.3 구종별 비율 (Statcast pitch_type 기준)

#### Gerrit Cole (2019)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| FF (4-Seam Fastball) | 51.6% | Fastball |
| SL (Slider) | 23.2% | Slider |
| KC/CU (Knuckle Curve) | 15.4% | Curveball |
| CH (Changeup) | 7.4% | Changeup |
| SI (Sinker) | 2.4% | (군집화 시 Fastball에 흡수 가능) |

#### Dylan Cease (2024)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| FF (4-Seam Fastball) | 43.4% | Fastball |
| SL (Slider) | 43.0% | Slider |
| KC (Knuckle Curve) | 7.7% | Curveball |
| ST (Sweeper) | 4.4% | Sweeper |
| CH (Changeup) | 0.8% | (1% 미만) |
| FC (Cutter) | 0.6% | (1% 미만) |
| SI (Sinker) | 0.1% | (1% 미만) |

#### Dylan Cease (2025)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| SL (Slider) | 45.3% | Slider |
| FF (4-Seam Fastball) | 42.2% | Fastball |
| KC (Knuckle Curve) | 7.3% | Curveball |
| SI (Sinker) | 1.9% | Sinker |
| CH (Changeup) | 1.7% | Changeup |
| ST (Sweeper) | 1.6% | Sweeper |

> 2024 vs 2025 구종 분포 급변 없음 (모든 구종 ±10pp 이내)

#### Zac Gallen (2024)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| FF (4-Seam Fastball) | 46.3% | Fastball |
| KC (Knuckle Curve) | 27.7% | Curveball |
| CH (Changeup) | 14.0% | Changeup |
| SL (Slider) | 7.9% | Slider |
| FC (Cutter) | 3.9% | Cutter |
| SI (Sinker) | 0.2% | (1% 미만) |

#### Zac Gallen (2025)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| FF (4-Seam Fastball) | 46.7% | Fastball |
| KC (Knuckle Curve) | 24.6% | Curveball |
| CH (Changeup) | 14.0% | Changeup |
| SL (Slider) | 13.3% | Slider |
| SI (Sinker) | 1.2% | Sinker |
| FC (Cutter) | 0.1% | (1% 미만) |

> 2024 vs 2025 구종 분포 급변 없음 (모든 구종 ±10pp 이내)

---

## 3. 각 투수 구종 목록 (실제 학습 결과)

`clustering.py`가 UMAP+KMeans로 구종을 자동 식별합니다.
아래는 실제 학습 시 결정된 결과입니다.

| 투수 | 시즌 | 구종 (K) | Action Space | 실루엣 점수 |
|------|------|----------|-------------|------------|
| **Cole** | 2019 | Fastball, Slider, Curveball, Changeup (4) | 4 × 13 = **52** | 학습 완료 |
| **Cease** | 2024+2025 | Fastball (97mph), Slider, Changeup (3) | 3 × 13 = **39** | 0.8216 |
| **Gallen** | 2024+2025 | Fastball (93.5mph), Slider (88.2), Changeup (86.7), Curveball (81.0) (4) | 4 × 13 = **52** | 0.7851 |

> **Cease**: 3구종만 식별됨 — 슬라이더/스위퍼/커브가 하나의 "Slider" 군집으로 통합.
> **Gallen**: 4구종 식별 — FF/SL/CH/KC가 명확히 분리됨.

---

## 4. Mock 응답 예시 (UI 팀원용)

### 4.1 API 엔드포인트 설계 (안)

```
POST /api/recommend
```

### 4.2 요청 (Request)

```json
{
  "pitcher_id": "cease",
  "balls": 0,
  "strikes": 2,
  "outs": 1,
  "on_1b": 0,
  "on_2b": 1,
  "on_3b": 0,
  "batter_id": 660271
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `pitcher_id` | string | Y | 투수 식별자: `"cole"`, `"cease"`, `"gallen"` |
| `balls` | int (0~3) | Y | 볼 카운트 |
| `strikes` | int (0~2) | Y | 스트라이크 카운트 |
| `outs` | int (0~2) | Y | 아웃 수 |
| `on_1b` | int (0/1) | Y | 1루 주자 |
| `on_2b` | int (0/1) | Y | 2루 주자 |
| `on_3b` | int (0/1) | Y | 3루 주자 |
| `batter_id` | int | N | MLB batter ID (없으면 랜덤 군집) |

### 4.3 응답 (Response)

```json
{
  "pitcher": {
    "name": "Dylan Cease",
    "mlbam_id": 656302,
    "cluster": 0,
    "cluster_name": "파워 FB/SL"
  },
  "situation": {
    "count": "0-2",
    "outs": 1,
    "runners": "_ 2B _",
    "batter_cluster": 5,
    "batter_cluster_desc": "컨택 히터"
  },
  "recommendation": {
    "pitch_name": "Slider",
    "zone": 14,
    "zone_description": "하우 바깥 (Down & Away) 체이스",
    "action_index": 27
  },
  "alternatives": [
    {"pitch_name": "Slider", "zone": 9, "zone_description": "스트라이크존 우하 코너"},
    {"pitch_name": "Curveball", "zone": 13, "zone_description": "하좌 바깥 (Down & In) 체이스"}
  ],
  "context": {
    "model_type": "DQN",
    "training_data": "Dylan Cease 2024+2025 (5,400 pitches)",
    "action_space": 39
  }
}
```

### 4.4 투수별 실제 DQN 추천 예시

아래는 **실제 학습된 DQN 모델**의 정책 출력입니다.

#### 상황: 0아웃 주자 없음 — 볼카운트별 추천

**Cease DQN (평균 보상: +0.280 ± 1.125):**
| 카운트 | 추천 구종 | 추천 존 | 설명 |
|--------|----------|--------|------|
| 0-0 | Fastball | Zone 11 | 높고 몸쪽 (Up & In) |
| 0-2 | Changeup | Zone 2 | 스트라이크존 중상 |
| 1-2 | Changeup | Zone 2 | 스트라이크존 중상 |
| 3-2 | Fastball | Zone 11 | 높고 몸쪽 (Up & In) |

> Cease DQN은 패스트볼 비중 82.6%로 압도적. 2스트라이크 후 체인지업 전환 패턴.
> 구종 분포: Fastball 82.6%, Changeup 10.2%, Slider 7.1%

**Gallen DQN (평균 보상: +0.266 ± 1.115):**
| 카운트 | 추천 구종 | 추천 존 | 설명 |
|--------|----------|--------|------|
| 0-0 | Curveball | Zone 1 | 스트라이크존 좌상 |
| 0-2 | Curveball | Zone 1 | 스트라이크존 좌상 |
| 1-1 | Changeup | Zone 7 | 스트라이크존 좌하 |
| 3-2 | Changeup | Zone 7 | 스트라이크존 좌하 |

> Gallen DQN은 4구종 고루 사용. 커브볼로 리드 + 체인지업으로 마무리하는 패턴.
> 구종 분포: Fastball 35.3%, Curveball 33.6%, Slider 18.3%, Changeup 12.8%

> **참고**: 같은 상황이라도 `batter_cluster`에 따라 추천이 달라집니다.
> 위 예시는 랜덤 타자 군집 기준입니다.

---

## 5. 학습 데이터 요구사항

### 5.1 학습 결과 비교

| 항목 | Cole (2019) | Cease (2024+2025) | Gallen (2024+2025) |
|------|------------|------------------|-------------------|
| 투구 수 | 3,362 | 5,400 | 4,572 |
| 시즌 | 1 시즌 | 2 시즌 | 2 시즌 |
| 식별 구종 (K) | 4 | 3 | 4 |
| Action Space | 52 | 39 | 52 |
| DQN 학습 스텝 | 300,000 | 300,000 | 300,000 |
| **평균 보상** | **+0.436 ± 1.255** | **+0.280 ± 1.125** | **+0.266 ± 1.115** |
| 주력 구종 | Fastball 51.3% | Fastball 82.6% | Fastball 35.3% |

### 5.2 데이터 충분성

| 투수 | 사용 시즌 | 투구 수 | Cole 대비 | 판정 |
|------|----------|---------|----------|------|
| Cease | 2024+2025 | 5,400 | 161% | **충분** |
| Gallen | 2024+2025 | 4,572 | 136% | **충분** |

> 2시즌 결합 시 구종 분포 급변 없음 확인 (모든 구종 ±10pp 이내).

### 5.3 권장 사항

1. **최소 투구 수 기준**: 약 **2,000건 이상**이면 학습 가능
   - DQN은 MLP 시뮬레이터 위에서 학습하므로 투구 데이터는 구종 식별에만 사용
   - 범용 모델 사용 시 개인 데이터는 구종 식별에만 필요
   - 범용 모델 사용이면 **수백 건만으로도 구종 식별 가능** (군집화만 하면 됨)

2. **범용 모델 vs 개인 모델**:

   | 방식 | 개인 데이터 용도 | MLP 학습 | 필요 투구 수 |
   |------|----------------|---------|-------------|
   | `USE_UNIVERSAL_MODEL=True` (권장) | 구종 식별만 | 72만 건 범용 모델 재활용 | ~500건+ |
   | `USE_UNIVERSAL_MODEL=False` | MLP까지 직접 학습 | 개인 데이터만 | ~3,000건+ |

---

## 6. 학습 실행 방법 (참고)

```bash
# Cease 2024+2025 학습
uv run scripts/main_cease.py

# Gallen 2024+2025 학습
uv run scripts/main_gallen.py
```

학습 완료 후 산출물:
- `dqn_cease_2024_2025.zip` — Cease DQN 모델
- `dqn_gallen_2024_2025.zip` — Gallen DQN 모델
- `best_dqn_model_cease/best_model.zip` — EvalCallback 기준 최고 모델
- `best_dqn_model_gallen/best_model.zip` — EvalCallback 기준 최고 모델
- W&B에 학습 곡선, 구종 분포, 정책 테이블 자동 로깅
  - Cease: `Dylan_Cease_Pipeline` (run `lytwhlai`)
  - Gallen: `Zac_Gallen_Pipeline` (run `6xmr9b50`)

---

## 7. UI 구현 시 고려사항

1. **투수 선택 UI**: 드롭다운으로 Cole / Cease / Gallen 선택
2. **상황 입력 UI**: 볼카운트(0-0 ~ 3-2), 아웃(0~2), 주자 토글(1루/2루/3루)
3. **타자 입력** (선택): MLB 타자 이름 → ID 조회 → 군집 매핑, 없으면 랜덤
4. **결과 표시**: 추천 구종 + 존 시각화 (스트라이크존 그리드 위에 표시)
5. **응답 시간**: DQN `.predict()` ≈ 1ms 미만 (사전 로드 시)
6. **존 다이어그램**: `docs/image.png` 참조 — UI에서 스트라이크존 시각화 시 이 배치(포수 시점, 4코너 볼존)를 따를 것
7. **좌타/우타 시점 주의**: 존 11~14의 In/Away 의미가 좌타자와 우타자에서 반대. UI에 표시 시점(포수/타자) 명시 필요
