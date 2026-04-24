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

| 투수 | 2019 | 2023 | 2024 | 합계 (학습 대상) |
|------|------|------|------|-----------------|
| Gerrit Cole | **3,362** | — | — | 3,362 (2019) |
| Dylan Cease | — | **3,268** | **3,263** | 6,531 (2023+2024) |
| Zac Gallen | — | **3,252** | **2,588** | 5,840 (2023+2024) |

### 2.3 구종별 비율 (Statcast pitch_type 기준)

#### Gerrit Cole (2019)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| FF (4-Seam Fastball) | 51.6% | Fastball |
| SL (Slider) | 23.2% | Slider |
| KC/CU (Knuckle Curve) | 15.4% | Curveball |
| CH (Changeup) | 7.4% | Changeup |
| SI (Sinker) | 2.4% | (군집화 시 Fastball에 흡수 가능) |

#### Dylan Cease (2023)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| FF (4-Seam Fastball) | 43.2% | Fastball |
| SL (Slider) | 38.6% | Slider |
| KC (Knuckle Curve) | 15.2% | Curveball |
| CH (Changeup) | 3.0% | Changeup |

#### Dylan Cease (2024)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| FF (4-Seam Fastball) | 43.4% | Fastball |
| SL (Slider) | 42.7% | Slider |
| KC (Knuckle Curve) | 8.1% | Curveball |
| ST (Sweeper) | 4.2% | Sweeper |
| CH (Changeup) | 0.9% | (1% 미만, 제외 가능) |
| FC (Cutter) | 0.6% | (1% 미만, 제외) |

#### Zac Gallen (2023)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| FF (4-Seam Fastball) | 49.7% | Fastball |
| KC (Knuckle Curve) | 22.7% | Curveball |
| CH (Changeup) | 13.9% | Changeup |
| FC (Cutter) | 9.6% | Cutter |
| SL (Slider) | 4.0% | Slider |

#### Zac Gallen (2024)

| pitch_type | 비율 | 구종명 |
|-----------|------|--------|
| FF (4-Seam Fastball) | 46.0% | Fastball |
| KC (Knuckle Curve) | 27.5% | Curveball |
| CH (Changeup) | 13.9% | Changeup |
| SL (Slider) | 8.2% | Slider |
| FC (Cutter) | 4.1% | Cutter |

---

## 3. 각 투수 구종 목록 (예상 Action Space)

`clustering.py`가 UMAP+KMeans로 구종을 자동 식별하므로
실제 결과는 학습 시 결정됩니다. 아래는 Statcast 분포 기반 예상입니다.

| 투수 | 시즌 | 예상 구종 (K) | Action Space | 비고 |
|------|------|-------------|-------------|------|
| **Cole** | 2019 | Fastball, Slider, Curveball, Changeup (4) | 4 × 13 = **52** | 기존 학습 완료 |
| **Cease** | 2023 | Fastball, Slider, Curveball, Changeup (4) | 4 × 13 = **52** | CH 3%로 3구종 가능 |
| **Cease** | 2024 | Fastball, Slider, Curveball, Sweeper (4) | 4 × 13 = **52** | CH/FC 1% 미만 제외 |
| **Gallen** | 2023 | Fastball, Curveball, Changeup, Cutter, Slider (5) | 5 × 13 = **65** | SL 4%로 4구종 가능 |
| **Gallen** | 2024 | Fastball, Curveball, Changeup, Slider, Cutter (5) | 5 × 13 = **65** | 5구종 모두 4%+ |

> **주의**: 구종 수(K)는 `clustering.py`가 실루엣 점수 기반으로 K=3~6 중 자동 선택합니다.
> 위 표는 예상치이며, 실제 학습 시 달라질 수 있습니다.

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
    "training_data": "Dylan Cease 2023 (3,268 pitches)",
    "action_space": 52
  }
}
```

### 4.4 투수별 Mock 응답 예시

#### 상황: 1-2 카운트, 1아웃, 주자 1루

**Cole 추천:**
```json
{ "pitch_name": "Slider", "zone": 14, "zone_description": "하우 바깥 (Down & Away) 체이스" }
```
> 1-2 카운트에서 슬라이더를 낮고 바깥(zone 14)으로 빠지게 던져 헛스윙 유도 — Cole의 대표 결정구 패턴.

**Cease 추천:**
```json
{ "pitch_name": "Slider", "zone": 14, "zone_description": "하우 바깥 (Down & Away) 체이스" }
```
> 쫓는 카운트(0-2)에서 슬라이더를 낮은 바깥(zone 14)으로 빠뜨려 체이스 유도. Cease의 슬라이더(38.6%)는 주력 결정구.

**Gallen 추천:**
```json
{ "pitch_name": "Curveball", "zone": 13, "zone_description": "하좌 바깥 (Down & In) 체이스" }
```
> Gallen의 너클커브(22.7%)는 수직 낙차가 큰 구종. 낮은 존(zone 13)으로 떨어뜨려 타자의 스윙을 유도하는 전략.

> **주의**: 위 Mock 데이터는 예상 응답이며, 실제 DQN 학습 후 결과와 다를 수 있습니다.
> 같은 상황이라도 `batter_cluster`에 따라 추천이 달라집니다.

---

## 5. 학습 데이터 요구사항

### 5.1 기준: Cole 2019

| 항목 | 값 |
|------|-----|
| 투구 수 | 3,362 |
| 시즌 | 1 시즌 (2019) |
| DQN 학습 스텝 | 300,000 |
| 학습 시간 | ~10~15분 (CPU) |
| 평가 결과 | +0.436 ± 1.255 |

### 5.2 Cease / Gallen 데이터 충분성

| 투수 | 사용 시즌 | 투구 수 | Cole 대비 | 판정 |
|------|----------|---------|----------|------|
| Cease | 2023 단독 | 3,268 | 97% | **충분** |
| Cease | 2023+2024 | 6,531 | 194% | **충분** (구종 변화 주의) |
| Gallen | 2023 단독 | 3,252 | 97% | **충분** |
| Gallen | 2023+2024 | 5,840 | 174% | **충분** |

### 5.3 권장 사항

1. **단일 시즌 사용 권장**: 2023 또는 2024 중 택 1
   - Cease 2024: 슬라이더 비중이 급증(38.6% → 42.7%), 커브 비중 반감(15.2% → 8.1%)
   - 시즌 간 구종 믹스가 크게 다르면 `clustering.py`가 다른 K를 선택할 수 있음
   - 2시즌 합치면 구종 경계가 불명확해질 위험

2. **최소 투구 수 기준**: 약 **2,000건 이상**이면 학습 가능
   - DQN은 MLP 시뮬레이터 위에서 학습하므로 투구 데이터는 MLP 학습에만 사용
   - 범용 모델(`USE_UNIVERSAL_MODEL=True`) 사용 시 개인 데이터는 구종 식별에만 필요
   - 범용 모델 사용이면 **수백 건만으로도 구종 식별 가능** (군집화만 하면 됨)

3. **범용 모델 vs 개인 모델**:

   | 방식 | 개인 데이터 용도 | MLP 학습 | 필요 투구 수 |
   |------|----------------|---------|-------------|
   | `USE_UNIVERSAL_MODEL=True` (권장) | 구종 식별만 | 72만 건 범용 모델 재활용 | ~500건+ |
   | `USE_UNIVERSAL_MODEL=False` | MLP까지 직접 학습 | 개인 데이터만 | ~3,000건+ |

---

## 6. 학습 실행 방법 (참고)

```bash
# Cease 2023 학습
uv run src/main.py
# 프롬프트: Dylan Cease / 2023-03-30 / 2023-10-01

# Gallen 2023 학습
uv run src/main.py
# 프롬프트: Zac Gallen / 2023-03-30 / 2023-10-01
```

학습 완료 후 산출물:
- `smartpitch_dqn_final.zip` → 투수별로 이름 변경 필요 (예: `dqn_cease_2023.zip`)
- W&B에 학습 곡선, 구종 분포, 정책 테이블 자동 로깅

---

## 7. UI 구현 시 고려사항

1. **투수 선택 UI**: 드롭다운으로 Cole / Cease / Gallen 선택
2. **상황 입력 UI**: 볼카운트(0-0 ~ 3-2), 아웃(0~2), 주자 토글(1루/2루/3루)
3. **타자 입력** (선택): MLB 타자 이름 → ID 조회 → 군집 매핑, 없으면 랜덤
4. **결과 표시**: 추천 구종 + 존 시각화 (스트라이크존 그리드 위에 표시)
5. **응답 시간**: DQN `.predict()` ≈ 1ms 미만 (사전 로드 시)
6. **존 다이어그램**: `docs/image.png` 참조 — UI에서 스트라이크존 시각화 시 이 배치(포수 시점, 4코너 볼존)를 따를 것
7. **좌타/우타 시점 주의**: 존 11~14의 In/Away 의미가 좌타자와 우타자에서 반대. UI에 표시 시점(포수/타자) 명시 필요
