# SmartPitch 코드 워크스루 — 처음 배우는 팀원용

이 문서는 강화학습(RL), MLP(신경망), MDP(마르코프 결정 과정)를 처음 접하는 팀원이
소스 코드를 직접 읽으며 이해할 수 있도록 작성했습니다.

각 파일을 데이터 흐름 순서대로 설명합니다.

---

## 1. `src/batter_clustering.py` — 타자를 유형별로 나누는 파일

### 이 파일이 뭐 하는 파일인지
> 야구 스카우트가 "이 타자는 파워히터", "이 타자는 컨택히터"처럼 분류하는 작업을
> 9개 지표를 기반으로 컴퓨터가 자동으로 하는 파일.

### 꼭 봐야 하는 함수 목록

| 함수 | 줄 | 한 줄 설명 |
|------|-----|----------|
| `__init__` | 65 | 클래스 초기화 (최소 투구 수 기준 설정) |
| `fetch_statcast_data` | 80 | MLB 전체 투구 데이터를 인터넷에서 다운로드 |
| `_extract_batter_features` | 95 | 타자마다 9개 타격 지표를 계산 |
| `_apply_umap_kmeans` | 223 | 9개 지표를 2D로 압축 → 8개 그룹으로 분류 |
| `run_clustering_pipeline` | 300 | 위 함수들을 순서대로 실행하는 총괄 함수 |

### 각 함수가 하는 일

**`fetch_statcast_data`** (줄 80)
MLB 공식 트래킹 시스템(Statcast)에서 2023 시즌 전체 투구 데이터(~72만 건)를 다운로드한다.
마치 야구장에 설치된 카메라가 모든 공의 궤적을 기록한 것을 가져오는 것.

**`_extract_batter_features`** (줄 95)
각 타자별로 "이 타자는 어떤 스타일인가"를 숫자로 요약한다.
예를 들어 Mike Trout의 데이터가 이렇게 변환된다:

```
입력: Mike Trout의 2023 시즌 투구 1,847개 (매 투구의 결과·궤적·각도 등)
출력: [whiff%=28.1, z_contact%=80.2, o_swing%=31.5, o_contact%=52.3,
       avg_launch_angle=12.5, avg_launch_speed=91.3, barrel%=18.2,
       pull%=42.1, high_ff_whiff%=32.0]
       → "이 타자는 빠른 공에 약하고, 타구 속도가 빠르고, 당겨치는 스타일"
```

**`_apply_umap_kmeans`** (줄 223)
9개 숫자를 가진 타자들을 비슷한 특성끼리 8개 그룹으로 묶는다.

```
UMAP(차원 축소 = 9개 숫자를 2개 좌표로 압축) → 지도 위에 점 찍기
KMeans(군집화 = 가까운 점끼리 묶기) → 8개 색으로 칠하기
```

### 입력과 출력

```
입력: pybaseball.statcast("2023-03-30", "2023-10-01")
      → 2023 MLB 전체 투구 ~72만 건 DataFrame

출력: data/batter_clusters_2023.csv
      batter_id  stand  cluster
      545361     R      3         ← "이 타자는 3번 유형"
      660271     L      7         ← "이 타자는 7번 유형"
      ...
```

### 핵심 코드 해설

```python
# 줄 109-111: 500구 이상 받은 타자만 분석 (샘플 너무 적으면 통계가 불안정)
min_pitches = self.n_pitches_threshold  # 기본값 500
batter_counts = stand_df.groupby('batter').size()
valid_batters = batter_counts[batter_counts >= min_pitches].index

# 줄 196: 0으로 나누기 방지 함수 (분모가 0이면 0을 반환)
def safe_div(a, b): return a / b if b > 0 else 0.0

# 줄 250-252: K-Means로 8개 그룹으로 분류 (K=8은 실험 결과 가장 안정적)
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
labels = kmeans.fit_predict(embedding)  # embedding = UMAP 결과 (타자 수 × 2)

# 줄 336-340: 최종 결과를 CSV로 저장
cluster_df.to_csv(output_path, index=False)
# → data/batter_clusters_2023.csv
```

---

## 2. `src/pitcher_clustering.py` — 투수를 유형별로 나누는 파일

### 이 파일이 뭐 하는 파일인지
> 야구 해설가가 "Cole은 파워 투수", "Kershaw는 다구종 투수"라고 분류하는 작업을
> 15개 투구 지표로 컴퓨터가 자동으로 4가지 유형으로 나누는 파일.

### 꼭 봐야 하는 함수 목록

| 함수 | 줄 | 한 줄 설명 |
|------|-----|----------|
| `_extract_pitcher_features` | 105 | 투수마다 15개 투구 지표를 계산 |
| `_apply_umap_kmeans` | 185 | 15개 지표 → 2D 압축 → 최적 K 탐색(4~8) |
| `run_clustering_pipeline` | 261 | 전체 파이프라인 실행 |

### 각 함수가 하는 일

**`_extract_pitcher_features`** (줄 105)
타자 분석과 비슷하지만, 투수 관점의 지표 15개를 계산한다:

```
입력: Gerrit Cole의 2023 시즌 투구 2,847개
출력: [avg_speed=96.5, ff_pct=44.7%, sl_pct=25.6%, whiff%=25.9%,
       zone%=48.2%, release_pos_x=-1.2, pfx_x=5.3, pfx_z=14.2, ...]
       → "이 투수는 빠른 직구와 슬라이더 중심, 높은 헛스윙 유도율"
```

**`_apply_umap_kmeans`** (줄 185)
타자와 달리 K를 고정하지 않고, 4~8 중 가장 잘 나뉘는 K를 자동 탐색한다.

```
실루엣 점수(silhouette score) = "군집이 얼마나 깔끔하게 나뉘었는가" (0~1, 높을수록 좋음)
2023 결과: K=4가 실루엣 0.4502로 최고 → 4개 유형으로 결정
```

### 입력과 출력

```
입력: 같은 Statcast 72만 건 (batter_clustering과 동일 데이터 재활용 가능)

출력: data/pitcher_clusters_2023.csv
      pitcher_id  cluster
      543037      0         ← "파워 패스트볼/슬라이더형"
      477132      3         ← "멀티피치/아스날형"

군집 결과 (K=4, 479명):
  0: 파워형 (157명) — 직구+슬라이더 중심, 구속 89.9
  1: 핀세스형 (102명) — 제구 중심, 변화구 다양
  2: 싱커볼형 (103명) — 투심 중심, 땅볼 유도
  3: 멀티피치형 (117명) — 구종수 4.4개, Darvish·Kershaw 타입
```

### 핵심 코드 해설

```python
# 줄 62: 분석할 구종 목록 (MLB 주요 6구종)
PITCH_TYPES = ['FF', 'SI', 'SL', 'CH', 'CU', 'FC']
# FF=포심, SI=투심, SL=슬라이더, CH=체인지업, CU=커브, FC=커터

# 줄 164-169: 각 구종의 사용 비율 계산 (전체 투구 중 몇 %를 이 구종으로 던지나)
for pt in PITCH_TYPES:
    col = f'{pt}_pct'
    pitcher_features[col] = pitcher_features[pt] / pitcher_features['total_pitches']

# 줄 213-220: 실루엣 점수로 최적 K 자동 탐색
for k in range(4, 9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embedding)
    score = silhouette_score(embedding, labels)
    # score가 가장 높은 k를 선택 → 2023 시즌에서는 k=4
```

---

## 3. `src/data_loader.py` — 특정 투수의 투구 데이터를 가져오는 파일

### 이 파일이 뭐 하는 파일인지
> 실제 분석할 투수(예: Gerrit Cole)의 특정 시즌 투구 기록을 가져와서
> "우리 모델이 먹을 수 있는 형태"로 정리하는 파일. 식당의 식재료 손질 담당.

### 꼭 봐야 하는 함수 목록

| 함수 | 줄 | 한 줄 설명 |
|------|-----|----------|
| `_fetch_data` | 62 | 투수 이름으로 MLB ID 찾고 → 그 투수의 투구 데이터 다운로드 |
| `_preprocess_data` | 83 | 결측값 처리, 필요한 컬럼만 추출 |
| `upload_to_wandb` | 106 | 정제된 데이터를 W&B(실험 관리 플랫폼)에 업로드 |
| `load_and_prepare_data` | 140 | 위 3개를 순서대로 실행 |

### 각 함수가 하는 일

**`_fetch_data`** (줄 62)
```
입력: "Gerrit", "Cole", "2019-03-28", "2019-09-29"
처리: playerid_lookup("Cole", "Gerrit") → MLBAM ID = 543037
      statcast_pitcher("2019-03-28", "2019-09-29", 543037)
출력: Cole의 2019 시즌 투구 ~2,800건 DataFrame
```

**`_preprocess_data`** (줄 83)
```
입력: raw 데이터 (100개 이상의 컬럼, NaN 포함)
처리:
  - on_1b/2b/3b: NaN → "0" (주자 없음), 선수ID → "1" (주자 있음)
  - 필요한 16개 컬럼만 추출:
    features (6개): release_speed, release_spin_rate, pfx_x, pfx_z, plate_x, plate_z
    meta (10개): description, zone, balls, strikes, outs_when_up, on_1b, on_2b, on_3b, ...
  - NaN 있는 행 삭제
출력: 깨끗한 DataFrame (~2,700건)
```

### 핵심 코드 해설

```python
# 줄 67-72: 투수 이름으로 MLB 공식 ID를 찾고, 그 ID로 투구 데이터 다운로드
player_info = playerid_lookup(self.last_name, self.first_name)
self.pitcher_mlbam_id = int(player_info['key_mlbam'].iloc[0])
self.raw_data = statcast_pitcher(self.start_date, self.end_date, self.pitcher_mlbam_id)

# 줄 91-93: 주자 정보를 0/1로 변환 (NaN=주자 없음, 숫자=주자 있음)
for base in ['on_1b', 'on_2b', 'on_3b']:
    df[base] = df[base].apply(lambda x: '0' if pd.isna(x) else '1')

# 줄 96-97: 모델에 필요한 컬럼만 추출
features = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z']
meta = ['description', 'zone', 'balls', 'strikes', 'outs_when_up', ...]
```

---

## 4. `src/model.py` — 던지면 결과가 뭐가 나올지 예측하는 파일 (MLP)

### 이 파일이 뭐 하는 파일인지
> "3-2 카운트, 1아웃, 주자 1루, 타자 유형 3번에게 직구를 몸쪽 높게 던지면?"
> → "볼 35%, 스트라이크 30%, 파울 20%, 인플레이 15%" 확률을 예측하는 두뇌.
> 이것이 MLP(Multi-Layer Perceptron, 다층 신경망)이다.

### MLP가 뭔지 야구로 비유하면
```
MLP = 경험 많은 포수의 두뇌

포수가 "이 상황에서 이 공을 던지면 어떤 결과가 나올까?" 판단하는 것처럼,
MLP는 40개의 숫자 입력을 받아서 4가지 결과의 확률을 출력한다.

입력 40개 = 상황 정보(볼카운트·아웃·주자·타자유형·투수유형·구종·코스)
출력 4개  = ball / strike / foul / hit_into_play 의 확률

"두뇌"의 구조:
  입력(40개) → [256개 뉴런] → [128개 뉴런] → [64개 뉴런] → 출력(4개 확률)
  각 화살표에 "가중치(weight)"가 있고, 학습 = 이 가중치를 조절하는 과정.
```

### 꼭 봐야 하는 함수 목록

| 함수 | 줄 | 한 줄 설명 |
|------|-----|----------|
| `MLP.__init__` | 66 | 신경망 구조 정의 (레이어 쌓기) |
| `MLP.forward` | 85 | 입력 → 출력 계산 (순전파) |
| `load_from_checkpoint` | 106 | 저장된 모델 불러오기 (범용 모델 로드) |
| `_prepare_data` | 168 | 투구 데이터를 신경망 입력 형태로 변환 |
| `train_model` | 296 | 신경망 학습 (가중치 조절 반복) |
| `predict_proba` | 488 | 학습된 모델로 결과 확률 예측 |
| `run_modeling_pipeline` | 470 | 데이터 준비 + 학습을 한 번에 실행 |

### 각 함수가 하는 일

**`MLP.__init__`** (줄 66)
신경망 "설계도"를 만든다. 건물의 층수와 방 수를 결정하는 것.
```
입력: input_dim=40, output_dim=4, hidden_dims=[256,128,64]
결과: 4층짜리 건물
  1층: 40개 입구 → 256개 방 (+ BatchNorm + ReLU + Dropout)
  2층: 256개 → 128개 방
  3층: 128개 → 64개 방
  4층: 64개 → 4개 출구 (ball/strike/foul/hit_into_play)
```

**`_prepare_data`** (줄 168)
투구 데이터를 MLP가 이해할 수 있는 숫자 배열로 변환한다.

```
입력: 투구 DataFrame (각 행 = 하나의 투구)
      description  zone  balls  strikes  outs  on_1b  mapped_pitch_name  batter_cluster  pitcher_cluster
      "strike"     5     1      2        1     1      "Fastball"         3               0

처리:
  1. hit_by_pitch 제거 (전체의 0.3%, 비의도적 결과)
  2. 수치 피처 6개: balls=1, strikes=2, outs=1, on_1b=1, on_2b=0, on_3b=0
  3. 범주형 피처 one-hot 인코딩 (하나의 칸만 1, 나머지 0):
     mapped_pitch_name_Fastball=1, ..._Slider=0, ..._Curveball=0, ...
     zone_5=1, zone_1=0, zone_2=0, ...
     batter_cluster_3=1, batter_cluster_0=0, ...
     pitcher_cluster_0=1, pitcher_cluster_1=0, ...
  4. 결과를 concat → [1, 2, 1, 1, 0, 0, 1, 0, 0, ..., 1, 0, ...] (40차원 벡터)

출력: X = (72만, 40) float 배열, y = (72만,) 정수 배열 (0=ball, 1=foul, 2=hit_into_play, 3=strike)
```

**`train_model`** (줄 296)
학습 = "정답과 비교해서 틀린 만큼 가중치를 수정"하는 과정을 반복하는 것.

```
비유: 신입 포수가 선배의 기록을 보고 연습하는 과정

for epoch (반복 회차) 1 ~ 20:
    for batch (묶음) in 훈련 데이터:
        1. 예측: MLP에 입력 넣고 → 예측 확률 받기
        2. 오차 계산: 정답(실제 결과)과 비교 → loss(손실값) 계산
        3. 역전파: loss를 줄이는 방향으로 가중치 수정
    검증 데이터로 현재 실력 테스트:
        val_acc = 맞춘 개수 / 전체 개수
        val_loss = 평균 오차

    EarlyStopping(조기 종료):
        "5번 연속 실력이 안 늘면 연습 중단"
        → 더 연습해도 안 느니까, 가장 잘했을 때의 가중치를 저장
```

**`predict_proba`** (줄 488)
학습 완료된 MLP에 새로운 상황을 넣고 확률을 받는 함수.

```
입력: 40차원 벡터 (특정 상황 + 구종 + 코스)
출력: [0.35, 0.20, 0.15, 0.30]
      = ball 35%, foul 20%, hit_into_play 15%, strike 30%
```

**`load_from_checkpoint`** (줄 106)
이미 학습된 모델을 파일에서 불러오는 함수. "이전에 연습 끝난 포수의 경험을 복원."

```python
# 이 함수가 읽는 파일 4개:
model_path           = "best_transition_model_universal.pth"   # 학습된 가중치
feature_columns_path = "data/feature_columns_universal.json"   # 입력 컬럼 순서
target_classes_path  = "data/target_classes_universal.json"    # 출력 클래스 이름
model_config_path    = "data/model_config_universal.json"      # 신경망 구조 (hidden_dims)
```

### 핵심 코드 해설

```python
# 줄 66-84: MLP 신경망 구조 정의
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128,64], dropout_rate=0.2):
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:       # [256, 128, 64] 각 층마다 반복
            layers.append(nn.Linear(prev_dim, h_dim))   # 선형 변환 (행렬 곱셈)
            layers.append(nn.BatchNorm1d(h_dim))         # 정규화 (학습 안정화)
            layers.append(nn.ReLU())                     # 활성화 (음수→0, 비선형성 추가)
            layers.append(nn.Dropout(dropout_rate))      # 20% 뉴런 랜덤 차단 (과적합 방지)
            prev_dim = h_dim

# 줄 228-235: 범주형 데이터를 one-hot으로 변환 + 수치 데이터와 합치기
X_cat_encoded = pd.get_dummies(self.df[['mapped_pitch_name', 'zone',
                                         'batter_cluster', 'pitcher_cluster']])
X_encoded = pd.concat([X_num, X_cat_encoded], axis=1).astype(float)
# .astype(float) = pandas 3.0에서 get_dummies가 bool을 반환하는 버그 대응

# 줄 394-405: EarlyStopping (5번 연속 개선 없으면 학습 중단)
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    torch.save(self.model.state_dict(), "best_transition_model.pth")
else:
    patience_counter += 1
    if patience_counter >= patience:  # patience = 5
        print(f"Early stopping: epoch {epoch}에서 중단")
        break
```

---

## 5. `src/universal_model_trainer.py` — 전체 MLB 데이터로 MLP를 학습시키는 파일

### 이 파일이 뭐 하는 파일인지
> model.py의 MLP를 "한 투수"가 아니라 "2023 시즌 전체 투수"의 72만 건 데이터로
> 학습시키는 파일. 신입 포수를 한 팀이 아니라 MLB 전체 경기 영상으로 교육하는 것.

### 꼭 봐야 하는 함수 목록

| 함수 | 줄 | 한 줄 설명 |
|------|-----|----------|
| `_preprocess_raw` | 144 | 72만 건 원시 데이터를 model.py 입력 형식으로 정리 |
| `_run_single_experiment` | 231 | 하나의 실험 설정으로 MLP 학습 + W&B 기록 |
| `main` | 317 | 3가지 실험을 순차 실행하고 최고 성능 모델 선정 |

### 각 함수가 하는 일

**`_preprocess_raw`** (줄 144)
Statcast 원시 데이터에서 MLB 고유 구종 코드를 우리 시스템의 구종 이름으로 변환한다.

```
PITCH_TYPE_MAP (줄 51-69):
  'FF' → 'Fastball'       (포심 패스트볼)
  'SI' → 'Sinker'         (투심 패스트볼)
  'SL' → 'Slider'         (슬라이더)
  'CH' → 'Changeup'       (체인지업)
  'CU' → 'Curveball'      (커브)
  ...

DESCRIPTION_MAP (줄 74-94):
  'called_strike'     → 'strike'         (심판 스트라이크 콜)
  'swinging_strike'   → 'strike'         (헛스윙)
  'ball'              → 'ball'           (볼)
  'foul'              → 'foul'           (파울)
  'hit_into_play'     → 'hit_into_play'  (인플레이 타구)
  'hit_by_pitch'      → 'hit_by_pitch'   (몸에 맞는 볼, 이후 제거됨)
  ... (12개 세부 코드 → 5개 그룹으로 병합)
```

**`_run_single_experiment`** (줄 231)
하나의 실험 설정(모델 크기, 학습률 스케줄러 등)으로 MLP를 학습하고 결과를 기록한다.

```
실험 3가지 (줄 116-141):
  Exp1: 더 큰 모델 [256,128,64] → val_acc 58.1% (최고)
  Exp2: LR 스케줄러 [128,64]    → val_acc 58.0%
  Exp3: 클래스 가중치 [128,64]  → val_acc 57.3% (소수 클래스 F1 개선)
```

**`main`** (줄 317)
3개 실험을 돌리고, val_acc(검증 정확도)가 가장 높은 모델을 "최종 범용 모델"로 저장한다.

```
입력: statcast("2023-03-30", "2023-10-01") → ~72만 건
출력:
  best_transition_model_universal.pth     ← 신경망 가중치
  data/feature_columns_universal.json     ← 입력 컬럼 순서 (40개)
  data/target_classes_universal.json      ← 출력 클래스 이름 (4개)
  data/model_config_universal.json        ← 신경망 구조 (hidden_dims=[256,128,64])
```

### 핵심 코드 해설

```python
# 줄 164-167: 구종 코드 변환 (MLB 코드 → 우리 시스템 이름)
df['mapped_pitch_name'] = df['pitch_type'].map(PITCH_TYPE_MAP)
df = df.dropna(subset=['mapped_pitch_name'])  # 매핑 안 되는 희귀 구종 제거

# 줄 348: 최고 실험 선정 (val_acc 기준, val_loss가 아님!)
best_result = max(results, key=lambda r: r["final_val_acc"])
# 이유: class-weighted loss를 쓰면 loss 스케일이 달라져서 loss 비교 불가

# 줄 352-355: 최고 실험의 파일을 "universal" 경로로 복사
shutil.copy(best_result["model_path"], MODEL_SAVE_PATH)
shutil.copy(best_result["feat_path"],  FEATURE_COLS_PATH)
shutil.copy(best_result["cls_path"],   TARGET_CLS_PATH)
shutil.copy(best_result["cfg_path"],   MODEL_CONFIG_PATH)
```

---

## 6. `src/mdp_solver.py` — 수학적으로 최적의 구종을 계산하는 파일 (MDP)

### 이 파일이 뭐 하는 파일인지
> 모든 가능한 상황(9,216개)에 대해 "이 상황에서 어떤 구종+코스가 최적인가"를
> 수학적으로 계산해서 정답표를 만드는 파일. 시험 전에 모든 문제의 정답지를 미리 만드는 것.

### MDP가 뭔지 야구로 비유하면
```
MDP = "매 순간 최선의 선택을 찾는 수학적 프레임워크"

상태(State): 지금 볼카운트가 뭐고, 몇 아웃이고, 주자가 어디 있고, 타자가 어떤 유형인가
행동(Action): 어떤 구종을 어느 코스에 던질 것인가
전이(Transition): 그 공을 던지면 어떤 결과(볼/스트라이크/파울/인플레이)가 나올 확률
보상(Reward): 그 결과로 실점 위험이 얼마나 줄었는가 (RE24 기준)

"0-0 카운트, 0아웃, 주자 없음, 타자 유형 3"에서
"직구 몸쪽"을 던지면 → strike 확률 35% → 카운트 0-1로 → 실점 위험 감소
"슬라이더 바깥쪽"을 던지면 → ball 확률 45% → 카운트 1-0으로 → 실점 위험 증가
→ 실점 위험을 가장 많이 줄이는 구종+코스를 선택
```

### 꼭 봐야 하는 함수 목록

| 함수 | 줄 | 한 줄 설명 |
|------|-----|----------|
| `_get_re24` | 85 | 현재 상황의 기대 실점 조회 |
| `_advance_runners_walk` | 91 | 볼넷 시 주자 이동 계산 |
| `_get_next_states_and_rewards` | 103 | "이 공을 던지면 다음에 어떻게 되나" 계산 |
| `solve_mdp` | 192 | 9,216개 상태 전부에 대해 최적 행동 계산 (핵심!) |
| `run_optimizer` | 338 | 전체 MDP 파이프라인 실행 |

### 각 함수가 하는 일

**`_get_re24`** (줄 85)
RE24 매트릭스에서 현재 상황의 "기대 실점"을 조회한다.

```
RE24 = "이 상황에서 이닝 끝까지 평균 몇 점이 나는가" (2019 MLB 기준)

예시:
  0아웃, 주자 없음 → 0.481점 (가장 안전)
  0아웃, 만루     → 2.282점 (가장 위험)
  2아웃, 주자 없음 → 0.098점 (거의 안전)
```

**`_get_next_states_and_rewards`** (줄 103)
"이 상황에서 이 공을 던지면, 4가지 결과 각각에 대해 다음 상태와 보상이 뭔가"를 계산.

```
현재: 1-1 카운트, 0아웃, 주자 1루, 타자 유형 3, 투수 유형 0
행동: "Slider + Zone 5"

결과별 다음 상태:
  ball(35%)  → 2-1 카운트, 0아웃, 주자 1루 (카운트만 변경)
  strike(30%) → 1-2 카운트, 0아웃, 주자 1루
  foul(20%)  → 1-2 카운트, 0아웃, 주자 1루 (2스트라이크 이후 파울은 카운트 유지)
  hit_into_play(15%) → 새 타석 시작 (아웃/안타/홈런 확률적)

보상 = RE24(현재) - RE24(다음) - 실점
  → 양수 = 실점 위험이 줄어들었다 (좋은 결과)
  → 음수 = 실점 위험이 늘어났다 (나쁜 결과)
```

**`solve_mdp`** (줄 192) — 이 파일의 핵심
9,216개 모든 상태에 대해 "어떤 행동이 최적인가"를 벨만 방정식(Bellman equation)으로 계산.

```
총 상태 수: 12(카운트) × 3(아웃) × 8(주자) × 8(타자군집) × 4(투수군집) = 9,216개

벨만 방정식 (가치 반복):
  V(상태) = max_행동 [ Σ 확률(결과) × (보상 + V(다음 상태)) ]
  = "이 상태의 가치 = 최적 행동을 골랐을 때 기대할 수 있는 최대 보상"

가치 반복(Value Iteration)을 5회 돌리는 이유:
  파울이 나면 카운트가 유지됨 → 순환 구조 발생
  5회 반복하면 파울 순환이 수렴함
```

### 핵심 코드 해설

```python
# 줄 73-80: RE24 매트릭스 (2019 MLB 평균 기대 실점)
re24_matrix = {
    "0_000": 0.481,  # 0아웃, 주자 없음 → 이닝 끝까지 평균 0.481점
    "0_100": 0.859,  # 0아웃, 1루 주자 → 0.859점
    "0_111": 2.282,  # 0아웃, 만루 → 2.282점
    "2_000": 0.098,  # 2아웃, 주자 없음 → 0.098점 (거의 이닝 끝)
}

# 줄 208-211: 9,216개 상태 생성 (모든 가능한 야구 상황)
for count in ALL_COUNTS:           # 12개 (0-0, 0-1, 0-2, 1-0, ..., 3-2)
    for outs in ['0', '1', '2']:   # 3개
        for runners in ALL_RUNNERS: # 8개 (000, 001, 010, ..., 111)
            for batter in ALL_BATTERS:   # 8개
                for pitcher in self.pitcher_clusters:  # 4개

# 줄 270-286: 기대 보상 계산 (벨만 방정식의 핵심)
expected_value = 0.0
for (prob, next_state, immediate_reward) in transitions:
    future_value = self.state_values.get(next_state, 0.0)  # 다음 상태의 가치
    expected_value += prob * (immediate_reward + future_value)
# → expected_value가 가장 높은 행동을 최적 행동으로 선택
```

---

## 7. `src/pitch_env.py` — 가상 야구 시뮬레이터 (Gymnasium 환경)

### 이 파일이 뭐 하는 파일인지
> 컴퓨터 안에 야구장을 만들어서, AI 에이전트가 수만 번 투구를 연습할 수 있게 하는 파일.
> 게임 속 야구장이라고 생각하면 된다. AI가 공을 던지면 결과가 나오고, 점수가 매겨진다.

### Gymnasium 환경이 뭔지 야구로 비유하면
```
Gymnasium = "AI 훈련용 시뮬레이터 규격"

모든 AI 게임은 이 규격을 따름:
  reset()  → 게임 시작 (새 이닝 시작, 초기 상태 설정)
  step(action) → 한 수 두기 (공 하나 던지기)
    → 반환: (새 상태, 보상, 게임 끝?, 추가 정보)

예시:
  env.reset()          → 0아웃, 0-0, 주자 없음, 랜덤 타자
  env.step(42)         → 행동 42 = "직구 + 존 5"
    → (새 상태, 보상 +0.05, False, {})
  env.step(17)         → 행동 17 = "슬라이더 + 존 12"
    → (새 상태, 보상 -0.30, False, {})
  ...
  env.step(8)          → 3아웃 달성!
    → (최종 상태, 보상 +0.10, True, {})  ← 이닝 종료
```

### 꼭 봐야 하는 함수 목록

| 함수 | 줄 | 한 줄 설명 |
|------|-----|----------|
| `__init__` | 64 | 야구장 설정 (구종 수, 존 수, 행동 공간 정의) |
| `reset` | 136 | 새 이닝 시작 (랜덤 아웃·주자·타자 배치) |
| `step` | 166 | 공 하나 던지기 → 결과 반영 → 보상 계산 |
| `_sample_outcome` | 233 | MLP에게 "이 공 던지면 뭐가 나와?" 물어보기 |
| `_apply_outcome` | 275 | 결과를 야구 규칙대로 적용 (볼 → 카운트 변경 등) |
| `_apply_walk` | 321 | 볼넷/사구 → 주자 이동 |
| `_apply_batted_ball` | 328 | 인플레이 타구 → 아웃/안타/홈런 확률적 판정 |

### 각 함수가 하는 일

**`step`** (줄 166) — 이 파일의 핵심
AI가 행동(구종+코스)을 선택하면, 결과를 시뮬레이션하고 보상을 준다.

```
1. 행동 디코딩: action=42 → pitch_idx=3(Slider), zone_idx=3(존4)
2. MLP에게 물어보기: _sample_outcome() → "strike" (확률적 샘플링)
3. 보상 계산:
   re24_before = 현재 상황의 기대 실점 (예: 0.859)
   _apply_outcome("strike") → 카운트 변경
   re24_after = 다음 상황의 기대 실점 (예: 0.481)
   reward = re24_before - re24_after - runs_scored
          = 0.859 - 0.481 - 0 = +0.378 (좋은 결과!)
4. 이닝 종료 체크: 3아웃이면 done=True
```

**`_sample_outcome`** (줄 233)
MLP 모델에게 확률을 물어보고, 그 확률에 따라 결과를 "주사위 굴리기"로 결정.

```
MLP 예측: [ball=0.35, foul=0.20, hit_into_play=0.15, strike=0.30]
np.random.choice(결과, p=확률) → "strike" (30% 확률로 당첨)
```

**`_apply_batted_ball`** (줄 328)
인플레이 타구의 결과를 확률적으로 결정. (현재 하드코딩)

```
아웃: 70%   → 아웃 카운트 +1
1루타: 15%  → 주자 1루 배치
2루타: 10%  → 주자 2루 배치
홈런: 5%    → 모든 주자 + 타자 득점
```

### 핵심 코드 해설

```python
# 줄 48-55: RE24 매트릭스 (mdp_solver.py와 동일한 값을 두 곳에 보관)
RE24_MATRIX = {
    "0_000": 0.481,  "0_100": 0.859,  "0_010": 1.100, ...
    "2_111": 0.736,  # 2아웃 만루에서도 0.736점 기대
}

# 줄 178-190: 보상 계산 = "실점 위험 감소량"
re24_before = self._get_re24()            # 현재 기대 실점
self._apply_outcome(outcome)              # 결과 적용 (카운트/아웃/주자 변경)
re24_after = self._get_re24()             # 변경 후 기대 실점
reward = re24_before - re24_after - runs  # 보상 = 기대 실점 감소 - 실제 실점

# 줄 340-353: 인플레이 타구 결과 확률 (하드코딩)
outcomes = ['ground_out', 'single', 'double', 'home_run']
probs = [0.70, 0.15, 0.10, 0.05]
result = np.random.choice(outcomes, p=probs)

# 줄 145-146: 이닝 시작 시 랜덤 상황 생성 (다양한 상황 경험 → 학습 효과 향상)
self.outs = np.random.randint(0, 3)
self.on_1b, self.on_2b, self.on_3b = [np.random.randint(0, 2) for _ in range(3)]
```

---

## 8. `src/rl_trainer.py` — 시뮬레이터에서 반복 연습하는 파일 (DQN)

### 이 파일이 뭐 하는 파일인지
> AI 에이전트가 가상 야구장(pitch_env)에서 30만 번 이상 투구를 연습하며
> "어떤 상황에서 어떤 공을 던져야 실점을 줄이는가"를 스스로 학습하는 파일.

### DQN이 뭔지 야구로 비유하면
```
DQN(Deep Q-Network) = "시행착오로 배우는 AI"

MDP(mdp_solver)는 "수학 공식"으로 정답을 계산한다.
DQN은 "직접 던져보고 결과를 보면서" 배운다.

비유: MDP = 교과서로 공부, DQN = 실전 연습

DQN의 학습 과정:
  1. 처음에는 랜덤으로 던짐 (탐색, exploration)
  2. 좋은 결과가 나온 선택을 기억 (경험 리플레이, experience replay)
  3. 점점 좋은 선택을 더 자주 함 (활용, exploitation)
  4. 탐색률: 100% → 5% 로 점점 줄어듦 (처음엔 모험, 나중엔 실력 발휘)
```

### 꼭 봐야 하는 함수 목록

| 함수 | 줄 | 한 줄 설명 |
|------|-----|----------|
| `DQNTrainer.__init__` | 97 | 훈련 환경 + 평가 환경 설정 |
| `DQNTrainer.build` | 108 | DQN 에이전트 생성 (신경망 + 학습 전략) |
| `DQNTrainer.train` | 149 | 30만 스텝 학습 실행 |
| `DQNTrainer.evaluate` | 206 | 학습된 에이전트로 100이닝 테스트 |
| `DQNTrainer.print_policy_sample` | 276 | "0아웃 주자 없음"에서 12개 카운트별 추천 구종 출력 |

### 각 함수가 하는 일

**`build`** (줄 108)
DQN 에이전트를 만든다. 핵심 하이퍼파라미터:

```
DQN 설정 (줄 123-137):
  policy = "MlpPolicy"          → 신경망 기반 정책 (입력: 8D 관측 → 출력: 행동)
  learning_rate = 0.0001        → 한 번에 얼마나 배울지 (너무 크면 불안정)
  buffer_size = 100,000         → 최근 10만 경험을 기억 (경험 리플레이 버퍼)
  batch_size = 64               → 한 번에 64개 경험을 꺼내서 학습
  gamma = 0.99                  → 미래 보상 할인율 (0.99 = 미래도 거의 동등하게 중시)
  exploration_fraction = 0.30   → 전체의 30% 구간은 랜덤 탐색
  exploration_final_eps = 0.05  → 이후 5%만 랜덤 (95%는 학습된 대로)
  net_arch = [128, 64]          → DQN 내부 신경망 구조
```

**`train`** (줄 149)
30만 스텝(= 약 34,000 이닝) 동안 학습 실행.

```
train(total_timesteps=300_000)

내부적으로:
  - EvalCallback: 1만 스텝마다 성능 평가, 최고 성능 모델 자동 저장
  - WandbDQNCallback: 매 에피소드(이닝)마다 보상/탐색률을 W&B에 기록
  - 학습 완료 후 smartpitch_dqn_final.zip 저장
```

**`evaluate`** (줄 206)
학습된 에이전트로 100이닝을 시뮬레이션하고 성능 측정.

```
출력 예시:
  평균 보상: 0.436 ± 1.255
  구종 분포: {'Fastball': 447, 'Slider': 213, 'Curveball': 130, 'Changeup': 94}
  존 분포: {7: 287, 5: 110, 3: 93, ...}
  → Fastball 51.3%, Slider 24.3% — Cole의 실제 투구 비율과 유사
```

### 핵심 코드 해설

```python
# 줄 123-137: DQN 에이전트 생성
self.model = DQN(
    "MlpPolicy",           # 정책 = 신경망 (8D 관측 → 행동)
    self.train_env,         # 학습할 환경 (PitchEnv)
    buffer_size=100_000,    # 경험 기억 용량
    gamma=0.99,             # 미래 보상 할인
    exploration_fraction=exploration_fraction,  # 0.30 = 30% 탐색
    net_arch=[128, 64],     # DQN 내부 신경망 크기
)

# 줄 177-185: 콜백(학습 중 자동 실행되는 함수들) 설정
eval_callback = EvalCallback(
    self.eval_env,
    eval_freq=10_000,             # 1만 스텝마다 평가
    best_model_save_path="./best_dqn_model/",  # 최고 성능 모델 자동 저장
)

# 줄 227-230: 100이닝 평가 — 학습된 정책대로 행동 결정
action, _ = self.model.predict(obs, deterministic=True)
# deterministic=True: 항상 가장 좋은 행동 선택 (랜덤 없음)
# 학습 중에는 탐색(랜덤)도 하지만, 평가 시에는 실력 발휘만
```

---

## 9. `src/main.py` — 전체를 연결하는 지휘자

### 이 파일이 뭐 하는 파일인지
> 위의 모든 파일을 순서대로 호출하는 오케스트라 지휘자.
> "데이터 수집 → 구종 식별 → 모델 로드 → MDP 계산 → DQN 학습 → 평가"
> 전체 흐름을 하나의 명령(`uv run src/main.py`)으로 실행한다.

### 꼭 봐야 하는 함수 목록

| 함수 | 줄 | 한 줄 설명 |
|------|-----|----------|
| `_lookup_pitcher_cluster` | 44 | 투수 ID로 군집 번호 찾기 |
| `_get_all_pitcher_clusters` | 66 | 전체 투수 군집 ID 목록 가져오기 |
| `main` | 86 | 전체 파이프라인 실행 (핵심!) |

### main 함수의 실행 순서 (줄 86~)

```
실행: uv run src/main.py
입력: 투수 이름(Gerrit Cole), 시작일(2019-03-28), 종료일(2019-09-29)

[단계 1] 데이터 수집 (줄 134)
  PitchDataLoader("Gerrit", "Cole", ...).load_and_prepare_data()
  → Cole의 2019 시즌 투구 ~2,800건

[단계 2] 구종 식별 (줄 150)
  PitchClustering(df).run_clustering_pipeline()
  → UMAP+KMeans로 Cole의 구종 자동 식별
  → ["Fastball", "Slider", "Curveball", "Changeup"]

[단계 3] 모델 로드 또는 학습 (줄 160)
  if USE_UNIVERSAL_MODEL:  ← True (권장)
      TransitionProbabilityModel.load_from_checkpoint(...)
      → 72만 건으로 학습된 범용 모델 로드
  else:  ← 레거시
      model.run_modeling_pipeline(...)
      → Cole의 2,800건으로 직접 학습 (정확도 낮음)

[단계 4] MDP 최적 정책 (줄 193)
  MDPOptimizer.run_optimizer()
  → 9,216개 상태 전부에 대해 최적 구종+코스 계산
  → optimal_policy = {"0-0_0_000_3_0": ("Fastball", "Zone 5"), ...}

[단계 5] DQN 강화학습 (줄 226)
  DQNTrainer.build() → .train() → .evaluate()
  → 30만 스텝 학습, 100이닝 평가
  → 평균 보상 0.436, Fastball 51.3%

[결과 출력]
  12개 볼카운트별 추천 구종+코스 테이블
  W&B 대시보드에 전체 결과 업로드
```

### 핵심 코드 해설

```python
# 줄 41: 범용 모델 사용 여부 (True가 기본값이자 권장값)
USE_UNIVERSAL_MODEL = True

# 줄 160-167: 범용 모델 로드 (4개 파일을 읽어서 MLP 복원)
model_module = TransitionProbabilityModel.load_from_checkpoint(
    model_path=os.path.join(_root, "best_transition_model_universal.pth"),
    feature_columns_path=os.path.join(_root, "data", "feature_columns_universal.json"),
    target_classes_path=os.path.join(_root, "data", "target_classes_universal.json"),
    model_config_path=os.path.join(_root, "data", "model_config_universal.json"),
)

# 줄 193-203: MDP 최적 정책 계산 (모든 상황에 대한 정답표)
optimizer = MDPOptimizer(
    model=model_module,
    pitch_names=identified_pitch_names,   # ["Fastball", "Slider", ...]
    zones=[str(z) for z in range(1, 15)], # ["1", "2", ..., "14"]
    pitcher_clusters=all_pitcher_clusters, # ["0", "1", "2", "3"]
)
optimal_policy = optimizer.run_optimizer()

# 줄 226-240: DQN 학습 (시뮬레이터에서 30만 번 연습)
trainer = DQNTrainer(env=train_env, eval_env=eval_env)
trainer.build(total_timesteps=config.dqn_total_timesteps)  # 300,000
trainer.train()
trainer.evaluate()
```

---

## 전체 파이프라인 흐름도

```
[선행 작업 — 최초 1회]

  batter_clustering.py                pitcher_clustering.py
  2023 MLB 72만 건 투구               같은 데이터 재활용
        │                                    │
   좌/우타 분리                         투수별 15개 피처
   타자별 9개 피처                      실루엣 탐색 (K=4~8)
   UMAP → KMeans(K=8)                  UMAP → KMeans(K=4)
        │                                    │
        ▼                                    ▼
  batter_clusters_2023.csv          pitcher_clusters_2023.csv
  (타자 → 8개 유형)                  (투수 → 4개 유형)
        │                                    │
        └──────────┬─────────────────────────┘
                   ▼
         universal_model_trainer.py
         72만 건 + 타자군집 + 투수군집
         MLP [256→128→64] 학습 (epochs=20)
         3개 실험 → val_acc 58.1% (Exp1 선정)
                   │
                   ▼
         best_transition_model_universal.pth
         + feature_columns / target_classes / model_config .json


[메인 파이프라인 — uv run src/main.py]

  ┌─ main.py 실행 ─────────────────────────────────────────┐
  │                                                         │
  │  [1] data_loader.py                                     │
  │      투수 이름 입력 → Statcast 데이터 수집              │
  │                │                                        │
  │  [2] clustering.py                                      │
  │      UMAP+KMeans → 구종 자동 식별                      │
  │                │                                        │
  │  [3] model.py (load_from_checkpoint)                    │
  │      범용 모델 로드 (72만 건 학습 완료된 MLP)           │
  │                │                                        │
  │  [4] mdp_solver.py                                      │
  │      9,216개 상태 × 모든 행동 → 최적 정책표             │
  │      MLP.predict_proba()로 전이 확률 계산               │
  │                │                                        │
  │  [5] pitch_env.py + rl_trainer.py                       │
  │      가상 야구장에서 30만 번 투구 연습 (DQN)            │
  │      MLP.predict_proba()로 결과 시뮬레이션              │
  │                │                                        │
  │  [결과] 100이닝 평가 + 정책 테이블 출력                 │
  │         W&B 대시보드 업로드                             │
  └─────────────────────────────────────────────────────────┘
```

---

## 용어 사전 — 야구 비유로 한 줄 설명

### 머신러닝/딥러닝 용어

| 용어 | 뜻 | 야구 비유 |
|------|-----|----------|
| **MLP** (Multi-Layer Perceptron) | 여러 층으로 된 신경망 | 경험 많은 포수의 두뇌 — 상황을 보고 결과를 예측 |
| **epoch** (에폭) | 전체 훈련 데이터를 한 바퀴 도는 것 | 연습 1세트 — 72만 건 전부를 한 번 보는 것 |
| **batch_size** (배치 크기) | 한 번에 몇 개 데이터를 묶어서 학습하나 | 연습 때 한 번에 256구씩 분석 |
| **hidden_dims** (은닉층 차원) | 신경망 각 층의 뉴런 수 | 두뇌의 크기 — [256,128,64]는 3층짜리 큰 두뇌 |
| **val_acc** (검증 정확도) | 안 본 데이터로 테스트한 정답률 | 연습경기에서의 적중률 (실전 성적 예상치) |
| **val_loss** (검증 손실) | 예측이 정답에서 얼마나 벗어났는지 | 예측 오차 — 낮을수록 정확 |
| **EarlyStopping** (조기 종료) | 실력이 안 늘면 연습 중단 | "5경기 연속 성적 안 올랐으니 오늘 연습 끝" |
| **dropout** (드롭아웃) | 학습 때 일부 뉴런을 랜덤으로 끄기 | 한쪽 눈 가리고 연습 — 특정 패턴에 과의존 방지 |
| **one-hot encoding** (원-핫 인코딩) | 범주형 데이터를 0/1 벡터로 변환 | "Slider"를 [0,0,1,0,0,0,0,0,0]으로 표현 |
| **confusion matrix** (혼동 행렬) | 예측 vs 실제 결과를 표로 정리 | "스트라이크를 볼로 잘못 예측한 횟수" 같은 상세 성적표 |
| **softmax** | 숫자들을 확률(합=1)로 변환 | [2.1, 1.0, 0.5, 0.3] → [0.45, 0.15, 0.10, 0.30] |
| **CrossEntropyLoss** | 예측 확률과 정답의 차이를 측정하는 함수 | 포수가 "strike 70%"라고 예측했는데 실제 ball이면 큰 벌점 |
| **class-weighted loss** | 소수 클래스에 더 큰 벌점을 주는 기법 | 홈런 예측을 틀리면 볼 예측을 틀린 것보다 더 크게 감점 |
| **learning_rate** (학습률) | 한 번에 가중치를 얼마나 수정하나 | 연습 강도 — 너무 세면 폼이 무너지고, 너무 약하면 안 늠 |

### 강화학습 용어

| 용어 | 뜻 | 야구 비유 |
|------|-----|----------|
| **MDP** (Markov Decision Process) | 상태→행동→보상의 수학적 프레임워크 | 야구 규칙을 수학 공식으로 표현한 것 |
| **DQN** (Deep Q-Network) | 신경망으로 각 행동의 가치를 추정하는 RL 알고리즘 | AI 포수가 직접 던져보고 배우는 방식 |
| **state** (상태) | 현재 게임 상황 | 카운트 + 아웃 + 주자 + 타자유형 + 투수유형 |
| **action** (행동) | 에이전트의 선택 | 어떤 구종을 어느 코스에 던지나 |
| **reward** (보상) | 행동의 결과에 대한 점수 | 실점 위험 감소량 (RE24 변화) |
| **exploration** (탐색) | 랜덤 행동으로 새로운 경험 수집 | 평소 안 던지는 공도 던져보기 |
| **exploitation** (활용) | 현재까지 최선인 행동 선택 | 가장 잘 먹히는 공으로 승부 |
| **exploration_fraction** | 학습 중 탐색 비율 | "전체 연습의 30%는 모험, 이후 5%만 모험" |
| **gamma** (감마, 할인율) | 미래 보상의 중요도 | 0.99 = "다음 타석의 결과도 거의 동등하게 중요" |
| **experience replay** | 과거 경험을 저장하고 재사용 | 지난 경기 영상을 돌려보며 복습 |
| **episode** (에피소드) | 하나의 완결된 게임 단위 | 1이닝 (0아웃 → 3아웃까지) |

### 야구 통계 용어

| 용어 | 뜻 | 상세 설명 |
|------|-----|----------|
| **RE24** (Run Expectancy 24) | 24개 base-out 상태별 기대 실점 | "0아웃 만루"면 평균 2.28점, "2아웃 주자 없음"이면 0.10점 |
| **Statcast** | MLB 공식 투구 추적 시스템 | 야구장 카메라가 모든 공의 속도·회전·각도를 기록 |
| **whiff%** | 헛스윙 비율 | 스윙한 것 중 빗맞춘 비율 |
| **barrel%** | 강한 타구 비율 | 타구 속도+각도가 최적인 타구의 비율 |
| **zone%** | 스트라이크 존 투구 비율 | 전체 투구 중 존 안에 들어간 비율 |
| **pull%** | 당겨치기 비율 | 타자가 자기 쪽으로 당겨치는 비율 (좌타자→우측, 우타자→좌측) |
