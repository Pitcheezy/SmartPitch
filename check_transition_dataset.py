import pandas as pd
from pathlib import Path
import re

data_dir = Path("c:/Users/Dalab-server/Documents/kangmin/Pitcheezy/SmartPitch/data/datasets")
meta_dir = Path("c:/Users/Dalab-server/Documents/kangmin/Pitcheezy/SmartPitch/data/metadata")

splits = ['train', 'val', 'test']
dfs = {}

out = []
def log(msg):
    out.append(str(msg))
    print(msg)

# 1. 파일 존재 여부 확인
log("=== 1. 존재 확인 ===")
for s in splits:
    p = data_dir / f"{s}_transition.parquet"
    if p.exists():
        dfs[s] = pd.read_parquet(p)
        log(f"[O] {p.name} 존재함")
    else:
        log(f"[X] {p.name} 없음")

if len(dfs) != 3:
    log("파일이 부족해 검증 중단")
    exit()

# 2. Row 수 및 컬럼 목록
log("\n=== 2. Row 수 및 컬럼 목록 ===")
for s in splits:
    log(f"[{s.upper()}] Rows: {len(dfs[s]):,}, Columns: {len(dfs[s].columns)}")
log(f"Columns list: {list(dfs['train'].columns)}")

# 3. 필수 컬럼 확인
required_cols = [
    'game_pk', 'game_date', 'pitcher', 'batter', 'pitch_number',
    'balls', 'strikes', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b', 'count_state',
    'mapped_pitch_name', 'zone', 'action_id',
    'description', 'events', 'bb_type', 'outcome_class', 'outcome_id', 'split'
]
log("\n=== 3. 필수 컬럼 확인 ===")
missing = [c for c in required_cols if c not in dfs['train'].columns]
if not missing:
    log("[O] 모든 필수 컬럼 존재함")
else:
    log(f"[X] 누락된 컬럼: {missing}")

# 4 & 5. outcome_class 분포 비율 및 4개 클래스 확인
log("\n=== 4 & 5. outcome_class 분포 비율 ===")
valid_classes = {'strike', 'ball', 'foul', 'in_play'}
for s in splits:
    vc = dfs[s]['outcome_class'].value_counts()
    invalid = set(vc.index) - valid_classes
    if invalid:
        log(f"[X] [{s.upper()}] 비정상 클래스 발견: {invalid}")
    
    total = sum(vc)
    dist_str = ", ".join([f"{k}: {v:,} ({v/total*100:.1f}%)" for k, v in vc.items()])
    log(f"[{s.upper()}] {dist_str}")

# 6. count_state 형식 일관성 확인
log("\n=== 6. count_state 형식 확인 ===")
sample_states = dfs['train']['count_state'].sample(5, random_state=42).tolist()
valid_format = all(re.match(r"^\d-\d_\d_[01][01][01]$", st) for st in sample_states)
log(f"Sample states: {sample_states}")
if valid_format:
    log("[O] 모든 샘플이 올바른 포맷(balls-strikes_outs_runner123)입니다.")
else:
    log("[X] 일부 샘플이 잘못된 포맷입니다.")

# 7. action_id 일관성 확인 (Decision Dataset과 비교)
log("\n=== 7. action_id 일관성 점검 ===")
dec_train_path = data_dir / "train.parquet"
if dec_train_path.exists():
    dec_df = pd.read_parquet(dec_train_path)
    t_map = dfs['train'][['mapped_pitch_name', 'zone', 'action_id']].drop_duplicates().dropna()
    d_map = dec_df[['mapped_pitch_name', 'zone', 'action_id']].drop_duplicates().dropna()
    merged = t_map.merge(d_map, on=['mapped_pitch_name', 'zone', 'action_id'], how='inner')
    if len(merged) == len(t_map) and len(t_map) > 0:
        log("[O] Transition Dataset과 Decision Dataset의 action_id 룰이 완벽히 동일함")
    else:
        log("[X] 매핑 일관성 오류 발견!")

# 8. selected_pitchers 제외된 투수 포함 여부
log("\n=== 8. selected_pitchers 외 투수 포함 여부 ===")
sel_pitchers = pd.read_parquet(meta_dir / 'selected_pitchers.parquet')['pitcher'].unique()
for s in splits:
    outliers = dfs[s][~dfs[s]['pitcher'].isin(sel_pitchers)]
    if len(outliers) == 0:
        log(f"[O] [{s.upper()}] 모든 투수가 메타데이터 내에 존재함")
    else:
        log(f"[X] [{s.upper()}] 범위 외 투수 {len(outliers)}명 발견!")

# 9. 결측치 확인
log("\n=== 9. 결측치 확인 ===")
for s in splits:
    nulls = dfs[s][required_cols].isnull().sum()
    null_cols = nulls[nulls > 0]
    expected_null_cols = ['events', 'bb_type']
    unexpected_nulls = null_cols[~null_cols.index.isin(expected_null_cols)]
    if len(unexpected_nulls) == 0:
        log(f"[O] [{s.upper()}] 핵심 컬럼 결측치 없음")
    else:
        log(f"[!] [{s.upper()}] 일부 핵심 컬럼 결측치 존재:\n{unexpected_nulls.to_dict()}")

# 11. 샘플 점검
log("\n=== 11. 샘플 데이터 (10 rows) 점검 ===")
sample_cols = ['count_state', 'mapped_pitch_name', 'zone', 'action_id', 'description', 'outcome_class']
sample_df = dfs['train'][sample_cols].sample(10, random_state=100)
log(sample_df.to_string())

with open("report_chk.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
