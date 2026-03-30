"""
Decision Dataset 생성 파이프라인
요청하신 구조에 따라 src/dataset_builder.py 하나에 모든 생성 및 저장 논리를 통합하였습니다.
투수별로 분리된 Decision Dataset과 메타데이터(상태 공간 등)를 추출합니다.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

# 파라미터 기본값 설정
TOP_PITCHERS_LIMIT = 400
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_PITCHES_IN_ARSENAL = 30
MIN_ZONE_OCCURRENCE = 100


def load_data(parquet_path: str | Path) -> pd.DataFrame:
    """원본 데이터 로드"""
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    return df


def select_top_pitchers(df: pd.DataFrame, top_n: int = TOP_PITCHERS_LIMIT) -> list[int]:
    """투구 수 기준 상위 투수 선택"""
    pitcher_counts = df.groupby('pitcher').size().sort_values(ascending=False)
    selected = pitcher_counts.head(top_n).index.astype(int).tolist()
    return selected


def split_by_game(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """game_pk 단위로 Train/Val/Test 분할"""
    # game_date가 있다면 시간순 정렬 우선
    if 'game_date' in df.columns:
        games = df[['game_date', 'game_pk']].drop_duplicates().sort_values('game_date')
    else:
        games = df[['game_pk']].drop_duplicates().sort_values('game_pk')
        
    unique_games = games['game_pk'].unique()
    n_total = len(unique_games)
    
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_games = unique_games[:n_train]
    val_games = unique_games[n_train:n_train+n_val]
    test_games = unique_games[n_train+n_val:]
    
    train_df = df[df['game_pk'].isin(train_games)].copy()
    val_df = df[df['game_pk'].isin(val_games)].copy()
    test_df = df[df['game_pk'].isin(test_games)].copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    return train_df, val_df, test_df


def _apply_count_state(df: pd.DataFrame) -> pd.DataFrame:
    """상태(count_state) 문자열 및 결측치 방어 공통 함수"""
    for c in ['balls', 'strikes', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b']:
        df[c] = df[c].fillna(0).astype(int)
        
    df['count_state'] = (
        df['balls'].astype(str) + "-" + df['strikes'].astype(str) + "_" +
        df['outs_when_up'].astype(str) + "_" +
        df['on_1b'].astype(str) + df['on_2b'].astype(str) + df['on_3b'].astype(str)
    )
    return df


def build_valid_states(df_train: pd.DataFrame) -> pd.DataFrame:
    """Train 기준 valid_states 생성"""
    df_train = _apply_count_state(df_train.copy())
    state_counts = df_train.groupby('count_state').size().reset_index(name='count')
    state_counts = state_counts.sort_values(by=['count_state']).reset_index(drop=True)
    state_counts['state_id'] = range(len(state_counts))
    return state_counts[['count_state', 'state_id', 'count']]


def build_pitcher_arsenal(df_train: pd.DataFrame, min_pitches: int = MIN_PITCHES_IN_ARSENAL) -> pd.DataFrame:
    """Train 기준 투수별 유효 구종 생성 (희귀 구종 제거)"""
    pitch_col = 'pitch_cluster_id' if 'pitch_cluster_id' in df_train.columns else 'pitch_type'
    
    arsenal = df_train.groupby(['pitcher', pitch_col]).size().reset_index(name='pitch_count')
    arsenal = arsenal[arsenal['pitch_count'] >= min_pitches].copy()
    arsenal = arsenal.rename(columns={pitch_col: 'mapped_pitch_name'})
    return arsenal


def build_valid_zones(df_train: pd.DataFrame, min_count: int = MIN_ZONE_OCCURRENCE) -> pd.DataFrame:
    """Train 기준 유효 존 생성 (희귀 존 제거)"""
    df_train = df_train.dropna(subset=['zone']).copy()
    df_train['zone'] = df_train['zone'].astype(int)
    
    zone_counts = df_train.groupby('zone').size().reset_index(name='count')
    valid_zones = zone_counts[zone_counts['count'] >= min_count].copy()
    return valid_zones[['zone', 'count']].sort_values('zone').reset_index(drop=True)


def build_action_map(pitcher_arsenal: pd.DataFrame, valid_zones: pd.DataFrame) -> pd.DataFrame:
    """모든 Split에서 일관된 action_id를 보장하기 위해 구종 x Zone의 행동 매핑 테이블을 생성합니다."""
    arsenal_names = pitcher_arsenal[['mapped_pitch_name']].drop_duplicates()
    zones = valid_zones[['zone']].drop_duplicates()
    
    # 교차 결합(Cartesian Product)으로 가능한 모든 행동 정의
    action_map = arsenal_names.merge(zones, how='cross')
    action_map = action_map.sort_values(['mapped_pitch_name', 'zone']).reset_index(drop=True)
    action_map['action_id'] = action_map.index
    return action_map


def map_outcome_class(df: pd.DataFrame) -> pd.DataFrame:
    """description 기반으로 outcome을 strike, ball, foul, in_play로 매핑합니다."""
    mapping = {
        "called_strike": "strike",
        "swinging_strike": "strike",
        "swinging_strike_blocked": "strike",
        "missed_bunt": "strike",
        "foul": "foul",
        "foul_tip": "foul",
        "foul_bunt": "foul",
        "ball": "ball",
        "blocked_ball": "ball",
        "pitchout": "ball",
        "intent_ball": "ball",
        "hit_into_play": "in_play",
        "hit_into_play_score": "in_play",
        "hit_into_play_no_out": "in_play",
    }
    df['outcome_class'] = df['description'].map(mapping)
    
    # outcome_id 부여 (알파벳 순 정렬)
    outcome_to_id = {"ball": 0, "foul": 1, "in_play": 2, "strike": 3}
    df['outcome_id'] = df['outcome_class'].map(outcome_to_id)
    return df


def build_transition_dataset(
    df: pd.DataFrame, 
    split_name: str,
    valid_states: pd.DataFrame, 
    pitcher_arsenal: pd.DataFrame, 
    valid_zones: pd.DataFrame,
    action_map: pd.DataFrame
) -> pd.DataFrame:
    """슈퍼바이즈드 러닝용 Transition Dataset을 생성합니다."""
    df = df.copy()
    initial_len = len(df)
    
    # 1. 상태 및 행동 파싱
    df = _apply_count_state(df)
    pitch_col = 'pitch_cluster_id' if 'pitch_cluster_id' in df.columns else 'pitch_type'
    df['mapped_pitch_name'] = df[pitch_col]
    df['zone'] = pd.to_numeric(df['zone'], errors='coerce')
    
    # 2. 결과(outcome) 매핑
    df = map_outcome_class(df)
    
    # 3. 결측치 데이터 및 무효 조합 드랍
    df = df.dropna(subset=['outcome_class', 'zone', 'mapped_pitch_name', 'count_state'])
    df['zone'] = df['zone'].astype(int)
    
    # Train에서 정의된 Valid 범위로 이너 조인 (Data Leakage 방지 및 일관성)
    df = df.merge(valid_states[['count_state']], on='count_state', how='inner')
    df = df.merge(pitcher_arsenal[['pitcher', 'mapped_pitch_name']], on=['pitcher', 'mapped_pitch_name'], how='inner')
    df = df.merge(valid_zones[['zone']], on='zone', how='inner')
    
    # 4. 행동 ID 일관성 부여
    df = df.merge(action_map[['mapped_pitch_name', 'zone', 'action_id']], on=['mapped_pitch_name', 'zone'], how='inner')
    
    df['split'] = split_name
    
    # 로그 출력
    dropped_len = initial_len - len(df)
    print(f"[{split_name.upper()}_TRANSITION] 애매한 outcome 및 무효 상태로 인해 제외된 행 수: {dropped_len:,} / {initial_len:,}")
    
    dist = df['outcome_class'].value_counts().to_dict()
    print(f"[{split_name.upper()}_TRANSITION] 정상 생성 Row 수: {len(df):,}, Outcome 분포: {dist}")
    
    # 유지할 컬럼
    keep_cols = [
        'game_pk', 'game_date', 'pitcher', 'batter', 'pitch_number',
        'balls', 'strikes', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b', 'count_state',
        'mapped_pitch_name', 'zone', 'action_id',
        'description', 'events', 'bb_type', 'outcome_class', 'outcome_id', 'split'
    ]
    return df[[c for c in keep_cols if c in df.columns]]


def create_decision_dataset(
    df: pd.DataFrame, 
    valid_states: pd.DataFrame, 
    pitcher_arsenal: pd.DataFrame, 
    valid_zones: pd.DataFrame,
    action_map: pd.DataFrame
) -> pd.DataFrame:
    """투수별 Decision Dataset 생성에 앞서 필터링, 액션 매핑 적용"""
    df = _apply_count_state(df.copy())
    
    if 'pitch_cluster_id' in df.columns:
        df['mapped_pitch_name'] = df['pitch_cluster_id']
    else:
        df['mapped_pitch_name'] = df['pitch_type']
        
    df['zone'] = pd.to_numeric(df['zone'], errors='coerce')
    
    # Inner join을 통해 희귀 상태/구종/존 등을 필터링
    df = df.merge(valid_states[['count_state', 'state_id']], on='count_state', how='inner')
    df = df.merge(pitcher_arsenal[['pitcher', 'mapped_pitch_name']], on=['pitcher', 'mapped_pitch_name'], how='inner')
    df = df.dropna(subset=['zone']).copy()
    df['zone'] = df['zone'].astype(int)
    df = df.merge(valid_zones[['zone']], on='zone', how='inner')
    
    # 행동(action) 공간 일관적 매핑 적용
    df = df.merge(action_map[['mapped_pitch_name', 'zone', 'action_id']], on=['mapped_pitch_name', 'zone'], how='inner')
    df['is_observed_action'] = 1
    
    return df


def main_pipeline(raw_parquet_path: str, base_dir: str):
    base_path = Path(base_dir)
    meta_dir = base_path / "metadata"
    data_dir = base_path / "datasets"
    decision_dir = data_dir / "decision_dataset"
    
    meta_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 데이터 로드
    df_all = load_data(raw_parquet_path)
    if 'game_pk' not in df_all.columns:
        raise ValueError("Error: 'game_pk' 컬럼이 없습니다. (분할 기준)")
        
    # 2. 상위 투수 선택
    print("Selecting top pitchers...")
    top_pitchers = select_top_pitchers(df_all)
    df_top = df_all[df_all['pitcher'].isin(top_pitchers)].copy()
    
    # 3. Train / Val / Test 분리
    print("Splitting dataset by game_pk...")
    train_df, val_df, test_df = split_by_game(df_top, TRAIN_RATIO, VAL_RATIO)
    
    # 4. Train 데이터 기준 메타데이터 생성
    print("Building valid states, arsenal, and zones from train data...")
    valid_states = build_valid_states(train_df)
    pitcher_arsenal = build_pitcher_arsenal(train_df)
    valid_zones = build_valid_zones(train_df)
    action_map = build_action_map(pitcher_arsenal, valid_zones)
    
    # 5. 메타데이터 저장
    print("Saving metadata...")
    pd.DataFrame({'pitcher': top_pitchers}).to_parquet(meta_dir / "selected_pitchers.parquet", index=False)
    valid_states.to_parquet(meta_dir / "valid_states.parquet", index=False)
    pitcher_arsenal.to_parquet(meta_dir / "pitcher_arsenal.parquet", index=False)
    valid_zones.to_parquet(meta_dir / "valid_zones.parquet", index=False)
    action_map.to_parquet(meta_dir / "action_map.parquet", index=False)
    
    # [추가] 5.5 Transition Dataset 생성 및 저장
    print("Creating and saving Transition Datasets...")
    trans_train = build_transition_dataset(train_df, 'train', valid_states, pitcher_arsenal, valid_zones, action_map)
    trans_val = build_transition_dataset(val_df, 'val', valid_states, pitcher_arsenal, valid_zones, action_map)
    trans_test = build_transition_dataset(test_df, 'test', valid_states, pitcher_arsenal, valid_zones, action_map)
    
    trans_train.to_parquet(data_dir / "train_transition.parquet", index=False)
    trans_val.to_parquet(data_dir / "val_transition.parquet", index=False)
    trans_test.to_parquet(data_dir / "test_transition.parquet", index=False)
    
    # 6. 세트별 Decision Dataset 생성
    print("Creating decision datasets...")
    dec_train = create_decision_dataset(train_df, valid_states, pitcher_arsenal, valid_zones, action_map)
    dec_val = create_decision_dataset(val_df, valid_states, pitcher_arsenal, valid_zones, action_map)
    dec_test = create_decision_dataset(test_df, valid_states, pitcher_arsenal, valid_zones, action_map)
    
    # 7. 전체 분할 Parquet 저장
    print("Saving train, val, test splits...")
    dec_train.to_parquet(data_dir / "train.parquet", index=False)
    dec_val.to_parquet(data_dir / "val.parquet", index=False)
    dec_test.to_parquet(data_dir / "test.parquet", index=False)
    
    # 8. 투수별 통합 데이터셋(Pitcher Decision Dataset) 생성 및 저장
    print("Saving individual pitcher datasets...")
    df_decision_all = pd.concat([dec_train, dec_val, dec_test], ignore_index=True)
    pitcher_groups = df_decision_all.groupby('pitcher')
    for p_id, p_df in pitcher_groups:
        p_path = decision_dir / f"pitcher_{int(p_id)}.parquet"
        p_df.to_parquet(p_path, index=False)
        
    print("✅ Pipeline Completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("⚾ SmartPitch: 전채 투수 대상 Decision / Transition Dataset 생성 ⚾")
    print("=" * 60)
    
    start_date = input("▶ 원본 데이터 시작일을 입력하세요 (예: 2019-03-28): ").strip()
    end_date = input("▶ 원본 데이터 종료일을 입력하세요 (예: 2019-09-29): ").strip()
    
    if not start_date or not end_date:
        print("시작일과 종료일을 올바르게 입력해야 합니다.")
    else:
        current_dir = Path(__file__).resolve().parent.parent
        # 전처리 파이프라인(1~2단계)을 거쳐 생성되는 클러스터 데이터 기반으로 탐색
        target_file = current_dir / "data" / "processed" / f"pitch_umap_cluster_{start_date}_to_{end_date}.parquet"
        out_dir = current_dir / "data"
        
        if target_file.exists():
            main_pipeline(str(target_file), str(out_dir))
        else:
            print(f"\n❌ 파일을 찾을 수 없습니다: {target_file}")
            print(f"먼저 `uv run python main.py`를 통해 해당 기간({start_date} ~ {end_date})의 데이터에 대한 기초 파이프라인(전처리 및 클러스터링)을 구동해 주세요!")
