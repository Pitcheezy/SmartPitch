"""
Statcast 데이터 전처리

컬럼 선택, 타입 변환, 파생 변수(description_group, events_group 등) 생성.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# Statcast 원본에서 사용할 컬럼 (나머지는 제거)
SELECTED_COLS = [
    'pitch_type', 'game_date', 'release_speed', 'release_pos_x', 'release_pos_z',
    'batter', 'pitcher', 'events', 'description', 'zone', 'stand', 'p_throws',
    'bb_type', 'balls', 'strikes', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot',
    'launch_speed', 'launch_angle', 'release_spin_rate', 'release_extension',
    'pitch_number', 'bat_score_diff', 'arm_angle'
]


# description(투구 결과 설명) → 단순화된 그룹
DESC_TO_GROUP = {
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
    "hit_into_play": "inplay",
    "hit_into_play_score": "inplay",
    "hit_into_play_no_out": "inplay",
}

# events(경기 결과) → 단순화된 그룹
EVENT_TO_GROUP = {
    "single": "hit",
    "double": "hit",
    "triple": "hit",
    "home_run": "hit",
    "walk": "walk",
    "intent_walk": "walk",
    "hit_by_pitch": "walk",
    "strikeout": "out",
    "strikeout_double_play": "out",
    "field_out": "out",
    "force_out": "out",
    "double_play": "out",
    "grounded_into_double_play": "out",
    "field_error": "reached",
}


# UMAP·분석용 수치형 컬럼
NUM_COLS = [
    "release_speed","release_spin_rate","pfx_x","pfx_z",
    "release_pos_x","release_pos_z","release_extension","arm_angle",
    "plate_x","plate_z","launch_speed","launch_angle","bat_score_diff",
]


def preprocess_statcast(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Statcast 원시 데이터 전처리

    - SELECTED_COLS 컬럼만 선택
    - 타입 변환 (날짜, 숫자, 범주)
    - description_group, events_group 파생 변수 생성
    - base_state, count 파생 변수 생성

    Args:
        df_raw: pybaseball statcast 원본 DataFrame

    Returns:
        전처리된 DataFrame (pitch_clean)
    """
    # 컬럼 선택
    missing = [c for c in SELECTED_COLS if c not in df_raw.columns]
    if missing:
        raise KeyError(f"Missing columns in raw data: {missing}")

    df = df_raw[SELECTED_COLS].copy()

    # 타입 정리
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce")
    df["batter"] = pd.to_numeric(df["batter"], errors="coerce")

    df["balls"] = pd.to_numeric(df["balls"], errors="coerce").fillna(0).astype(int)
    df["strikes"] = pd.to_numeric(df["strikes"], errors="coerce").fillna(0).astype(int)

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 인플레이 아닐 때 결측치는 0.0으로 덮어쓰지 않고 NaN 유지 (타자 프로필 산출 시 평균 왜곡 방지)
    # df["launch_speed"] = df["launch_speed"].fillna(0.0)
    # df["launch_angle"] = df["launch_angle"].fillna(0.0)

    # 주자 있으면 1, 없으면 0
    df["on_1b"] = df["on_1b"].notna().astype(int)
    df["on_2b"] = df["on_2b"].notna().astype(int)
    df["on_3b"] = df["on_3b"].notna().astype(int)

    # 그룹 컬럼
    df["description_group"] = df["description"].map(DESC_TO_GROUP).fillna("other")
    df["events_group"] = df["events"].map(EVENT_TO_GROUP).fillna("other")

    df["description_group"] = df["description_group"].astype("object")
    df["events_group"] = df["events_group"].astype("object")

    # 파생
    df["base_state"] = df["on_1b"] + df["on_2b"] * 2 + df["on_3b"] * 4
    df["count"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

    # 필수키 drop
    df = df.dropna(subset=["pitcher", "batter"]).copy()
    df["pitcher"] = df["pitcher"].astype(int)
    df["batter"] = df["batter"].astype(int)

    # UMAP 입력용 결측 제거는 embedding 단계에서 다시 강하게 처리
    return df
