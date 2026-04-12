"""
generate_physical_lookup.py — (pitcher_cluster, mapped_pitch_name)별 평균 물리 피처 lookup 테이블 생성

Task 12 Phase 2:
    현재 Exp5 모델은 release_speed_n, pfx_x_n, pfx_z_n 3개 수치 피처로 학습됐지만,
    PitchEnv._sample_outcome()과 MDPOptimizer.solve_mdp()에서 추론 시 이 3개가 0으로 채워져 있다.

    이 스크립트는 2023 MLB 전체 시즌 데이터에서 (pitcher_cluster, mapped_pitch_name)별
    평균 release_speed, pfx_x, pfx_z를 계산하고, 정규화된 값(release_speed_n, pfx_x_n, pfx_z_n)과
    함께 CSV로 저장한다.

산출물:
    data/physical_feature_lookup.csv
    컬럼: pitcher_cluster, mapped_pitch_name, release_speed_n, pfx_x_n, pfx_z_n

실행:
    uv run scripts/generate_physical_lookup.py
"""
import os
import sys

import pandas as pd
from pybaseball import statcast, cache

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.universal_model_trainer import PITCH_TYPE_MAP, _preprocess_raw

cache.enable()

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASE, '..'))
_DATA_DIR = os.path.join(_ROOT, 'data')

PITCHER_CLUSTERS_PATH = os.path.join(_DATA_DIR, 'pitcher_clusters_2023.csv')
OUTPUT_PATH = os.path.join(_DATA_DIR, 'physical_feature_lookup.csv')

START_DATE = "2023-03-30"
END_DATE = "2023-10-01"


def main():
    print("=" * 60)
    print("물리 피처 lookup 테이블 생성 (Task 12 Phase 2)")
    print("=" * 60)

    # 1. 데이터 수집 (pybaseball 캐시 활용)
    print(f"\n[1/4] Statcast 데이터 수집 ({START_DATE} ~ {END_DATE})")
    raw_df = statcast(start_dt=START_DATE, end_dt=END_DATE)
    print(f"  수집 완료: {len(raw_df):,}건")

    # 2. 전처리 (universal_model_trainer와 동일)
    print("\n[2/4] 전처리")
    df = _preprocess_raw(raw_df)
    del raw_df

    # 3. 투수 군집 매핑
    print("\n[3/4] 투수 군집 매핑")
    if os.path.exists(PITCHER_CLUSTERS_PATH):
        df_pc = pd.read_csv(PITCHER_CLUSTERS_PATH)
        df = df.merge(
            df_pc[['pitcher_id', 'cluster']].rename(columns={'pitcher_id': 'pitcher', 'cluster': 'pitcher_cluster'}),
            on='pitcher',
            how='left',
        )
        df['pitcher_cluster'] = df['pitcher_cluster'].fillna(0).astype(int)
        print(f"  투수 군집 매핑 완료: {df['pitcher_cluster'].nunique()}개 군집")
    else:
        print(f"  Warning: {PITCHER_CLUSTERS_PATH} 없음. 모든 투수를 군집 0으로 처리.")
        df['pitcher_cluster'] = 0

    # 4. 정규화 + 집계
    print("\n[4/4] (pitcher_cluster, mapped_pitch_name)별 평균 물리 피처 계산")
    df['release_speed_n'] = (df['release_speed'].astype(float) - 90.0) / 5.0
    df['pfx_x_n'] = df['pfx_x'].astype(float)
    df['pfx_z_n'] = df['pfx_z'].astype(float)

    lookup = df.groupby(['pitcher_cluster', 'mapped_pitch_name']).agg(
        release_speed_n=('release_speed_n', 'mean'),
        pfx_x_n=('pfx_x_n', 'mean'),
        pfx_z_n=('pfx_z_n', 'mean'),
        count=('release_speed_n', 'size'),
    ).reset_index()

    print(f"\n  lookup 테이블: {len(lookup)}행")
    print(lookup.to_string(index=False))

    # 저장
    os.makedirs(_DATA_DIR, exist_ok=True)
    lookup.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
