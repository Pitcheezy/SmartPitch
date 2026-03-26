"""
pitcher_clustering.py — MLB 투수 유형 군집화 파이프라인

역할:
    2023 시즌 500구 이상 투구한 MLB 투수(479명)를 15개 피처로 군집화합니다.
    결과는 data/pitcher_clusters_2023.csv로 저장되어 model.py, pitch_env.py,
    mdp_solver.py에서 pitcher_cluster 피처로 활용됩니다.

실행 방법 (최초 1회 또는 데이터 갱신 시):
    uv run src/pitcher_clustering.py

군집화 결과 (2023 시즌, K=4):
    군집 0 (157명): 파워 패스트볼/슬라이더   — 구속↑, FF44%, SL26%, Whiff26%
    군집 1 (102명): 핀세스/커맨드             — 구속↓, Zone%↑, 다양한 변화구
    군집 2 (103명): 무브먼트/싱커볼           — 구속↑이지만 FF15% (싱커 중심)
    군집 3 (117명): 멀티피치/아스날           — 구종수↑(4.4종), 다구종 선발

사용하는 15개 피처:
    물리량: release_speed, release_spin_rate, pfx_x(수평무브), pfx_z(수직무브),
            release_pos_x(팔 각도), release_pos_z(릴리스 높이)
    효과:   whiff_pct(헛스윙 유도율), zone_pct(존 투구 비율)
    구사율: ff_pct, si_pct, sl_pct, ch_pct, cu_pct, fc_pct
    다양성: n_pitch_types (5% 이상 사용 구종 수)

알고리즘:
    1. 투수별 피처 집계 (groupby + pivot_table)
    2. StandardScaler 표준화
    3. UMAP 2D 임베딩 (n_jobs=-1 병렬)
    4. K-Means K=4~8 실루엣 탐색 → K=4 선택됨 (0.4502)
    5. data/pitcher_clusters_2023.csv 저장 + W&B Artifact 업로드

W&B 로깅:
    - Plotly 인터랙티브 산점도 (hover: pitcher_id, 주요 지표)
    - pitcher_cluster_mapping Artifact

캐시:
    cache.enable()로 동일 기간 재요청 시 로컬 캐시 사용
    batter_clustering.py와 같은 날짜 범위면 캐시 재활용 (중복 다운로드 없음)
"""
import pandas as pd
import numpy as np
import wandb
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pybaseball import statcast, cache

cache.enable()  # 동일 기간 재요청 시 로컬 캐시 재활용 (대용량 반복 다운로드 방지)

class PitcherClustering:
    """
    MLB 투수들의 투구 데이터를 분석하여 소수의 '투수 유형 군집'으로 묶습니다.
    - 최소 500구 이상 투구한 투수만 대상
    - 피처: 구속/회전수/무브먼트/릴리스포인트 + 구종 구사율 + whiff%/zone%/구종다양성
    - 알고리즘: StandardScaler → UMAP(2D) → K-Means (K=4~8 실루엣 탐색)
    - 출력: data/pitcher_clusters_2023.csv (pitcher_id, cluster)
    """

    # 집계할 주요 구종 코드 (Statcast pitch_type 기준)
    PITCH_TYPES = ['FF', 'SI', 'SL', 'CH', 'CU', 'FC']

    # UMAP + K-Means에 사용할 피처 컬럼 목록
    FEATURE_COLS = [
        'release_speed', 'release_spin_rate',
        'pfx_x', 'pfx_z',
        'release_pos_x', 'release_pos_z',
        'whiff_pct', 'zone_pct',
        'ff_pct', 'si_pct', 'sl_pct', 'ch_pct', 'cu_pct', 'fc_pct',
        'n_pitch_types',
    ]

    def __init__(self, start_date: str, end_date: str, n_pitches_threshold: int = 500):
        """
        :param start_date: 수집 시작일 (예: '2023-03-30')
        :param end_date: 수집 종료일 (예: '2023-10-01')
        :param n_pitches_threshold: 투수별 최소 투구 수 (기본값 500)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.n_pitches_threshold = n_pitches_threshold

        self.raw_df = None
        self.pitcher_features = None
        self.best_k = None

    def fetch_statcast_data(self) -> pd.DataFrame:
        """
        지정된 기간의 전체 Statcast 데이터 다운로드.
        batter_clustering.py와 동일한 raw_df를 재활용 가능 (외부에서 set_raw_df() 사용).
        """
        print(f"[{self.start_date} ~ {self.end_date}] 전체 Statcast 데이터 수집 중...")
        df = statcast(start_dt=self.start_date, end_dt=self.end_date)
        if df.empty:
            raise ValueError("수집된 데이터가 없습니다. 날짜를 확인해주세요.")
        print(f"총 {len(df)}건의 투구 데이터 수집 완료.")
        self.raw_df = df
        return df

    def set_raw_df(self, df: pd.DataFrame):
        """외부(예: BatterClustering)에서 이미 받아온 데이터를 주입하여 중복 다운로드 방지."""
        self.raw_df = df

    def _extract_pitcher_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [내부 메서드] 투수별 15개 피처 집계.
        - 기본 물리량: 구속, 회전수, 무브먼트(pfx_x/z), 릴리스 포인트(x/z)
        - 효과 지표: whiff%, zone%
        - 구종 구사율: FF/SI/SL/CH/CU/FC 비율
        - 다양성: 5% 이상 사용 구종 수
        """
        import time
        start_time = time.time()
        print("투수별 피처 추출 중...")

        # 1. 최소 투구수 필터링
        pitch_counts = df['pitcher'].value_counts()
        valid_pitchers = pitch_counts[pitch_counts >= self.n_pitches_threshold].index
        df_valid = df[df['pitcher'].isin(valid_pitchers)].copy()
        print(f" -> 최소 {self.n_pitches_threshold}구 이상 투구한 투수 수: {len(valid_pitchers)}명")

        if len(valid_pitchers) == 0:
            return pd.DataFrame()

        # 2. 헛스윙 / 스윙 / 존 마스크 (투수 관점)
        swings = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip',
                  'foul_bunt', 'missed_bunt', 'hit_into_play']
        whiffs = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
        in_zone = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        df_valid['cnt_swing'] = df_valid['description'].isin(swings).fillna(False).astype(int)
        df_valid['cnt_whiff'] = df_valid['description'].isin(whiffs).fillna(False).astype(int)
        df_valid['cnt_in_zone'] = df_valid['zone'].isin(in_zone).fillna(False).astype(int)

        # 3. 투수별 기본 집계
        agg_dict = {
            'release_speed': 'mean',
            'release_spin_rate': 'mean',
            'pfx_x': 'mean',
            'pfx_z': 'mean',
            'release_pos_x': 'mean',
            'release_pos_z': 'mean',
            'cnt_swing': 'sum',
            'cnt_whiff': 'sum',
            'cnt_in_zone': 'sum',
        }
        grouped = df_valid.groupby('pitcher').agg(agg_dict)
        total_pitches = df_valid.groupby('pitcher').size().rename('total_pitches')
        grouped = grouped.join(total_pitches)

        # 4. 비율 계산
        def safe_div(num, den):
            return np.where(den > 0, num / den, 0.0)

        grouped['whiff_pct'] = safe_div(grouped['cnt_whiff'].values, grouped['cnt_swing'].values)
        grouped['zone_pct'] = safe_div(grouped['cnt_in_zone'].values, grouped['total_pitches'].values)

        # 5. 구종 구사율 (Pivot 방식 — 결측 구종은 0으로 채움)
        pitch_type_counts = df_valid.groupby(['pitcher', 'pitch_type']).size().unstack(fill_value=0)
        pitch_totals = pitch_type_counts.sum(axis=1)
        pitch_type_pcts = pitch_type_counts.div(pitch_totals, axis=0)

        for pt_code in self.PITCH_TYPES:
            col_name = f'{pt_code.lower()}_pct'
            if pt_code in pitch_type_pcts.columns:
                grouped[col_name] = pitch_type_pcts[pt_code].reindex(grouped.index).fillna(0)
            else:
                grouped[col_name] = 0.0

        # 6. 구종 다양성: 5% 이상 구사하는 구종 수
        n_types = (pitch_type_pcts >= 0.05).sum(axis=1).reindex(grouped.index).fillna(0)
        grouped['n_pitch_types'] = n_types

        # 최종 정리
        features_df = grouped[self.FEATURE_COLS].copy().fillna(0).reset_index()
        features_df = features_df.rename(columns={'pitcher': 'pitcher_id'})

        elapsed = time.time() - start_time
        print(f" -> 투수 피처 추출 완료 (소요시간: {elapsed:.2f}초, {len(features_df)}명)")

        self.pitcher_features = features_df
        return features_df

    def _apply_umap_kmeans(self) -> int:
        """
        [내부 메서드] StandardScaler → UMAP(2D) → K-Means 수행.
        K는 4~8 범위에서 실루엣 점수가 가장 높은 값으로 자동 선택.
        """
        print("투수 데이터 UMAP 및 K-Means 군집화 진행 중...")
        features_df = self.pitcher_features

        if features_df is None or len(features_df) < 5:
            print(" -> 군집화하기에 데이터가 너무 적습니다.")
            return -1

        # 1. 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df[self.FEATURE_COLS])

        # 2. UMAP 2D 임베딩 (n_jobs=-1 병렬 처리)
        reducer = umap.UMAP(n_components=2, n_jobs=-1)
        embedding = reducer.fit_transform(X_scaled)
        print(" -> UMAP 차원 축소 완료")

        features_df['umap_1'] = embedding[:, 0]
        features_df['umap_2'] = embedding[:, 1]

        # 3. K=4~8 실루엣 탐색 (타자와 달리 투수 유형은 더 적은 편)
        print(" -> K-Means 실루엣 스코어 탐색 중 (K=4~8)...")
        best_k = 4
        best_score = -1
        for k in range(4, 9):
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = km.fit_predict(embedding)
            score = silhouette_score(embedding, labels)
            print(f"    K={k} → silhouette: {score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k

        print(f" -> 최적 K={best_k} 선택 (Score: {best_score:.4f})")

        # 4. 최적 K로 최종 레이블 부여
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        features_df['cluster'] = km_final.fit_predict(embedding)

        self.pitcher_features = features_df
        self.best_k = best_k
        return best_k

    def log_interactive_scatter_to_wandb(self):
        """
        Plotly 인터랙티브 2D 산점도를 W&B 대시보드에 로깅.
        hover data에 pitcher_id 및 주요 지표 포함.
        """
        features_df = self.pitcher_features
        if features_df is None or 'cluster' not in features_df.columns:
            return

        print(f"투수 인터랙티브 산점도 시각화 및 W&B 로깅 중 (K={self.best_k})...")
        features_df['Cluster_Label'] = "Type_" + features_df['cluster'].astype(str)

        fig = px.scatter(
            features_df,
            x='umap_1', y='umap_2',
            color='Cluster_Label',
            hover_data=['pitcher_id', 'release_speed', 'whiff_pct', 'ff_pct', 'sl_pct', 'n_pitch_types', 'zone_pct'],
            title=f"UMAP Projection of Pitcher Types (K={self.best_k})",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(autosize=False, width=800, height=600)

        if wandb.run:
            wandb.log({"Pitcher_Clustering": fig})
            print(" -> W&B 대시보드에 투수 군집 산점도 로깅 완료!")
        else:
            print(" -> W&B run 비활성화. HTML 파일로 저장합니다.")
            fig.write_html("pitcher_clustering_temp.html")

    def run_clustering_pipeline(self) -> pd.DataFrame:
        """
        전체 파이프라인 실행:
        1. Statcast 데이터 수집 (또는 set_raw_df()로 주입)
        2. 투수별 피처 추출
        3. UMAP + K-Means 군집화
        4. W&B 산점도 로깅
        5. data/pitcher_clusters_2023.csv 저장 및 W&B Artifact 업로드
        반환: pitcher_features DataFrame (pitcher_id, cluster, umap_1, umap_2, ...)
        """
        import os

        if self.raw_df is None:
            self.fetch_statcast_data()

        self._extract_pitcher_features(self.raw_df)

        if self.pitcher_features is not None and not self.pitcher_features.empty:
            self._apply_umap_kmeans()
            self.log_interactive_scatter_to_wandb()

            # CSV 저장
            export_df = self.pitcher_features[['pitcher_id', 'cluster']].copy()
            csv_path = "data/pitcher_clusters_2023.csv"
            export_df.to_csv(csv_path, index=False)
            print(f" -> 로컬 CSV 백업 완료: {csv_path} (총 {len(export_df)}명, K={self.best_k})")

            # W&B Artifact 영구 보관
            if wandb.run:
                artifact = wandb.Artifact(name="pitcher_cluster_mapping", type="dataset")
                artifact.add_file(csv_path)
                wandb.log_artifact(artifact)
                print(" -> WandB Artifact에 `pitcher_cluster_mapping` CSV 업로드 완료!")

        return self.pitcher_features


if __name__ == "__main__":
    wandb.init(project="SmartPitch-Portfolio", name="Pitcher_Clustering_2023_Full")

    pc = PitcherClustering(start_date="2023-03-30", end_date="2023-10-01", n_pitches_threshold=500)
    results = pc.run_clustering_pipeline()

    if results is not None and not results.empty:
        print(f"\n=== 투수 군집(K={pc.best_k}) 유형별 평균 지표 ===")
        summary_metrics = ['release_speed', 'whiff_pct', 'ff_pct', 'sl_pct', 'n_pitch_types', 'zone_pct']
        cluster_summary = results.groupby('cluster')[summary_metrics].mean().round(3)
        print(cluster_summary)
        print("=" * 50)

    wandb.finish()
