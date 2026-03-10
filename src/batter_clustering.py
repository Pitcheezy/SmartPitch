import pandas as pd
import numpy as np
import umap
import plotly.express as px
import plotly.graph_objects as go
import wandb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pybaseball import statcast

class BatterClustering:
    """
    타자들의 방대한 투구 결과 데이터를 분석하여 소수의 '타격 어프로치 군집'으로 묶습니다.
    이때 좌타(LHB), 우타(RHB)를 완벽하게 분리하여 각각 독립적으로 모델링합니다.
    (최소 500구 이상 상대 필터링)
    """

    def __init__(self, start_date: str, end_date: str, n_pitches_threshold: int = 500):
        """
        :param start_date: 수집 시작일 (예: '2023-03-30')
        :param end_date: 수집 종료일 (예: '2023-10-01')
        :param n_pitches_threshold: 타자별 최소 상대 투구 수 (기본값 500)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.n_pitches_threshold = n_pitches_threshold
        
        self.raw_df = None
        
        # 좌타/우타 분리 데이터프레임 보관
        self.batter_features = {'L': None, 'R': None}
        
    def fetch_statcast_data(self) -> pd.DataFrame:
        """
        지정된 기간 동안의 전체 statcast 플레이 레벨 데이터를 다운로드
        """
        print(f"[{self.start_date} ~ {self.end_date}] 전체 Statcast 데이터 수집 중...")
        # statcast()는 전체 데이터를 가져오므로 메모리 주의
        df = statcast(start_dt=self.start_date, end_dt=self.end_date)
        
        if df.empty:
            raise ValueError("수집된 데이터가 없습니다. 날짜를 확인해주세요.")
            
        print(f"총 {len(df)}건의 투구 데이터 수집 완료.")
        self.raw_df = df
        return df

    def _extract_batter_features(self, df: pd.DataFrame, stand: str) -> pd.DataFrame:
        """
        [내부 메서드] 특정 타석(stand: 'L' or 'R')에 대한 타자별 지표 추출
        - Whiff%, Z-Contact%, O-Swing%, O-Contact%, Launch Angle, Barrel%, Pull%, High FF Whiff%
        (Pandas Vectorization & Groupby 기반 최적화 10초 컷)
        """
        import time
        start_time = time.time()
        print(f"타석 방향 [{stand}] - 벡터화된 특성 추출 중...")
        
        # 해당 타석 방향 데이터 필터링 (결측치 제거)
        df_side = df[df['stand'] == stand].copy()
        
        # 타자 이름용 컬럼 (player_name이 보통 "Last, First" 형태임)
        if 'player_name' not in df_side.columns:
            df_side['player_name'] = df_side['batter'].astype(str)

        # 1. 500구 이상 필터링
        pitch_counts = df_side['batter'].value_counts()
        valid_batters = pitch_counts[pitch_counts >= self.n_pitches_threshold].index
        df_side = df_side[df_side['batter'].isin(valid_batters)]
        print(f" -> 최소 {self.n_pitches_threshold}구 이상 상대한 [{stand}] 타자 수: {len(valid_batters)}명")
        
        if len(valid_batters) == 0:
            return pd.DataFrame()
            
        # 2. 기초 판정 마스킹 (Vectorized Boolean Flags)
        swings = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt', 
                  'missed_bunt', 'hit_into_play']
        whiffs = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
        contacts = ['foul', 'foul_tip', 'foul_bunt', 'hit_into_play']
        in_zone = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        high_zones = [1, 2, 3, 11, 12]
        
        # 기본 마스크 (Bool -> Int)
        is_swing = df_side['description'].isin(swings).fillna(False).astype(int)
        is_whiff = df_side['description'].isin(whiffs).fillna(False).astype(int)
        is_contact = df_side['description'].isin(contacts).fillna(False).astype(int)
        is_in_zone = df_side['zone'].isin(in_zone).fillna(False).astype(int)
        is_out_zone = (~df_side['zone'].isin(in_zone)).fillna(False).astype(int)
        is_hit_into_play = (df_side['description'] == 'hit_into_play').fillna(False).astype(int)
        
        # 합성 마스크 (Bool -> Int)
        is_z_swing = (is_swing & is_in_zone).astype(int)
        is_z_contact = (is_z_swing & is_contact).astype(int)
        is_o_swing = (is_swing & is_out_zone).astype(int)
        is_o_contact = (is_o_swing & is_contact).astype(int)
        
        is_high_ff = ((df_side['pitch_type'] == 'FF') & (df_side['zone'].isin(high_zones))).fillna(False).astype(int)
        is_high_ff_swing = (is_high_ff & is_swing).astype(int)
        is_high_ff_whiff = (is_high_ff_swing & is_whiff).astype(int)
        
        # Barrel 처리 (launch_angle 26~30 & launch_speed >= 98)
        if 'barrel' in df_side.columns:
            is_barrel = (is_hit_into_play.astype(bool) & df_side['barrel'].fillna(0).astype(bool)).fillna(False).astype(int)
        else:
            is_barrel = (is_hit_into_play.astype(bool) & df_side['launch_angle'].between(26, 30).fillna(False) & (df_side['launch_speed'] >= 98).fillna(False)).fillna(False).astype(int)

        # 연산에 필요한 요소들을 새로운 컬럼으로 할당
        df_side['cnt_swing'] = is_swing
        df_side['cnt_whiff'] = is_whiff
        df_side['cnt_z_swing'] = is_z_swing
        df_side['cnt_z_contact'] = is_z_contact
        df_side['cnt_out_zone'] = is_out_zone
        df_side['cnt_o_swing'] = is_o_swing
        df_side['cnt_o_contact'] = is_o_contact
        df_side['cnt_high_ff_swing'] = is_high_ff_swing
        df_side['cnt_high_ff_whiff'] = is_high_ff_whiff
        df_side['cnt_hit_into_play'] = is_hit_into_play
        df_side['cnt_barrel'] = is_barrel
        
        # 3. Groupby Aggregation (루프 1초 컷 최적화)
        agg_funcs = {
            'cnt_swing': 'sum',
            'cnt_whiff': 'sum',
            'cnt_z_swing': 'sum',
            'cnt_z_contact': 'sum',
            'cnt_out_zone': 'sum',
            'cnt_o_swing': 'sum',
            'cnt_o_contact': 'sum',
            'cnt_high_ff_swing': 'sum',
            'cnt_high_ff_whiff': 'sum',
            'cnt_hit_into_play': 'sum',
            'cnt_barrel': 'sum',
            'launch_angle': 'mean' # NaN이 무시되며 자동으로 평균을 계산
        }
        
        grouped = df_side.groupby(['batter', 'player_name']).agg(agg_funcs).reset_index()
        
        # 4. 비율(Pct) 수식 벡터 연산 (0나누기 방지 위해 np.where 또는 div 사용)
        def safe_div(num, den):
            return np.where(den > 0, num / den, 0.0)
            
        grouped['whiff_pct'] = safe_div(grouped['cnt_whiff'], grouped['cnt_swing'])
        grouped['z_contact_pct'] = safe_div(grouped['cnt_z_contact'], grouped['cnt_z_swing'])
        grouped['o_swing_pct'] = safe_div(grouped['cnt_o_swing'], grouped['cnt_out_zone'])
        grouped['o_contact_pct'] = safe_div(grouped['cnt_o_contact'], grouped['cnt_o_swing'])
        grouped['avg_launch_angle'] = grouped['launch_angle']
        grouped['barrel_pct'] = safe_div(grouped['cnt_barrel'], grouped['cnt_hit_into_play'])
        grouped['pull_pct'] = 0.40 # TODO: 정확한 Pull% 계산 로직 대체
        grouped['high_ff_whiff_pct'] = safe_div(grouped['cnt_high_ff_whiff'], grouped['cnt_high_ff_swing'])
        
        # 컬럼 정리 및 fillna
        features_df = grouped[['batter', 'player_name', 'whiff_pct', 'z_contact_pct', 'o_swing_pct', 
                               'o_contact_pct', 'avg_launch_angle', 'barrel_pct', 'pull_pct', 'high_ff_whiff_pct']].copy()
        
        features_df = features_df.rename(columns={'batter': 'batter_id'})
        features_df = features_df.fillna(0)
        
        elapsed_time = time.time() - start_time
        print(f" -> 벡터화 피처 추출 완료 (소요시간: {elapsed_time:.2f}초)")
        
        self.batter_features[stand] = features_df
        return features_df

    def _apply_umap_kmeans(self, stand: str) -> int:
        """
        [내부 메서드] 추출된 특성을 바탕으로 StandardScaler -> UMAP -> K-Means (K=8 고정) 적용
        """
        print(f"[{stand}] 타자 데이터 UMAP 및 K-Means 군집화 진행 중...")
        features_df = self.batter_features[stand]
        if features_df is None or len(features_df) < 8:
            print(" -> 군집화하기에 데이터가 너무 적습니다.")
            return -1
            
        # 1. StandardScaler 전처리
        feature_cols = ['whiff_pct', 'z_contact_pct', 'o_swing_pct', 'o_contact_pct', 
                        'avg_launch_angle', 'barrel_pct', 'pull_pct', 'high_ff_whiff_pct']
                        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df[feature_cols])
        
        # 2. UMAP 차원 축소 (random_state 제거, n_jobs=-1 병렬 처리 추가)
        print(" -> UMAP 차원 축소 완료")
        reducer = umap.UMAP(n_components=2, n_jobs=-1)
        embedding = reducer.fit_transform(X_scaled)
        
        features_df['umap_1'] = embedding[:, 0]
        features_df['umap_2'] = embedding[:, 1]
        
        # 3. K-Means 군집화 (K=8 고정)
        print(" -> K-Means (K=8) 최적화 및 실루엣 점수 측정...")
        k = 8
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embedding)
        
        score = silhouette_score(embedding, labels)
        print(f" -> [{stand}] 타자 고정 탐색 완료: K=8 (Score: {score:.4f})")
        
        features_df['cluster'] = labels
        self.batter_features[stand] = features_df
        return k
        
    def log_interactive_scatter_to_wandb(self, stand: str):
        """
        Plotly를 사용하여 인터랙티브 2D 산점도를 그리고 W&B에 로깅 (hover data에 player_name 포함)
        """
        features_df = self.batter_features[stand]
        if features_df is None or 'cluster' not in features_df.columns:
            return
            
        print(f"[{stand}] 타자 인터랙티브 산점도 시각화 및 W&B 로깅 중...")
        
        # Plotly Express를 활용한 산점도
        features_df['Cluster_Label'] = "Approach_" + features_df['cluster'].astype(str)
        
        fig = px.scatter(
            features_df, 
            x='umap_1', 
            y='umap_2', 
            color='Cluster_Label',
            hover_data=['player_name', 'whiff_pct', 'avg_launch_angle', 'high_ff_whiff_pct'],
            title=f"UMAP Projection of Batter Approaches ({stand}HB, K=8)",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig.update_layout(
            autosize=False,
            width=800,
            height=600,
        )
        
        if wandb.run:
            # W&B에 Plotly 객체 그대로 전송 (인터랙티브 기능 유지)
            wandb.log({f"Batter_Clustering_{stand}HB": wandb.Html(fig.to_html())})
            print(f" -> W&B 대시보드에 [{stand}HB] 산점도 로깅 완료!")
        else:
            print(" -> W&B run이 비활성화됨. Html 파일로 임시 저장합니다.")
            fig.write_html(f"batter_clustering_{stand}hb_temp.html")

    def run_clustering_pipeline(self) -> dict:
        """
        전체 파이프라인 (좌/우 타석 분리 모델링 및 차원 축소/로깅) 실행
        """
        import os
        
        if self.raw_df is None:
            self.fetch_statcast_data()
            
        # 1. 특성 추출 (좌타, 우타 분리)
        self._extract_batter_features(self.raw_df, stand='L')
        self._extract_batter_features(self.raw_df, stand='R')
        
        # 2. UMAP & K-Means 및 로깅 (좌타)
        if self.batter_features['L'] is not None and not self.batter_features['L'].empty:
            self._apply_umap_kmeans('L')
            self.log_interactive_scatter_to_wandb('L')
            
        # 3. UMAP & K-Means 및 로깅 (우타)
        if self.batter_features['R'] is not None and not self.batter_features['R'].empty:
            self._apply_umap_kmeans('R')
            self.log_interactive_scatter_to_wandb('R')
            
        # 4. 결과 통합 및 CSV Export (WandB 영구보관)
        merged_dfs = []
        if self.batter_features['L'] is not None:
            df_l = self.batter_features['L'].copy()
            df_l['stand'] = 'L'
            merged_dfs.append(df_l)
            
        if self.batter_features['R'] is not None:
            df_r = self.batter_features['R'].copy()
            df_r['stand'] = 'R'
            merged_dfs.append(df_r)
            
        if merged_dfs:
            final_df = pd.concat(merged_dfs, ignore_index=True)
            export_df = final_df[['batter_id', 'player_name', 'stand', 'cluster']]
            
            # data/ 폴더 (사전에 생성되었다고 가정)에 저장
            csv_path = "data/batter_clusters_2023.csv"
            export_df.to_csv(csv_path, index=False)
            print(f" -> 로컬 CSV 백업 완료: {csv_path} (총 {len(export_df)}명)")
            
            # Wandb Artifact 로그
            if wandb.run:
                artifact = wandb.Artifact(name="batter_cluster_mapping", type="dataset")
                artifact.add_file(csv_path)
                wandb.log_artifact(artifact)
                print(" -> WandB Artifact에 `batter_cluster_mapping` CSV 영구 보관 완료!")
            
        return {
            'LHB_features': self.batter_features['L'],
            'RHB_features': self.batter_features['R']
        }

if __name__ == "__main__":
    wandb.init(project="SmartPitch-Portfolio", name="Batter_Clustering_2023_Full")
    
    # 2023년 시즌 1년 전체 데이터 추출 (500 투구수)
    bc = BatterClustering(start_date="2023-03-30", end_date="2023-10-01", n_pitches_threshold=500)
    results = bc.run_clustering_pipeline()
    print("모든 실전 파이프라인 처리가 완료되었습니다.")
    
    # 타자 군집(K=8) 특성 확인용 터미널 출력
    if results.get('RHB_features') is not None and not results['RHB_features'].empty:
        print("\n=== 우타자(RHB) 타격 어프로치 중심(Centroid) 프로필 ===")
        metrics = ['whiff_pct', 'z_contact_pct', 'o_swing_pct', 'avg_launch_angle', 'barrel_pct', 'high_ff_whiff_pct']
        cluster_summary = results['RHB_features'].groupby('cluster')[metrics].mean().round(3)
        print(cluster_summary)
        print("====================================================\n")
        
    wandb.finish()
