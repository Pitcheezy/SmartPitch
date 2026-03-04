import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class PitchClustering:
    """
    전처리된 투구 데이터를 입력받아 UMAP 차원 축소와 최적의 K-Means 클러스터링을 수행하고,
    클러스터별 평균 구속에 따라 임의의 구종 이름을 매핑하는 클래스
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        초기화 메서드
        :param df: PitchDataLoader를 통해 전처리가 완료된 데이터프레임
        """
        self.df = df.copy()
        self.features = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'release_pos_x', 'release_pos_z']
        self.best_k = None
        self.pitch_map = {}
        
    def _apply_umap(self) -> np.ndarray:
        """
        [내부 메서드] 투구 피처를 표준화(StandardScaling)한 후 UMAP으로 2차원 압축
        """
        print("UMAP 차원 축소 진행 중...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.features])
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        
        # 나중에 시각화 시 편하게 쓰기 위해 데이터프레임에 좌표 저장
        self.df['umap_1'] = embedding[:, 0]
        self.df['umap_2'] = embedding[:, 1]
        
        return embedding
        
    def _find_optimal_clusters(self, embedding: np.ndarray, min_k: int = 3, max_k: int = 6) -> int:
        """
        [내부 메서드] K값을 min_k부터 max_k까지 변화시키며 실루엣 스코어가 가장 높은 최적의 K 탐색
        """
        print(f"최적의 군집 수(K) 탐색 중 (범위: {min_k} ~ {max_k})...")
        best_score = -1
        best_k = min_k
        best_labels = None
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embedding)
            score = silhouette_score(embedding, labels)
            print(f" - K={k}: Silhouette Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                
        print(f"선택된 최적의 K: {best_k} (Score: {best_score:.4f})")
        self.best_k = best_k
        self.df['true_pitch_type'] = best_labels
        
        return best_k
        
    def _map_pitch_names(self):
        """
        [내부 메서드] 식별된 군집들의 평균 구속을 계산하여 직관적인 구종 이름으로 매핑
        """
        print("클러스터별 평균 구속 기반 구종 이름 매핑 중...")
        avg_speeds = self.df.groupby('true_pitch_type')['release_speed'].mean().sort_values(ascending=False)
        
        # 구종 이름 후보 (가장 빠른 것부터 순서대로 매핑)
        pitch_names = ['Fastball', 'Slider', 'Changeup', 'Curveball', 'Sweeper', 'Sinker']
        
        for i, (cluster_num, speed) in enumerate(avg_speeds.items()):
            # 혹시 구종이 6개를 초과할 경우를 대비한 방어 코드
            pitch_name = pitch_names[i] if i < len(pitch_names) else f"Pitch_{i}"
            self.pitch_map[cluster_num] = pitch_name
            print(f" - 군집 {cluster_num}: {pitch_name} (평균 구속: {speed:.1f} mph)")
            
        self.df['mapped_pitch_name'] = self.df['true_pitch_type'].map(self.pitch_map)
        
    def log_umap_scatter_to_wandb(self):
        """
        seaborn을 사용해 UMAP 산점도를 그리고 W&B에 로깅
        """
        print("UMAP 산점도 시각화 및 W&B 로깅 중...")
        plt.figure(figsize=(10, 7))
        
        sns.scatterplot(
            data=self.df, 
            x='umap_1', 
            y='umap_2', 
            hue='mapped_pitch_name', 
            palette='Set1', 
            alpha=0.7
        )
        
        plt.title(f'UMAP Projection of True Pitch Arsenal (Optimal K={self.best_k})', fontsize=16, fontweight='bold')
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.legend(title='Mapped Pitch Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if wandb.run:
            wandb.log({"UMAP_Clustering": wandb.Image(plt)})
            print("W&B 대시보드에 UMAP 산점도 로깅 완료!")
        else:
            print("W&B run이 활성화되어 있지 않아 화면에만 표시합니다.")
            plt.show()
            
        plt.close()
        
    def run_clustering_pipeline(self) -> pd.DataFrame:
        """
        전체 클러스터링 파이프라인 실행
        1. UMAP 차원 축소
        2. 최적 K-Means 군집화
        3. 구속 기반 구종 이름 매핑
        4. W&B 시각화 로깅
        :return: 군집 라벨(true_pitch_type) 및 매핑된 구종 이름(mapped_pitch_name)이 추가된 데이터프레임
        """
        embedding = self._apply_umap()
        self._find_optimal_clusters(embedding, min_k=3, max_k=6)
        self._map_pitch_names()
        self.log_umap_scatter_to_wandb()
        
        return self.df
