"""
투수별 UMAP 임베딩 및 HDBSCAN 클러스터링

투수별로 구속, 스핀, 무브먼트 등 피처를 UMAP으로 차원 축소한 뒤
HDBSCAN으로 구종 유사 클러스터를 생성합니다.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# UMAP 입력 피처 (투구 특성)
PITCH_FEATURES_FOR_UMAP = [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "arm_angle",
]


@dataclass(frozen=True)
class EmbeddingConfig:
    """UMAP/HDBSCAN 파라미터"""
    # 최소 투구 수 (이하면 UMAP/클러스터링 스킵)
    min_pitches_for_umap: int = 40
    min_pitches_for_cluster: int = 120

    # PCA→UMAP
    use_pca: bool = True
    pca_n_components: int = 6

    # UMAP
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    umap_random_state: int = 42

    # HDBSCAN 기본값(표본 큰 경우)
    hdb_min_cluster_size_default: int = 60
    hdb_min_samples_default: int = 15

    # 로그
    log_every: int = 25


def _adaptive_hdbscan_params(n: int, cfg: EmbeddingConfig):
    if n < cfg.min_pitches_for_cluster:
        return None
    min_cluster_size = int(max(15, min(cfg.hdb_min_cluster_size_default, n * 0.10)))
    min_samples = int(max(5, min(cfg.hdb_min_samples_default, min_cluster_size // 2)))
    return min_cluster_size, min_samples


def run_pitcher_umap_cluster(df: pd.DataFrame, cfg: EmbeddingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    투수별 UMAP + HDBSCAN 클러스터링 수행

    Args:
        df: pitch_clean (행=투구)
        cfg: EmbeddingConfig

    Returns:
        df_out: 투구별 umap_x, umap_y, pitch_cluster_local, pitch_cluster_id 추가
        summary: 투수별 요약 (n_pitches, did_umap, did_cluster, n_clusters 등)
    """
    # 필요한 컬럼 체크
    need = ["pitcher"] + PITCH_FEATURES_FOR_UMAP
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for embedding: {missing}")

    # NaN 제거(UMAP/HDBSCAN NaN 불가)
    df2 = df.dropna(subset=need).copy()
    df2["pitcher"] = pd.to_numeric(df2["pitcher"], errors="coerce")
    df2 = df2.dropna(subset=["pitcher"]).copy()
    df2["pitcher"] = df2["pitcher"].astype(int)

    # 투수 목록
    pitch_counts = df2.groupby("pitcher").size().sort_values(ascending=False)
    pitchers = pitch_counts.index.tolist()

    # 라이브러리 import(환경에 없으면 여기서 명확히 터짐)
    import umap
    import hdbscan

    results = []
    summary_rows = []

    t0 = time.time()
    total = len(pitchers)

    for idx, pid in enumerate(pitchers, start=1):
        sub = df2[df2["pitcher"] == pid].copy()
        n = len(sub)

        if n < cfg.min_pitches_for_umap:
            summary_rows.append({
                "pitcher": int(pid),
                "n_pitches": int(n),
                "did_umap": 0,
                "did_cluster": 0,
                "n_clusters": 0,
                "noise_ratio": np.nan,
                "note": f"UMAP skipped (<{cfg.min_pitches_for_umap})"
            })
            if idx % cfg.log_every == 0 or idx == 1 or idx == total:
                print(f"[{idx}/{total}] pitcher={pid} skipped (n={n})")
            continue

        X = sub[PITCH_FEATURES_FOR_UMAP].to_numpy(dtype=float)

        # 투수별 스케일
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # PCA
        if cfg.use_pca:
            n_comp = min(cfg.pca_n_components, Xs.shape[1])
            pca = PCA(n_components=n_comp, random_state=cfg.umap_random_state)
            X_in = pca.fit_transform(Xs)
        else:
            X_in = Xs

        # UMAP neighbors는 표본 적을 때 자동 축소
        n_neighbors = min(cfg.umap_n_neighbors, max(5, n - 1))
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=cfg.umap_min_dist,
            n_components=cfg.umap_n_components,
            random_state=cfg.umap_random_state,
            n_jobs=1,  # 명시적으로 지정 (random_state와 함께 사용할 때 경고 제거)
        )
        emb = reducer.fit_transform(X_in)
        sub["umap_x"] = emb[:, 0]
        sub["umap_y"] = emb[:, 1]

        # clustering
        params = _adaptive_hdbscan_params(n, cfg)
        if params is None:
            labels = np.full((n,), -1, dtype=int)
            did_cluster = 0
            n_clusters = 0
            noise_ratio = 1.0
            note = f"Clustering skipped (<{cfg.min_pitches_for_cluster})"
        else:
            min_cluster_size, min_samples = params
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            labels = clusterer.fit_predict(emb)
            did_cluster = 1
            n_clusters = int(np.unique(labels[labels != -1]).shape[0])
            noise_ratio = float(np.mean(labels == -1))
            note = f"HDBSCAN(min_cluster_size={min_cluster_size}, min_samples={min_samples})"

        sub["pitch_cluster_local"] = labels
        sub["pitch_cluster_id"] = sub["pitcher"].astype(str) + "_" + sub["pitch_cluster_local"].astype(str)

        results.append(sub)

        row = {
            "pitcher": int(pid),
            "n_pitches": int(n),
            "did_umap": 1,
            "did_cluster": int(did_cluster),
            "n_clusters": int(n_clusters),
            "noise_ratio": float(noise_ratio),
            "note": note,
        }

        if did_cluster:
            vc = pd.Series(labels).value_counts()
            top = vc[vc.index != -1].head(8)
            for k, v in top.items():
                row[f"local_cluster_{int(k)}_ratio"] = float(v / n)

        summary_rows.append(row)

        if idx % cfg.log_every == 0 or idx == 1 or idx == total:
            elapsed = time.time() - t0
            print(f"[{idx}/{total}] pitcher={pid} done (n={n}) | elapsed={elapsed:.1f}s")

    df_out = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    return df_out, summary
