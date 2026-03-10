"""
투수·타자 프로필 생성

- 투수: 클러스터(did_cluster=1) 또는 pitch_type(did_cluster=0) 기반 mix/feature
- 타자: launch_speed/angle 평균, stand, events/description 분포
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 투수 프로필용 피처
PITCH_FEATURES = [
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z", "release_extension", "arm_angle"
]


def build_pitcher_profiles(dfp: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    투수 프로필 생성

    - did_cluster=1 투수: 클러스터 mix + 가중 평균 피처
    - did_cluster=0 투수: pitch_type mix + 가중 평균 피처 (fallback)

    Args:
        dfp: pitch_umap_cluster (행=투구)
        summary: pitcher_cluster_summary (did_cluster 포함)
    """
    required = ["pitcher", "pitch_type", "pitch_cluster_id", "pitch_cluster_local"]
    missing = [c for c in required if c not in dfp.columns]
    if missing:
        raise KeyError(f"Missing required columns in pitch_umap_cluster: {missing}")

    dfp = dfp.copy()
    dfp["pitcher"] = pd.to_numeric(dfp["pitcher"], errors="coerce").astype(int)

    summary = summary.copy()
    if "pitcher" in summary.columns:
        summary["pitcher"] = pd.to_numeric(summary["pitcher"], errors="coerce").astype(int)
    else:
        summary["pitcher"] = []

    cluster_ok = set(summary.loc[summary.get("did_cluster", 0) == 1, "pitcher"].astype(int).tolist())
    all_pitchers = sorted(dfp["pitcher"].unique().tolist())

    feats = [c for c in PITCH_FEATURES if c in dfp.columns]

    # A) cluster 기반
    df_cluster = dfp[dfp["pitcher"].isin(cluster_ok)].copy()

    cluster_counts = (
        df_cluster.groupby(["pitcher", "pitch_cluster_id"]).size().rename("n").reset_index()
    )
    totals = cluster_counts.groupby("pitcher")["n"].sum().rename("n_total").reset_index()
    cluster_counts = cluster_counts.merge(totals, on="pitcher", how="left")
    cluster_counts["ratio"] = cluster_counts["n"] / cluster_counts["n_total"]

    mix_wide = cluster_counts.pivot(index="pitcher", columns="pitch_cluster_id", values="ratio").fillna(0.0)
    mix_wide.columns = [f"mix_{c}" for c in mix_wide.columns]

    if feats:
        cluster_feat_mean = (
            df_cluster.groupby(["pitcher", "pitch_cluster_id"])[feats].mean().reset_index()
        )
        tmp = cluster_feat_mean.merge(
            cluster_counts[["pitcher", "pitch_cluster_id", "ratio"]],
            on=["pitcher", "pitch_cluster_id"],
            how="left"
        )
        for c in feats:
            tmp[c] = tmp[c] * tmp["ratio"]
        pitcher_feat = tmp.groupby("pitcher")[feats].sum()
        pitcher_feat.columns = [f"feat_{c}_wavg" for c in pitcher_feat.columns]
        pitcher_profile_cluster = mix_wide.join(pitcher_feat, how="outer").fillna(0.0)
    else:
        pitcher_profile_cluster = mix_wide.copy()

    pitcher_profile_cluster["profile_source"] = "cluster"

    # B) fallback (pitch_type 기반)
    fallback_pitchers = [p for p in all_pitchers if p not in cluster_ok]
    df_fb = dfp[dfp["pitcher"].isin(fallback_pitchers)].copy()

    pt_counts = (
        df_fb.groupby(["pitcher", "pitch_type"]).size().rename("n").reset_index()
    )
    pt_totals = pt_counts.groupby("pitcher")["n"].sum().rename("n_total").reset_index()
    pt_counts = pt_counts.merge(pt_totals, on="pitcher", how="left")
    pt_counts["ratio"] = pt_counts["n"] / pt_counts["n_total"]

    pt_wide = pt_counts.pivot(index="pitcher", columns="pitch_type", values="ratio").fillna(0.0)
    pt_wide.columns = [f"mix_pitchtype_{c}" for c in pt_wide.columns]

    if feats:
        pt_feat = (
            df_fb.groupby(["pitcher", "pitch_type"])[feats].mean().reset_index()
            .merge(pt_counts[["pitcher", "pitch_type", "ratio"]], on=["pitcher", "pitch_type"], how="left")
        )
        for c in feats:
            pt_feat[c] = pt_feat[c] * pt_feat["ratio"]
        pitcher_feat_fb = pt_feat.groupby("pitcher")[feats].sum()
        pitcher_feat_fb.columns = [f"feat_{c}_wavg" for c in pitcher_feat_fb.columns]
        pitcher_profile_fb = pt_wide.join(pitcher_feat_fb, how="outer").fillna(0.0)
    else:
        pitcher_profile_fb = pt_wide.copy()

    pitcher_profile_fb["profile_source"] = "pitch_type_fallback"

    pitcher_profiles = pd.concat([pitcher_profile_cluster, pitcher_profile_fb], axis=0)
    pitcher_profiles = pitcher_profiles.reset_index().rename(columns={"index": "pitcher"})

    # summary 정보 붙이기
    keep_cols = [c for c in ["pitcher","n_pitches","did_umap","did_cluster","n_clusters","noise_ratio"] if c in summary.columns]
    if keep_cols:
        pitcher_profiles = pitcher_profiles.merge(summary[keep_cols], on="pitcher", how="left")

    # 결측 안전 처리
    if "n_pitches" not in pitcher_profiles.columns:
        pitcher_profiles["n_pitches"] = dfp.groupby("pitcher").size().reindex(pitcher_profiles["pitcher"]).values

    for c in ["did_umap","did_cluster","n_clusters"]:
        if c in pitcher_profiles.columns:
            pitcher_profiles[c] = pitcher_profiles[c].fillna(0).astype(int)

    if "noise_ratio" in pitcher_profiles.columns:
        pitcher_profiles["noise_ratio"] = pitcher_profiles["noise_ratio"].astype(float)

    pitcher_profiles = pitcher_profiles.fillna(0.0)

    return pitcher_profiles


def build_batter_profiles(dfp: pd.DataFrame) -> pd.DataFrame:
    """
    타자 프로필 생성

    - launch_speed, launch_angle 평균
    - stand(타석) 최빈값
    - events_group, description_group 분포 비율
    """
    required = ["batter"]
    missing = [c for c in required if c not in dfp.columns]
    if missing:
        raise KeyError(f"Missing required columns for batter profiles: {missing}")

    dfb = dfp.copy()
    dfb["batter"] = pd.to_numeric(dfb["batter"], errors="coerce").astype(int)

    # numeric
    num_cols = [c for c in ["launch_speed","launch_angle"] if c in dfb.columns]
    if num_cols:
        batter_num = dfb.groupby("batter")[num_cols].mean()
        batter_num = batter_num.rename(columns={c: f"batter_avg_{c}" for c in num_cols})
    else:
        batter_num = pd.DataFrame(index=sorted(dfb["batter"].unique().tolist()))

    # stand mode
    if "stand" in dfb.columns:
        batter_stand = dfb.groupby("batter")["stand"].agg(lambda x: x.value_counts().index[0])
        batter_stand = batter_stand.to_frame("batter_stand_mode")
    else:
        batter_stand = pd.DataFrame(index=batter_num.index, columns=["batter_stand_mode"])

    # distributions
    dist_frames = []
    for c in ["events_group", "description_group"]:
        if c not in dfb.columns:
            continue
        cnt = dfb.groupby(["batter", c]).size().rename("n").reset_index()
        tot = cnt.groupby("batter")["n"].sum().rename("n_total").reset_index()
        cnt = cnt.merge(tot, on="batter", how="left")
        cnt["ratio"] = cnt["n"] / cnt["n_total"]
        wide = cnt.pivot(index="batter", columns=c, values="ratio").fillna(0.0)
        wide.columns = [f"batter_{c}_{v}" for v in wide.columns]
        dist_frames.append(wide)

    batter_profiles = batter_num.join(batter_stand, how="outer")
    for wide in dist_frames:
        batter_profiles = batter_profiles.join(wide, how="outer")

    batter_profiles["n_pitches_seen"] = dfb.groupby("batter").size()
    batter_profiles = batter_profiles.fillna(0.0).reset_index()

    return batter_profiles
