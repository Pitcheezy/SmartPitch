"""
투수-타자 매치업 테이블 생성

- pitch_level: 투구 단위 (pitcher/batter 프로필 조인)
- pair_level: (pitcher, batter) 쌍 단위 집계 + 상위 K 클러스터 요약
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchupConfig:
    """매치업 설정"""
    topk: int = 5  # pair_level에서 상위 K개 클러스터만 요약


def _ensure_labels(dfp: pd.DataFrame) -> pd.DataFrame:
    dfp = dfp.copy()

    # events_group 라벨
    if "events_group" in dfp.columns:
        dfp["label_events_group"] = dfp["events_group"].fillna("other")
    elif "events" in dfp.columns:
        EVENT_TO_GROUP = {
            "single": "hit", "double": "hit", "triple": "hit", "home_run": "hit",
            "walk": "walk", "intent_walk": "walk", "hit_by_pitch": "walk",
            "strikeout": "out", "strikeout_double_play": "out",
            "field_out": "out", "force_out": "out", "double_play": "out",
            "grounded_into_double_play": "out",
            "field_error": "reached",
        }
        dfp["label_events_group"] = dfp["events"].map(EVENT_TO_GROUP).fillna("other")
    else:
        dfp["label_events_group"] = "other"

    # description_group 라벨
    if "description_group" in dfp.columns:
        dfp["label_desc_group"] = dfp["description_group"].fillna("other")
    elif "description" in dfp.columns:
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
        dfp["label_desc_group"] = dfp["description"].map(DESC_TO_GROUP).fillna("other")
    else:
        dfp["label_desc_group"] = "other"

    return dfp


def build_matchup_tables(
    dfp: pd.DataFrame,
    pitcher_profiles: pd.DataFrame,
    batter_profiles: pd.DataFrame,
    cfg: MatchupConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    매치업 테이블 3종 생성

    Returns:
        pitch_level: 투구 단위 (투수/타자 프로필 조인)
        pair_level_final: (pitcher, batter) 쌍 단위 + 상위 K 클러스터 요약
        pair_cluster_level: (pitcher, batter, cluster) 단위 (시각화/분석용)
    """
    dfp = _ensure_labels(dfp)

    required = ["pitcher","batter","pitch_type","balls","strikes","pitch_cluster_id","pitch_cluster_local"]
    missing = [c for c in required if c not in dfp.columns]
    if missing:
        raise KeyError(f"Missing columns for matchup: {missing}")

    dfp = dfp.copy()
    dfp["pitcher"] = pd.to_numeric(dfp["pitcher"], errors="coerce").astype(int)
    dfp["batter"] = pd.to_numeric(dfp["batter"], errors="coerce").astype(int)

    # pitch-level base features
    base_pitch_features = [
        "pitcher","batter","game_date",
        "pitch_type","balls","strikes",
        "plate_x","plate_z",
        "release_speed","release_spin_rate","pfx_x","pfx_z",
        "release_pos_x","release_pos_z","release_extension","arm_angle",
        "outs_when_up","inning","inning_topbot","bat_score_diff",
        "on_1b","on_2b","on_3b",
        "pitch_cluster_id","pitch_cluster_local",
        "umap_x","umap_y",
        "launch_speed","launch_angle",
        "base_state","count",
    ]
    base_pitch_features = [c for c in base_pitch_features if c in dfp.columns]

    pitch_level = dfp[base_pitch_features + ["label_events_group","label_desc_group"]].copy()
    pitch_level["matchup_id"] = pitch_level["pitcher"].astype(str) + "_" + pitch_level["batter"].astype(str)

    # join profiles
    pitch_level = pitch_level.merge(pitcher_profiles, on="pitcher", how="left")
    pitch_level = pitch_level.merge(batter_profiles, on="batter", how="left")

    # pair-level 기본 집계
    pair_base = pitch_level.groupby(["pitcher","batter"]).size().rename("n_pitches").reset_index()

    ev = pitch_level.groupby(["pitcher","batter","label_events_group"]).size().rename("n").reset_index()
    ev = ev.merge(pair_base, on=["pitcher","batter"], how="left")
    ev["ratio"] = ev["n"] / ev["n_pitches"]
    ev_wide = ev.pivot(index=["pitcher","batter"], columns="label_events_group", values="ratio").fillna(0.0)
    ev_wide.columns = [f"ev_{c}_ratio" for c in ev_wide.columns]

    dg = pitch_level.groupby(["pitcher","batter","label_desc_group"]).size().rename("n").reset_index()
    dg = dg.merge(pair_base, on=["pitcher","batter"], how="left")
    dg["ratio"] = dg["n"] / dg["n_pitches"]
    dg_wide = dg.pivot(index=["pitcher","batter"], columns="label_desc_group", values="ratio").fillna(0.0)
    dg_wide.columns = [f"desc_{c}_ratio" for c in dg_wide.columns]

    pair_level = pair_base.set_index(["pitcher","batter"]).join(ev_wide, how="left").join(dg_wide, how="left").fillna(0.0).reset_index()

    # (pitcher,batter,cluster) 단위
    pcb = pitch_level.groupby(["pitcher","batter","pitch_cluster_id"]).size().rename("n").reset_index()

    evc = pitch_level.groupby(["pitcher","batter","pitch_cluster_id","label_events_group"]).size().rename("n_ev").reset_index()
    evc = evc.merge(pcb, on=["pitcher","batter","pitch_cluster_id"], how="left")
    evc["ratio"] = evc["n_ev"] / evc["n"]
    evc_wide = evc.pivot(index=["pitcher","batter","pitch_cluster_id"], columns="label_events_group", values="ratio").fillna(0.0)
    evc_wide.columns = [f"cl_ev_{c}_ratio" for c in evc_wide.columns]
    evc_wide = evc_wide.reset_index()

    dgc = pitch_level.groupby(["pitcher","batter","pitch_cluster_id","label_desc_group"]).size().rename("n_dg").reset_index()
    dgc = dgc.merge(pcb, on=["pitcher","batter","pitch_cluster_id"], how="left")
    dgc["ratio"] = dgc["n_dg"] / dgc["n"]
    dgc_wide = dgc.pivot(index=["pitcher","batter","pitch_cluster_id"], columns="label_desc_group", values="ratio").fillna(0.0)
    dgc_wide.columns = [f"cl_desc_{c}_ratio" for c in dgc_wide.columns]
    dgc_wide = dgc_wide.reset_index()

    agg_cols = [c for c in ["launch_speed","launch_angle"] if c in pitch_level.columns]
    if agg_cols:
        cl_bip = pitch_level.groupby(["pitcher","batter","pitch_cluster_id"])[agg_cols].mean().reset_index()
        cl_bip = cl_bip.rename(columns={c: f"cl_avg_{c}" for c in agg_cols})
    else:
        cl_bip = pd.DataFrame(columns=["pitcher","batter","pitch_cluster_id"])

    pair_cluster_level = pcb.merge(evc_wide, on=["pitcher","batter","pitch_cluster_id"], how="left")
    pair_cluster_level = pair_cluster_level.merge(dgc_wide, on=["pitcher","batter","pitch_cluster_id"], how="left")
    if not cl_bip.empty:
        pair_cluster_level = pair_cluster_level.merge(cl_bip, on=["pitcher","batter","pitch_cluster_id"], how="left")
    pair_cluster_level = pair_cluster_level.fillna(0.0)

    # topK 요약(문자열/숫자 fill 분리로 parquet 안전)
    pair_cluster_level["cluster_ratio_in_pair"] = pair_cluster_level["n"] / pair_cluster_level.groupby(["pitcher","batter"])["n"].transform("sum")
    pair_cluster_level["rank_in_pair"] = pair_cluster_level.groupby(["pitcher","batter"])["cluster_ratio_in_pair"].rank(ascending=False, method="first")
    top = pair_cluster_level[pair_cluster_level["rank_in_pair"] <= cfg.topk].copy()

    keep_metrics = [c for c in top.columns if c.startswith("cl_ev_") or c.startswith("cl_desc_") or c.startswith("cl_avg_")]
    keep_metrics = sorted(set(keep_metrics))

    rows = []
    for (p,b), g in top.groupby(["pitcher","batter"]):
        g = g.sort_values("rank_in_pair")
        row = {"pitcher": int(p), "batter": int(b)}
        for i, (_, r) in enumerate(g.iterrows(), start=1):
            row[f"top{i}_cluster_id"] = str(r["pitch_cluster_id"])
            row[f"top{i}_cluster_ratio"] = float(r["cluster_ratio_in_pair"])
            for m in keep_metrics:
                row[f"top{i}_{m}"] = float(r[m])
        rows.append(row)

    pair_topk = pd.DataFrame(rows)

    pair_level_final = pair_level.merge(pair_topk, on=["pitcher","batter"], how="left")

    # 문자열/숫자 fill 분리(햄이 겪은 ArrowTypeError 방지)
    cluster_id_cols = [c for c in pair_level_final.columns if c.startswith("top") and c.endswith("_cluster_id")]
    for c in cluster_id_cols:
        pair_level_final[c] = pair_level_final[c].astype("string").fillna("NONE")

    num_cols = pair_level_final.select_dtypes(include=[np.number]).columns
    pair_level_final[num_cols] = pair_level_final[num_cols].fillna(0.0)

    obj_cols = pair_level_final.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        pair_level_final[c] = pair_level_final[c].fillna("")

    return pitch_level, pair_level_final, pair_cluster_level
