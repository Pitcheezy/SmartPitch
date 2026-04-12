"""
발표용 투구 위치 히트맵 비교 시각화

왼쪽: 실제 Statcast 투구 분포 (pitcher cluster 0의 Fastball)
오른쪽: DQN 추천 (지정 상태에서 Fastball × 존별 Q값)

출력: docs/presentation_charts/pitch_location_comparison.png (300dpi)

실행:
    uv run scripts/generate_pitch_location_heatmaps.py
"""

import os
import platform
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import torch

# ── 한글 폰트 설정 ───────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = "AppleGothic"
else:
    matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "docs", "presentation_charts")
os.makedirs(OUT_DIR, exist_ok=True)

PITCHER_CLUSTERS_CSV = os.path.join(ROOT, "data", "pitcher_clusters_2023.csv")
DQN_MODEL_PATH = os.path.join(ROOT, "smartpitch_dqn_final.zip")

# ── 스트라이크존 경계 (MLB 표준, 단위: feet) ─────────────────────────────────
SZ_LEFT, SZ_RIGHT = -0.71, 0.71     # 17인치 홈플레이트 폭 / 12 ≈ 0.71
SZ_BOTTOM, SZ_TOP = 1.5, 3.5        # 일반적인 평균 스트라이크존 높이

# ═══════════════════════════════════════════════════════════════════════════
# 1. 실제 Statcast 투구 데이터 수집 (pitcher cluster 0, FF only)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("[1/3] Statcast 데이터 수집 (pitcher cluster 0, Fastball)")
print("=" * 70)

from pybaseball import statcast, cache
cache.enable()

# cluster 0 투수 ID 목록 로드
cluster_df = pd.read_csv(PITCHER_CLUSTERS_CSV)
cluster0_ids = set(cluster_df[cluster_df["cluster"] == 0]["pitcher_id"].tolist())
print(f"  cluster 0 투수 수: {len(cluster0_ids)}")

# 2023 시즌 중 1주일치만 사용 (속도 최적화, pybaseball 캐시 활용)
print("  Statcast 2023-04-01 ~ 2023-04-14 다운로드 중 (캐시 있으면 빠름)...")
raw = statcast(start_dt="2023-04-01", end_dt="2023-04-14")
print(f"  전체 투구 수: {len(raw):,}")

# cluster 0 투수의 Fastball(FF) 만 필터
ff_mask = raw["pitch_type"] == "FF"
c0_mask = raw["pitcher"].isin(cluster0_ids)
loc_mask = raw["plate_x"].notna() & raw["plate_z"].notna()
ff_c0 = raw[ff_mask & c0_mask & loc_mask].copy()
print(f"  cluster 0 Fastball 투구 수: {len(ff_c0):,}")

plate_x = ff_c0["plate_x"].to_numpy()
plate_z = ff_c0["plate_z"].to_numpy()

# ═══════════════════════════════════════════════════════════════════════════
# 2. DQN 모델 로드 및 Q값 추출
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[2/3] DQN 정책 Q값 추출")
print("=" * 70)

from stable_baselines3 import DQN

print(f"  모델 로드: {DQN_MODEL_PATH}")
dqn = DQN.load(DQN_MODEL_PATH, device="cpu")

# 지정 상태: 0-0 카운트, 0 아웃, 무주자, 투수군집 0, 타자군집 3
# obs = [balls, strikes, outs, 1b, 2b, 3b, batter_cluster, pitcher_cluster]
state = np.array([[0, 0, 0, 0, 0, 0, 3, 0]], dtype=np.float32)
print(f"  상태: balls=0 strikes=0 outs=0 runners=000 batter_c=3 pitcher_c=0")

obs_t = torch.as_tensor(state)
with torch.no_grad():
    q_all = dqn.q_net(obs_t).cpu().numpy()[0]   # shape (52,)

# action = pitch_idx * n_zones + zone_idx, n_pitches=4, n_zones=13
PITCH_NAMES = ["Fastball", "Slider", "Curveball", "Changeup"]
ZONE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
q_grid = q_all.reshape(len(PITCH_NAMES), len(ZONE_IDS))
fastball_q = q_grid[0]  # Fastball 행

print(f"  Q값 범위: min={q_all.min():.3f}, max={q_all.max():.3f}")
print(f"  Fastball Q값 per zone:")
for zid, q in zip(ZONE_IDS, fastball_q):
    print(f"    zone {zid:2d}: {q:+.3f}")

# 존별 도형 좌표 정의
# 1-9: 스트라이크존 3×3 그리드 (사각형)
# 11-14: 스트라이크존 외부 4사분면 (L자 polygon — SZ 영역 제외)
sz_w = SZ_RIGHT - SZ_LEFT
sz_h = SZ_TOP - SZ_BOTTOM
col_w = sz_w / 3.0
row_h = sz_h / 3.0
x_cols = [SZ_LEFT, SZ_LEFT + col_w, SZ_LEFT + 2 * col_w, SZ_RIGHT]
# z_rows: 상단부터
z_rows = [SZ_TOP, SZ_TOP - row_h, SZ_TOP - 2 * row_h, SZ_BOTTOM]

# 플롯 바깥 경계
X_MIN, X_MAX = -2.0, 2.0
Z_MIN, Z_MAX = 0.3, 4.7

# 스트라이크존 중심점 (Z11~14의 경계 기준)
SZ_MID_X = 0.0
SZ_MID_Z = (SZ_TOP + SZ_BOTTOM) / 2.0   # 2.5

# 스트라이크존 내부 9개 사각형 (x0, x1, z0, z1)
zone_inner_rects = {
    1: (x_cols[0], x_cols[1], z_rows[1], z_rows[0]),
    2: (x_cols[1], x_cols[2], z_rows[1], z_rows[0]),
    3: (x_cols[2], x_cols[3], z_rows[1], z_rows[0]),
    4: (x_cols[0], x_cols[1], z_rows[2], z_rows[1]),
    5: (x_cols[1], x_cols[2], z_rows[2], z_rows[1]),
    6: (x_cols[2], x_cols[3], z_rows[2], z_rows[1]),
    7: (x_cols[0], x_cols[1], z_rows[3], z_rows[2]),
    8: (x_cols[1], x_cols[2], z_rows[3], z_rows[2]),
    9: (x_cols[2], x_cols[3], z_rows[3], z_rows[2]),
}

# 스트라이크존 외부 4사분면 L-polygon 정점 (시계반대 방향)
# 각 quadrant는 "큰 사각형"에서 SZ와 겹치는 부분을 빼낸 L자 모양
zone_outer_polygons = {
    11: [  # 상단-좌 (X_MIN~SZ_MID_X, SZ_MID_Z~Z_MAX 에서 SZ 좌상부 제외)
        (X_MIN, SZ_MID_Z), (SZ_LEFT, SZ_MID_Z), (SZ_LEFT, SZ_TOP),
        (SZ_MID_X, SZ_TOP), (SZ_MID_X, Z_MAX), (X_MIN, Z_MAX),
    ],
    12: [  # 상단-우
        (SZ_MID_X, SZ_TOP), (SZ_RIGHT, SZ_TOP), (SZ_RIGHT, SZ_MID_Z),
        (X_MAX, SZ_MID_Z), (X_MAX, Z_MAX), (SZ_MID_X, Z_MAX),
    ],
    13: [  # 하단-좌
        (X_MIN, Z_MIN), (SZ_MID_X, Z_MIN), (SZ_MID_X, SZ_BOTTOM),
        (SZ_LEFT, SZ_BOTTOM), (SZ_LEFT, SZ_MID_Z), (X_MIN, SZ_MID_Z),
    ],
    14: [  # 하단-우
        (SZ_MID_X, Z_MIN), (X_MAX, Z_MIN), (X_MAX, SZ_MID_Z),
        (SZ_RIGHT, SZ_MID_Z), (SZ_RIGHT, SZ_BOTTOM), (SZ_MID_X, SZ_BOTTOM),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# 3. 비교 시각화 (1×2 subplot)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[3/3] 비교 시각화 생성")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

# ── 왼쪽: 실제 투구 분포 (hist2d) ──────────────────────────────────────────
ax = axes[0]
h, xedges, yedges, img = ax.hist2d(
    plate_x, plate_z, bins=[40, 40],
    range=[[X_MIN, X_MAX], [Z_MIN, Z_MAX]],
    cmap="YlOrRd",
)
cbar = fig.colorbar(img, ax=ax, shrink=0.85)
cbar.set_label("투구 빈도", fontsize=11)

# 스트라이크존 오버레이
ax.add_patch(Rectangle(
    (SZ_LEFT, SZ_BOTTOM), SZ_RIGHT - SZ_LEFT, SZ_TOP - SZ_BOTTOM,
    fill=False, edgecolor="black", linewidth=2.5, zorder=5,
))

ax.set_title(f"실제 투구 분포\n(Cluster 0 파워형, Fastball, n={len(ff_c0):,})",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("plate_x (ft, 포수 시점)", fontsize=12)
ax.set_ylabel("plate_z (ft)", fontsize=12)
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Z_MIN, Z_MAX)
ax.set_aspect("equal")

# ── 오른쪽: DQN 추천 (Q값 히트맵) ──────────────────────────────────────────
ax = axes[1]

from matplotlib import colormaps
from matplotlib import cm
from matplotlib.colors import Normalize

# in-zone vs out-of-zone 별도 정규화 + 서로 다른 colormap
q_dict = dict(zip(ZONE_IDS, fastball_q))
inner_qs = np.array([q_dict[z] for z in [1, 2, 3, 4, 5, 6, 7, 8, 9]])
outer_qs = np.array([q_dict[z] for z in [11, 12, 13, 14]])

norm_inner = Normalize(vmin=inner_qs.min(), vmax=inner_qs.max())
norm_outer = Normalize(vmin=outer_qs.min(), vmax=outer_qs.max())
cmap_inner = colormaps["YlOrRd"]    # 존 내부: 노랑→빨강
cmap_outer = colormaps["Blues"]     # 존 외부: 연파랑→진파랑

# 존 외부 L-polygon (Z11~14) — 먼저 그림
for zid, verts in zone_outer_polygons.items():
    q = q_dict[zid]
    color = cmap_outer(norm_outer(q))
    ax.add_patch(Polygon(
        verts, closed=True, facecolor=color, edgecolor="gray",
        linewidth=1.0, zorder=2,
    ))
    # 라벨 위치: L자의 무게중심 대신, 경험적으로 보기 좋은 곳 지정
    label_pos = {
        11: (-1.3, 4.0), 12: (1.3, 4.0),
        13: (-1.3, 1.0), 14: (1.3, 1.0),
    }[zid]
    cx, cz = label_pos
    text_color = "black" if norm_outer(q) < 0.55 else "white"
    ax.text(cx, cz + 0.12, f"Z{zid}", ha="center", va="center",
            fontsize=10, fontweight="bold", zorder=6, color=text_color)
    ax.text(cx, cz - 0.15, f"{q:+.3f}", ha="center", va="center",
            fontsize=9, zorder=6, color=text_color)

# 존 내부 사각형 (Z1~9) — 나중에 그려서 위에 올라가게
for zid, (x0, x1, z0, z1) in zone_inner_rects.items():
    q = q_dict[zid]
    color = cmap_inner(norm_inner(q))
    ax.add_patch(Rectangle(
        (x0, z0), x1 - x0, z1 - z0,
        facecolor=color, edgecolor="gray", linewidth=1.0, zorder=3,
    ))
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    text_color = "black" if norm_inner(q) < 0.55 else "white"
    ax.text(cx, cz + 0.07, f"Z{zid}", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=6, color=text_color)
    ax.text(cx, cz - 0.10, f"{q:+.3f}", ha="center", va="center",
            fontsize=8, zorder=6, color=text_color)

# 스트라이크존 테두리 오버레이
ax.add_patch(Rectangle(
    (SZ_LEFT, SZ_BOTTOM), SZ_RIGHT - SZ_LEFT, SZ_TOP - SZ_BOTTOM,
    fill=False, edgecolor="black", linewidth=2.5, zorder=5,
))

# 두 colorbar: 내부(빨강) + 외부(파랑)
sm_inner = cm.ScalarMappable(cmap=cmap_inner, norm=norm_inner)
sm_inner.set_array([])
cbar_inner = fig.colorbar(sm_inner, ax=ax, shrink=0.4, pad=0.02,
                          location="right", anchor=(0.0, 1.0))
cbar_inner.set_label("존 내부 Q값", fontsize=10)
cbar_inner.ax.tick_params(labelsize=8)

sm_outer = cm.ScalarMappable(cmap=cmap_outer, norm=norm_outer)
sm_outer.set_array([])
cbar_outer = fig.colorbar(sm_outer, ax=ax, shrink=0.4, pad=0.02,
                          location="right", anchor=(0.0, 0.0))
cbar_outer.set_label("존 외부 Q값", fontsize=10)
cbar_outer.ax.tick_params(labelsize=8)

ax.set_title("DQN 추천\n(0-0 카운트, 0아웃, 무주자, 타자군집 3)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("plate_x (ft, 포수 시점)", fontsize=12)
ax.set_ylabel("plate_z (ft)", fontsize=12)
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Z_MIN, Z_MAX)
ax.set_aspect("equal")

fig.suptitle("투구 위치 비교 — 실제 분포 vs DQN 추천 정책",
             fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "pitch_location_comparison.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"  -> {out_path}")
print(f"\n완료! 파일 크기: {os.path.getsize(out_path):,} bytes")
