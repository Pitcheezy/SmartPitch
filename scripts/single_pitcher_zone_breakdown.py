"""
단일 투수(Gerrit Cole, 2023 시즌)의 구종별 투구 존 분포 시각화.

출력: docs/presentation_charts/single_pitcher_zone_breakdown.png

실행:
    uv run scripts/single_pitcher_zone_breakdown.py
"""

import os
import platform
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── 한글 폰트 ────────────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = "AppleGothic"
else:
    matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "docs", "presentation_charts")
os.makedirs(OUT_DIR, exist_ok=True)

# 스트라이크존 (MLB 표준)
SZ_LEFT, SZ_RIGHT = -0.71, 0.71
SZ_BOTTOM, SZ_TOP = 1.5, 3.5
X_MIN, X_MAX = -2.0, 2.0
Z_MIN, Z_MAX = 0.3, 4.7

# ═══════════════════════════════════════════════════════════════════════════
# 1. Gerrit Cole 2023 시즌 Statcast 데이터 수집
# ═══════════════════════════════════════════════════════════════════════════
from pybaseball import statcast_pitcher, cache
cache.enable()

PITCHER_ID = 543037   # Gerrit Cole
PITCHER_NAME = "Gerrit Cole"
print(f"[{PITCHER_NAME} (id={PITCHER_ID})] 2023 시즌 데이터 수집 중...")
df = statcast_pitcher("2023-03-30", "2023-10-01", PITCHER_ID)
df = df[df["plate_x"].notna() & df["plate_z"].notna() & df["zone"].notna() & df["pitch_type"].notna()]
print(f"  총 투구 수: {len(df):,}")

# 상위 4개 구종만 (DQN 체계와 동일)
top_pitches = df["pitch_type"].value_counts().head(4)
print(f"  주요 4구종:\n{top_pitches.to_string()}")
pitch_order = top_pitches.index.tolist()

# Statcast pitch_type 코드 → 한국식 이름
PITCH_CODE_KO = {
    "FF": "포심 (Fastball)", "FC": "커터 (Cutter)",
    "SI": "싱커 (Sinker)",   "FT": "투심 (Two-seam)",
    "SL": "슬라이더 (Slider)", "ST": "스위퍼 (Sweeper)",
    "CU": "커브 (Curveball)", "KC": "너클커브 (KCurve)",
    "CH": "체인지업 (Changeup)", "FS": "스플리터 (Splitter)",
    "FA": "Fastball",
}

# ═══════════════════════════════════════════════════════════════════════════
# 2. 구종별 존 분포 시각화 (2×2 grid)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 12), facecolor="white")
axes = axes.flatten()

# 존 라벨 위치 계산 (스트라이크존 3×3 그리드 중심)
sz_w = SZ_RIGHT - SZ_LEFT
sz_h = SZ_TOP - SZ_BOTTOM
col_w = sz_w / 3.0
row_h = sz_h / 3.0
zone_center = {
    1: (SZ_LEFT + col_w * 0.5, SZ_TOP - row_h * 0.5),
    2: (SZ_LEFT + col_w * 1.5, SZ_TOP - row_h * 0.5),
    3: (SZ_LEFT + col_w * 2.5, SZ_TOP - row_h * 0.5),
    4: (SZ_LEFT + col_w * 0.5, SZ_TOP - row_h * 1.5),
    5: (SZ_LEFT + col_w * 1.5, SZ_TOP - row_h * 1.5),
    6: (SZ_LEFT + col_w * 2.5, SZ_TOP - row_h * 1.5),
    7: (SZ_LEFT + col_w * 0.5, SZ_TOP - row_h * 2.5),
    8: (SZ_LEFT + col_w * 1.5, SZ_TOP - row_h * 2.5),
    9: (SZ_LEFT + col_w * 2.5, SZ_TOP - row_h * 2.5),
    11: (-1.3, 4.1), 12: (1.3, 4.1),
    13: (-1.3, 0.9), 14: (1.3, 0.9),
}

# 스트라이크존 내부 3×3 그리드선
grid_xs = [SZ_LEFT, SZ_LEFT + col_w, SZ_LEFT + 2 * col_w, SZ_RIGHT]
grid_zs = [SZ_BOTTOM, SZ_BOTTOM + row_h, SZ_BOTTOM + 2 * row_h, SZ_TOP]

for i, pitch in enumerate(pitch_order):
    ax = axes[i]
    sub = df[df["pitch_type"] == pitch]
    n_pitches = len(sub)

    # hist2d 히트맵
    h = ax.hist2d(
        sub["plate_x"], sub["plate_z"], bins=[30, 30],
        range=[[X_MIN, X_MAX], [Z_MIN, Z_MAX]],
        cmap="YlOrRd", cmin=1,
    )
    cbar = fig.colorbar(h[3], ax=ax, shrink=0.8)
    cbar.set_label("투구 빈도", fontsize=9)

    # 스트라이크존 3×3 그리드선 (회색)
    for gx in grid_xs[1:-1]:
        ax.plot([gx, gx], [SZ_BOTTOM, SZ_TOP], color="gray", lw=0.8, zorder=3)
    for gz in grid_zs[1:-1]:
        ax.plot([SZ_LEFT, SZ_RIGHT], [gz, gz], color="gray", lw=0.8, zorder=3)

    # 스트라이크존 외곽 사각형 (굵게)
    ax.add_patch(Rectangle(
        (SZ_LEFT, SZ_BOTTOM), SZ_RIGHT - SZ_LEFT, SZ_TOP - SZ_BOTTOM,
        fill=False, edgecolor="black", linewidth=2.5, zorder=4,
    ))

    # 존 외부 구분선 (plate_x=0, plate_z=2.5, 점선)
    ax.plot([0, 0], [Z_MIN, SZ_BOTTOM], color="gray", lw=0.6, linestyle="--", zorder=3)
    ax.plot([0, 0], [SZ_TOP, Z_MAX], color="gray", lw=0.6, linestyle="--", zorder=3)
    ax.plot([X_MIN, SZ_LEFT], [2.5, 2.5], color="gray", lw=0.6, linestyle="--", zorder=3)
    ax.plot([SZ_RIGHT, X_MAX], [2.5, 2.5], color="gray", lw=0.6, linestyle="--", zorder=3)

    # 존 번호 라벨 + 실제 빈도
    zone_counts = sub["zone"].value_counts()
    total = zone_counts.sum()
    for zid, (cx, cz) in zone_center.items():
        count = int(zone_counts.get(zid, 0))
        pct = count / total * 100 if total > 0 else 0
        ax.text(cx, cz, f"Z{zid}\n{count}\n({pct:.0f}%)",
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.75, edgecolor="none"))

    ax.set_title(f"{PITCH_CODE_KO.get(pitch, pitch)}  (n={n_pitches:,})",
                 fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("plate_x (ft, 포수 시점)", fontsize=10)
    ax.set_ylabel("plate_z (ft)", fontsize=10)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Z_MIN, Z_MAX)
    ax.set_aspect("equal")

fig.suptitle(f"{PITCHER_NAME} · 2023 시즌 · 구종별 투구 위치 분포 (Statcast zone 1~14)",
             fontsize=16, fontweight="bold", y=0.995)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "single_pitcher_zone_breakdown.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\n저장: {out_path} ({os.path.getsize(out_path):,} bytes)")

# ═══════════════════════════════════════════════════════════════════════════
# 3. 전체 구종 × 존 분포 요약 (콘솔 출력)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"{PITCHER_NAME} 2023 · 구종×존 cross-tabulation")
print("=" * 70)
crosstab = pd.crosstab(
    df[df["pitch_type"].isin(pitch_order)]["pitch_type"],
    df[df["pitch_type"].isin(pitch_order)]["zone"],
    margins=True,
)
print(crosstab.to_string())
