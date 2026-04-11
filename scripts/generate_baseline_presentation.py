"""
generate_baseline_presentation.py — 발표용 시각 자료 3종

산출:
  docs/presentation_baseline_overall.png   Cole 2019 5-agent 비교 bar
  docs/presentation_baseline_by_cluster.png 4 군집 × 5 에이전트 grouped bar
  docs/presentation_summary_table.png       비교 요약 표 이미지
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASE, '..'))
_DOCS = os.path.join(_ROOT, 'docs')

# ── 한글 폰트 설정 (macOS) ───────────────────────────────────────────────
for _fam in ["AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic", "Malgun Gothic"]:
    if any(_fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _fam
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor']   = 'white'
plt.rcParams['axes.facecolor']     = 'white'

# ── 데이터 (evaluate_baselines.py 실행 결과, 1000 에피소드 / 군집) ───────
# Cluster 0 == Cole 2019 (파워 FB/SL)
AGENTS = ["Random", "MostFrequent", "Frequency", "MDP", "DQN"]
COLORS = {
    "Random":       "#9ca3af",
    "MostFrequent": "#60a5fa",
    "Frequency":    "#34d399",
    "MDP":          "#fbbf24",
    "DQN":          "#ef4444",
}

CLUSTER_LABELS = {
    0: "0\n파워 FB/SL",
    1: "1\n핀세스",
    2: "2\n싱커볼러",
    3: "3\n멀티피치",
}

# {cluster: {agent: (mean, std)}}  NaN mean → 미학습
DATA = {
    0: {
        "Random":       (0.231, 1.098),
        "MostFrequent": (0.151, 1.197),
        "Frequency":    (0.157, 1.203),
        "MDP":          (0.151, 1.264),
        "DQN":          (0.436, 1.255),
    },
    1: {
        "Random":       (0.194, 1.179),
        "MostFrequent": (0.223, 1.124),
        "Frequency":    (0.186, 1.136),
        "MDP":          (0.245, 1.134),
        "DQN":          (np.nan, np.nan),
    },
    2: {
        "Random":       (0.203, 1.138),
        "MostFrequent": (0.249, 1.109),
        "Frequency":    (0.181, 1.129),
        "MDP":          (0.208, 1.158),
        "DQN":          (np.nan, np.nan),
    },
    3: {
        "Random":       (0.176, 1.171),
        "MostFrequent": (0.165, 1.170),
        "Frequency":    (0.216, 1.142),
        "MDP":          (0.036, 1.406),
        "DQN":          (np.nan, np.nan),
    },
}


# ────────────────────────────────────────────────────────────────────────
# 차트 1: 전체 (Cole 2019, cluster 0) 5-agent bar + errorbar
# ────────────────────────────────────────────────────────────────────────
def chart1_overall():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    means = [DATA[0][a][0] for a in AGENTS]
    stds  = [DATA[0][a][1] for a in AGENTS]
    colors = [COLORS[a] for a in AGENTS]

    xs = np.arange(len(AGENTS))
    bars = ax.bar(xs, means, yerr=stds, capsize=6, color=colors,
                  edgecolor='black', linewidth=0.8, error_kw=dict(ecolor='#555', lw=1.2))

    for x, m in zip(xs, means):
        ax.text(x, m + 0.05, f"+{m:.3f}", ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(AGENTS, fontsize=12)
    ax.set_ylabel("평균 에피소드 보상 (RE24 기반)", fontsize=12)
    ax.set_title("베이스라인 비교 — Cole 2019 (pitcher_cluster=0, 1000 에피소드)",
                 fontsize=14, fontweight='bold', pad=14)
    ax.grid(axis='y', linestyle='--', color='#cccccc', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(-1.5, 2.0)

    # DQN 우위 강조
    ax.annotate("DQN이 모든 베이스라인 대비 +0.29 이상 우위",
                xy=(4, 0.436), xytext=(2.3, 1.5),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5))

    fig.tight_layout()
    out = os.path.join(_DOCS, "presentation_baseline_overall.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[save] {out}")


# ────────────────────────────────────────────────────────────────────────
# 차트 2: 군집별 × 에이전트별 grouped bar
# ────────────────────────────────────────────────────────────────────────
def chart2_by_cluster():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    n_clusters = 4
    n_agents = len(AGENTS)
    bar_w = 0.16
    xs = np.arange(n_clusters)

    for i, agent in enumerate(AGENTS):
        means = []
        for c in range(n_clusters):
            m = DATA[c][agent][0]
            means.append(m if not np.isnan(m) else 0.0)
        mask_nan = [np.isnan(DATA[c][agent][0]) for c in range(n_clusters)]
        positions = xs + (i - (n_agents - 1) / 2) * bar_w
        bars = ax.bar(positions, means, width=bar_w, label=agent,
                      color=COLORS[agent], edgecolor='black', linewidth=0.5)
        # 미학습 bar는 빗금 처리 (DQN 군집 1~3)
        for bar, is_nan in zip(bars, mask_nan):
            if is_nan:
                bar.set_height(0)
                bar.set_hatch('///')
                bar.set_alpha(0.25)

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([CLUSTER_LABELS[c] for c in range(n_clusters)], fontsize=11)
    ax.set_ylabel("평균 에피소드 보상 (RE24 기반)", fontsize=12)
    ax.set_xlabel("투수 군집 (K=4)", fontsize=12, labelpad=8)
    ax.set_title("투수 군집별 베이스라인 비교 (각 1000 에피소드)",
                 fontsize=14, fontweight='bold', pad=14)
    ax.legend(loc='upper right', ncol=5, frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', color='#cccccc', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 0.55)

    # 각주
    fig.text(0.5, -0.01,
             "DQN은 군집 0(Cole 2019)만 학습됨. 군집 1~3은 향후 계획.",
             ha='center', fontsize=9, style='italic', color='#555')

    fig.tight_layout()
    out = os.path.join(_DOCS, "presentation_baseline_by_cluster.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[save] {out}")


# ────────────────────────────────────────────────────────────────────────
# 차트 3: 요약 표 이미지
# ────────────────────────────────────────────────────────────────────────
def chart3_summary_table():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.axis('off')

    col_labels = ["투수 군집", *AGENTS]
    cluster_names = {
        0: "0 · 파워 FB/SL (Cole)",
        1: "1 · 핀세스",
        2: "2 · 싱커볼러",
        3: "3 · 멀티피치",
    }

    # 셀 값 + 각 행의 최고값 인덱스 기록
    rows = []
    best_mask = []  # row-wise bool
    for c in range(4):
        row_vals = []
        row_nums = []
        for a in AGENTS:
            m = DATA[c][a][0]
            if np.isnan(m):
                row_vals.append("—")
                row_nums.append(-np.inf)
            else:
                row_vals.append(f"+{m:.3f}")
                row_nums.append(m)
        best_idx = int(np.argmax(row_nums))
        rows.append([cluster_names[c], *row_vals])
        mask = [False] * len(col_labels)
        mask[best_idx + 1] = True
        best_mask.append(mask)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # 헤더 스타일
    for j, _ in enumerate(col_labels):
        cell = table[(0, j)]
        cell.set_facecolor("#1f2937")
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('white')
        cell.set_height(0.12)

    # 첫 컬럼(군집 레이블) 스타일
    for i in range(1, len(rows) + 1):
        cell = table[(i, 0)]
        cell.set_facecolor("#e5e7eb")
        cell.set_text_props(fontweight='bold')
        cell.set_height(0.12)

    # 데이터 셀 + 최고값 bold
    for i, row_mask in enumerate(best_mask, start=1):
        for j in range(1, len(col_labels)):
            cell = table[(i, j)]
            cell.set_height(0.12)
            if row_mask[j]:
                cell.set_facecolor("#fde68a")
                cell.set_text_props(fontweight='bold', color='#1f2937')
            else:
                cell.set_facecolor("#ffffff")

    ax.set_title("베이스라인 비교 요약 — 평균 에피소드 보상 (1000 에피소드, 최고값 강조)",
                 fontsize=14, fontweight='bold', pad=18)

    fig.text(0.5, 0.08,
             "DQN은 군집 0(Cole 2019)만 학습됨 · 표준편차 생략 (전체 std는 docs/baseline_by_cluster.md 참고)",
             ha='center', fontsize=9, style='italic', color='#555')

    out = os.path.join(_DOCS, "presentation_summary_table.png")
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[save] {out}")


def main():
    os.makedirs(_DOCS, exist_ok=True)
    chart1_overall()
    chart2_by_cluster()
    chart3_summary_table()
    print("\n[done]")


if __name__ == "__main__":
    main()
