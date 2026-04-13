"""
generate_baseline_presentation.py — 발표용 시각 자료 3종

산출:
  docs/presentation_baseline_overall.png   Cole 2019 5-agent 가로 막대그래프
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

# ── 데이터 (evaluate_baselines.py + train_dqn_all_clusters.py 결과) ─────
AGENTS = ["Random", "MostFrequent", "Frequency", "MDP", "DQN"]
COLORS = {
    "Random":       "#9CA3AF",
    "MostFrequent": "#9CA3AF",
    "Frequency":    "#9CA3AF",
    "MDP":          "#9CA3AF",
    "DQN":          "#2563EB",
}
COLORS_GROUPED = {
    "Random":       "#9CA3AF",
    "MostFrequent": "#60a5fa",
    "Frequency":    "#34d399",
    "MDP":          "#fbbf24",
    "DQN":          "#2563EB",
}

CLUSTER_LABELS = {
    0: "0\n파워 FB/SL",
    1: "1\n핀세스",
    2: "2\n싱커볼러",
    3: "3\n멀티피치",
}

# {cluster: {agent: (mean, std)}}
# Cluster 0: evaluate_baselines.py (물리피처 lookup + VI 17회 γ=0.99)
# Cluster 1~3: evaluate_baselines.py + train_dqn_all_clusters.py
DATA = {
    0: {
        "Random":       (0.185, 1.177),
        "MostFrequent": (0.220, 1.177),
        "Frequency":    (0.175, 1.123),
        "MDP":          (0.258, 1.091),
        "DQN":          (0.436, 1.255),
    },
    1: {
        "Random":       (0.136, 1.201),
        "MostFrequent": (0.140, 1.188),
        "Frequency":    (0.169, 1.174),
        "MDP":          (0.247, 1.096),
        "DQN":          (0.188, 1.127),
    },
    2: {
        "Random":       (0.229, 1.130),
        "MostFrequent": (0.201, 1.123),
        "Frequency":    (0.203, 1.111),
        "MDP":          (0.262, 1.092),
        "DQN":          (0.242, 1.130),
    },
    3: {
        "Random":       (0.171, 1.194),
        "MostFrequent": (0.251, 1.128),
        "Frequency":    (0.202, 1.158),
        "MDP":          (0.256, 1.072),
        "DQN":          (0.215, 1.157),
    },
}


# ────────────────────────────────────────────────────────────────────────
# 차트 1: 전체 (Cole 2019, cluster 0) — 가로 막대그래프
# ────────────────────────────────────────────────────────────────────────
def chart1_overall():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    # DQN이 위에 오도록 역순 정렬
    agents_sorted = sorted(AGENTS, key=lambda a: DATA[0][a][0])
    means = [DATA[0][a][0] for a in agents_sorted]
    colors = [COLORS[a] for a in agents_sorted]

    ys = np.arange(len(agents_sorted))
    bars = ax.barh(ys, means, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    # 막대 끝에 평균값 표시
    for y, m, bar in zip(ys, means, bars):
        ax.text(m + 0.008, y, f"+{m:.3f}", ha='left', va='center',
                fontsize=12, fontweight='bold',
                color='#2563EB' if agents_sorted[int(y)] == "DQN" else '#374151')

    ax.set_yticks(ys)
    ax.set_yticklabels(agents_sorted, fontsize=13)
    ax.set_xlabel("평균 에피소드 보상 (RE24 기반)", fontsize=12)
    ax.set_title("베이스라인 비교 — Cole 2019 (pitcher_cluster=0, 1000 에피소드)",
                 fontsize=14, fontweight='bold', pad=14)
    ax.grid(axis='x', linestyle='--', color='#e5e7eb', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = os.path.join(_DOCS, "presentation_baseline_overall.png")
    fig.savefig(out, dpi=200, facecolor='white')
    plt.close(fig)
    print(f"[save] {out}")


# ────────────────────────────────────────────────────────────────────────
# 차트 2: 군집별 × 에이전트별 grouped bar
# ────────────────────────────────────────────────────────────────────────
def chart2_by_cluster():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    n_clusters = 4
    n_agents = len(AGENTS)
    bar_w = 0.15
    xs = np.arange(n_clusters)

    for i, agent in enumerate(AGENTS):
        means = [DATA[c][agent][0] for c in range(n_clusters)]
        positions = xs + (i - (n_agents - 1) / 2) * bar_w
        bars = ax.bar(positions, means, width=bar_w, label=agent,
                      color=COLORS_GROUPED[agent], edgecolor='white', linewidth=0.5)
        # 막대 위에 수치 라벨
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                        f"{m:.2f}", ha='center', va='bottom', fontsize=7.5,
                        fontweight='bold' if agent == "DQN" else 'normal',
                        color='#2563EB' if agent == "DQN" else '#374151')

    ax.axhline(0, color='black', linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([CLUSTER_LABELS[c] for c in range(n_clusters)], fontsize=11)
    ax.set_ylabel("평균 에피소드 보상 (RE24 기반)", fontsize=12)
    ax.set_xlabel("투수 군집 (K=4)", fontsize=12, labelpad=8)
    ax.set_title("투수 군집별 베이스라인 비교 (각 1000 에피소드)",
                 fontsize=14, fontweight='bold', pad=14)
    ax.legend(loc='upper right', ncol=5, frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', color='#e5e7eb', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.02, 0.52)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 각주
    fig.text(0.5, -0.01,
             "군집 0 DQN: Cole 2019 전용 ~52 actions / 군집 1: 91 actions (7구종) / 군집 2~3: 104 actions (8구종)",
             ha='center', fontsize=9, style='italic', color='#666')

    fig.tight_layout()
    out = os.path.join(_DOCS, "presentation_baseline_by_cluster.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[save] {out}")


# ────────────────────────────────────────────────────────────────────────
# 차트 3: 요약 표 이미지
# ────────────────────────────────────────────────────────────────────────
def chart3_summary_table():
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    ax.axis('off')

    col_labels = ["투수 군집", *AGENTS]
    cluster_names = {
        0: "0 · 파워 FB/SL (Cole)",
        1: "1 · 핀세스",
        2: "2 · 싱커볼러",
        3: "3 · 멀티피치",
    }

    rows = []
    best_mask = []
    for c in range(4):
        row_vals = []
        row_nums = []
        for a in AGENTS:
            m = DATA[c][a][0]
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
    table.set_fontsize(12)
    table.scale(1.0, 2.2)

    # 열 너비 조정 — 첫 번째 열을 넓게
    for j in range(len(col_labels)):
        for i in range(len(rows) + 1):
            cell = table[(i, j)]
            if j == 0:
                cell.set_width(0.22)
            else:
                cell.set_width(0.13)

    # 헤더 스타일
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor("#1f2937")
        cell.set_text_props(color='white', fontweight='bold', fontsize=12)
        cell.set_edgecolor('white')
        cell.set_height(0.12)

    # 첫 컬럼(군집 레이블) 스타일
    for i in range(1, len(rows) + 1):
        cell = table[(i, 0)]
        cell.set_facecolor("#e5e7eb")
        cell.set_text_props(fontweight='bold', fontsize=11)
        cell.set_height(0.12)

    # 데이터 셀 + 최고값 강조
    for i, row_mask in enumerate(best_mask, start=1):
        for j in range(1, len(col_labels)):
            cell = table[(i, j)]
            cell.set_height(0.12)
            if row_mask[j]:
                cell.set_facecolor("#fde68a")
                cell.set_text_props(fontweight='bold', color='#1f2937', fontsize=12)
            else:
                cell.set_facecolor("#ffffff")
                cell.set_text_props(fontsize=11)

    ax.set_title("베이스라인 비교 요약 — 평균 에피소드 보상 (1000 에피소드, 최고값 강조)",
                 fontsize=14, fontweight='bold', pad=18)

    fig.text(0.5, 0.06,
             "군집 0 DQN: Cole 2019 ~52 actions · 군집 1: 91 actions (7구종) · 군집 2~3: 104 actions (8구종) · 표준편차 생략",
             ha='center', fontsize=9, style='italic', color='#555')

    out = os.path.join(_DOCS, "presentation_summary_table.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
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
