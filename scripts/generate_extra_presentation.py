"""
generate_extra_presentation.py — 발표용 추가 시각 자료 3종

산출:
  docs/presentation_dqn_pitch_distribution.png  군집별 DQN 구종 분포
  docs/presentation_mdp_improvement.png         MDP 수렴 개선 전후 비교표
  docs/presentation_pipeline_overview.png       3단계 파이프라인 흐름도
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

_DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────
for _fam in ["AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic", "Malgun Gothic"]:
    if any(_fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _fam
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# ── 구종 색상 팔레트 ─────────────────────────────────────────────────────
PITCH_COLORS = {
    "Fastball":    "#3B82F6",
    "Slider":      "#10B981",
    "Curveball":   "#8B5CF6",
    "Changeup":    "#F59E0B",
    "Sinker":      "#6366F1",
    "Cutter":      "#14B8A6",
    "Splitter":    "#EC4899",
    "Sweeper":     "#78716C",
    "Knuckleball": "#EF4444",  # 빨간색 강조
}
OTHER_COLOR = "#D1D5DB"


# ────────────────────────────────────────────────────────────────────────
# 차트 1: 군집별 DQN 구종 분포 — Stacked Horizontal Bar
# ────────────────────────────────────────────────────────────────────────
def chart1_dqn_pitch_distribution():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    clusters = {
        "군집 0\n파워 FB/SL\n(Cole, ~52 actions)": {
            "Fastball": 51.3, "Slider": 24.3, "Curveball": 14.9, "Changeup": 10.7,
        },
        "군집 1\n핀세스\n(117 actions)": {
            "Knuckleball": 45.4, "Fastball": 14.1, "Cutter": 13.2,
            "Changeup": 12.0, "Curveball": 5.5, "Sinker": 5.3,
            "Splitter": 2.3, "Sweeper": 1.4, "Slider": 0.7,
        },
        "군집 2\n싱커볼러\n(117 actions)": {
            "Knuckleball": 36.3, "Fastball": 23.1, "Splitter": 20.3,
            "Slider": 5.8, "Curveball": 4.0, "Cutter": 2.9,
            "Sinker": 2.8, "Sweeper": 2.5, "Changeup": 2.3,
        },
        "군집 3\n멀티피치\n(117 actions)": {
            "Knuckleball": 58.4, "Sinker": 11.2, "Fastball": 8.0,
            "Changeup": 7.4, "Sweeper": 5.6, "Cutter": 3.0,
            "Curveball": 2.9, "Splitter": 2.1, "Slider": 1.6,
        },
    }

    # 모든 구종을 수집 (Knuckleball을 가장 왼쪽에 배치)
    all_pitches_set = set()
    for dist in clusters.values():
        all_pitches_set.update(dist.keys())
    # Knuckleball 먼저, 나머지 알파벳순
    ordered_pitches = []
    if "Knuckleball" in all_pitches_set:
        ordered_pitches.append("Knuckleball")
        all_pitches_set.discard("Knuckleball")
    ordered_pitches.extend(sorted(all_pitches_set))

    labels = list(clusters.keys())
    ys = np.arange(len(labels))

    # Stacked horizontal bar
    lefts = np.zeros(len(labels))
    legend_handles = []

    for pitch in ordered_pitches:
        widths = []
        for cluster_name in labels:
            widths.append(clusters[cluster_name].get(pitch, 0.0))
        widths = np.array(widths)

        color = PITCH_COLORS.get(pitch, OTHER_COLOR)
        edgecolor = '#B91C1C' if pitch == "Knuckleball" else 'white'
        linewidth = 1.5 if pitch == "Knuckleball" else 0.5

        bars = ax.barh(ys, widths, left=lefts, height=0.6,
                       color=color, edgecolor=edgecolor, linewidth=linewidth)

        # 10% 이상인 세그먼트에 라벨 표시
        for i, (w, l) in enumerate(zip(widths, lefts)):
            if w >= 8:
                label_text = f"{pitch}\n{w:.0f}%" if w >= 12 else f"{w:.0f}%"
                fontsize = 8 if w >= 12 else 7
                ax.text(l + w / 2, ys[i], label_text,
                        ha='center', va='center', fontsize=fontsize,
                        fontweight='bold' if pitch == "Knuckleball" else 'normal',
                        color='white' if pitch == "Knuckleball" else '#1F2937')

        lefts += widths
        legend_handles.append(mpatches.Patch(color=color, label=pitch))

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("구종 비율 (%)", fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_title("군집별 DQN 추천 구종 분포 (1000 에피소드 평가)",
                 fontsize=14, fontweight='bold', pad=14)
    ax.legend(handles=legend_handles, loc='upper right', ncol=3,
              fontsize=9, frameon=True, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', color='#e5e7eb', alpha=0.5)
    ax.set_axisbelow(True)

    # 각주
    fig.text(0.5, -0.02,
             "군집 0은 Cole 전용 4구종(~52 actions), 군집 1~3은 범용 9구종(117 actions) · "
             "Knuckleball 편중은 MLP calibration 이슈",
             ha='center', fontsize=9, style='italic', color='#666')

    fig.tight_layout()
    out = os.path.join(_DOCS, "presentation_dqn_pitch_distribution.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[save] {out}")


# ────────────────────────────────────────────────────────────────────────
# 차트 2: MDP 수렴 개선 전후 비교표
# ────────────────────────────────────────────────────────────────────────
def chart2_mdp_improvement():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.axis('off')

    col_labels = ["항목", "개선 전 (VI 5회, γ=1.0)", "개선 후 (VI 17회, γ=0.99)"]
    rows = [
        ["수렴 상태",           "미수렴 (max|ΔV|=0.145)",      "수렴 (max|ΔV|=0.000075)"],
        ["Cluster 0 MDP 보상",  "+0.151 ± 1.264",              "+0.250 ± 1.093"],
        ["vs Random 비교",      "Random(+0.204)보다 낮음",     "Random(+0.204)보다 높음"],
        ["MDP 정책 엔트로피",   "~0 (Knuckleball 70.5%)",      "1.29 (다양한 구종)"],
        ["Knuckleball 비율",    "70.5% (9,216 상태 중 6,501)", "대폭 감소 (물리 피처 적용)"],
        ["물리 피처",           "0 입력 (미적용)",              "lookup 테이블 적용"],
        ["개선 폭",             "—",                           "+0.099 (+65%)"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.2)

    n_cols = len(col_labels)
    n_rows = len(rows)

    # 열 너비 설정
    for i in range(n_rows + 1):
        for j in range(n_cols):
            cell = table[(i, j)]
            if j == 0:
                cell.set_width(0.22)
            else:
                cell.set_width(0.33)

    # 헤더
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor("#1f2937")
        cell.set_text_props(color='white', fontweight='bold', fontsize=12)
        cell.set_edgecolor('#374151')
        cell.set_height(0.10)

    # 행별 스타일
    for i in range(1, n_rows + 1):
        # 항목 열
        cell0 = table[(i, 0)]
        cell0.set_facecolor("#F3F4F6")
        cell0.set_text_props(fontweight='bold', fontsize=11)
        cell0.set_height(0.10)

        # 개선 전 열 (연한 빨간)
        cell1 = table[(i, 1)]
        cell1.set_facecolor("#FEF2F2")
        cell1.set_text_props(color='#991B1B', fontsize=11)
        cell1.set_height(0.10)

        # 개선 후 열 (연한 초록)
        cell2 = table[(i, 2)]
        cell2.set_facecolor("#F0FDF4")
        cell2.set_text_props(color='#166534', fontweight='bold', fontsize=11)
        cell2.set_height(0.10)

    ax.set_title("MDP 수렴 개선 전후 비교 (Task 12P2 + Task 16)",
                 fontsize=14, fontweight='bold', pad=20)

    fig.text(0.5, 0.06,
             "개선 내용: (1) 물리 피처 lookup 테이블 적용 (2) VI 최대 20회 + γ=0.99 + δ<1e-4 조기종료",
             ha='center', fontsize=9, style='italic', color='#555')

    out = os.path.join(_DOCS, "presentation_mdp_improvement.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[save] {out}")


# ────────────────────────────────────────────────────────────────────────
# 차트 3: 3단계 파이프라인 흐름도
# ────────────────────────────────────────────────────────────────────────
def chart3_pipeline_overview():
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # 박스 정의: (x_center, y_center, width, height, color, title, lines)
    boxes = [
        {
            "x": 1.3, "y": 2.5, "w": 2.2, "h": 2.8,
            "color": "#DBEAFE", "border": "#2563EB",
            "title": "Statcast\n데이터",
            "lines": ["2023 MLB", "72만 건", "9구종 × 13존"],
        },
        {
            "x": 4.6, "y": 2.5, "w": 2.2, "h": 2.8,
            "color": "#FEF3C7", "border": "#D97706",
            "title": "MLP\n전이확률 예측",
            "lines": ["4클래스 확률분포", "val_acc 57.5%", "[256,128,64]"],
        },
        {
            "x": 7.9, "y": 2.5, "w": 2.2, "h": 2.8,
            "color": "#D1FAE5", "border": "#059669",
            "title": "MDP\n가치반복",
            "lines": ["9,216 상태", "Value Iteration 17회 수렴", "γ=0.99"],
        },
        {
            "x": 11.2, "y": 2.5, "w": 2.2, "h": 2.8,
            "color": "#FEE2E2", "border": "#DC2626",
            "title": "DQN\n강화학습",
            "lines": ["30만 스텝", "PitchEnv 시뮬", "보상 +0.436"],
        },
    ]

    for box in boxes:
        rect = mpatches.FancyBboxPatch(
            (box["x"] - box["w"]/2, box["y"] - box["h"]/2),
            box["w"], box["h"],
            boxstyle="round,pad=0.12",
            facecolor=box["color"],
            edgecolor=box["border"],
            linewidth=2.0,
        )
        ax.add_patch(rect)

        # 타이틀
        ax.text(box["x"], box["y"] + 0.65, box["title"],
                ha='center', va='center', fontsize=13, fontweight='bold',
                color=box["border"])

        # 설명 줄
        for j, line in enumerate(box["lines"]):
            ax.text(box["x"], box["y"] - 0.15 - j * 0.35, line,
                    ha='center', va='center', fontsize=10, color='#374151')

    # 화살표 + 라벨
    arrows = [
        {"x1": 2.4, "x2": 3.5, "label": "학습 데이터"},
        {"x1": 5.7, "x2": 6.8, "label": "확률 반환 함수"},
        {"x1": 9.0, "x2": 10.1, "label": "RE24 보상"},
    ]

    for arr in arrows:
        ax.annotate("",
                    xy=(arr["x2"], 2.5), xytext=(arr["x1"], 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#6B7280",
                                   lw=2.0, mutation_scale=18))
        ax.text((arr["x1"] + arr["x2"]) / 2, 2.9, arr["label"],
                ha='center', va='bottom', fontsize=9, color='#6B7280',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#D1D5DB', alpha=0.9))

    # 전체 타이틀
    ax.text(7, 4.65, "SmartPitch 3단계 파이프라인",
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1F2937')

    out = os.path.join(_DOCS, "presentation_pipeline_overview.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[save] {out}")


def main():
    os.makedirs(_DOCS, exist_ok=True)
    chart1_dqn_pitch_distribution()
    chart2_mdp_improvement()
    chart3_pipeline_overview()
    print("\n[done]")


if __name__ == "__main__":
    main()
