"""
W&B 실험 결과를 가져와 발표용 시각화 차트를 생성합니다.

대상 런: pitcheezy/SmartPitch-Portfolio/62uwspfk
출력 폴더: docs/presentation_charts/

실행:
    uv run scripts/generate_presentation_charts.py
"""

import os
import wandb
import matplotlib.pyplot as plt
import matplotlib
import platform

# ── 한글 폰트 설정 ───────────────────────────────────────────────────────────
if platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = "AppleGothic"
else:
    matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 설정 ─────────────────────────────────────────────────────────────────────
ENTITY = "pitcheezy"
PROJECT = "SmartPitch-Portfolio"
RUN_ID = "62uwspfk"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "presentation_charts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── W&B 데이터 수집 ─────────────────────────────────────────────────────────
print(f"W&B run {RUN_ID} 데이터 수집 중...")
api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
history = run.history(samples=10000)
summary = dict(run.summary)

print(f"  런 이름: {run.name}")
print(f"  컬럼: {list(history.columns)}")
print(f"  총 {len(history)} 행")

# MLP 메트릭만 추출 (epoch 단위 — NaN이 아닌 행)
mlp_cols = ["epoch", "train_loss", "val_loss", "val_accuracy"]
available_cols = [c for c in mlp_cols if c in history.columns]
df = history[available_cols].dropna(subset=["train_loss"])
df = df.reset_index(drop=True)

# Top-K accuracy 컬럼 탐색
topk_cols = [c for c in history.columns if "top" in c.lower() and "acc" in c.lower()]
print(f"  MLP 메트릭 {len(df)} 행 추출")
print(f"  Top-K 컬럼: {topk_cols}")

# ── 차트 1: Train Loss / Val Loss ────────────────────────────────────────────
print("\n[1/3] Loss 곡선 생성 중...")
fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

ax.plot(df["epoch"], df["train_loss"], marker="o", markersize=4, linewidth=2,
        color="#2196F3", label="Train Loss")
ax.plot(df["epoch"], df["val_loss"], marker="s", markersize=4, linewidth=2,
        color="#FF5722", label="Val Loss")

ax.set_title("MLP 전이 확률 모델 — 학습 손실 곡선", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("CrossEntropy Loss", fontsize=13)
ax.legend(fontsize=12, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_facecolor("white")

fig.tight_layout()
loss_path = os.path.join(OUT_DIR, "loss_curve.png")
fig.savefig(loss_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> {loss_path}")

# ── 차트 2: Val Accuracy ─────────────────────────────────────────────────────
print("[2/3] Accuracy 곡선 생성 중...")
fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

acc_pct = df["val_accuracy"] * 100 if df["val_accuracy"].max() <= 1.0 else df["val_accuracy"]
ax.plot(df["epoch"], acc_pct, marker="o", markersize=5, linewidth=2.5,
        color="#4CAF50", label="Val Accuracy")

ax.set_title("MLP 전이 확률 모델 — 검증 정확도", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.legend(fontsize=12, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_facecolor("white")

best_idx = acc_pct.idxmax()
best_epoch = df.loc[best_idx, "epoch"]
best_acc = acc_pct.max()
acc_range = acc_pct.max() - acc_pct.min()
ax.annotate(f"Best: {best_acc:.1f}% (epoch {int(best_epoch)})",
            xy=(best_epoch, best_acc),
            xytext=(best_epoch - 3, best_acc - acc_range * 0.3),
            fontsize=11, fontweight="bold", color="#2E7D32",
            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.5))

fig.tight_layout()
acc_path = os.path.join(OUT_DIR, "val_accuracy.png")
fig.savefig(acc_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> {acc_path}")

# ── 차트 3: 최종 성능 요약 테이블 ────────────────────────────────────────────
print("[3/3] 성능 요약 테이블 생성 중...")

# 요약 데이터 수집
last_row = df.iloc[-1]
final_train_loss = last_row["train_loss"]
final_val_loss = last_row["val_loss"]
final_val_acc = last_row["val_accuracy"]
if final_val_acc <= 1.0:
    final_val_acc *= 100
total_epochs = int(last_row["epoch"])

# Top-K accuracy (summary에서 가져오기)
topk_data = {}
for col in topk_cols:
    topk_row = history[col].dropna()
    if not topk_row.empty:
        val = topk_row.iloc[-1]
        topk_data[col] = f"{val * 100:.1f}%" if val <= 1.0 else f"{val:.1f}%"

# summary에서 추가 Top-K 탐색
for k, v in summary.items():
    if "top" in k.lower() and "acc" in k.lower() and isinstance(v, (int, float)):
        label = k.replace("_", " ").title()
        topk_data[label] = f"{v * 100:.1f}%" if v <= 1.0 else f"{v:.1f}%"

# 테이블 데이터 구성
table_data = [
    ["모델 아키텍처", "MLP [256, 128, 64]"],
    ["학습 데이터", "2023 MLB 전체 (~72만 투구)"],
    ["클래스 수", "4 (ball / strike / foul / hit_into_play)"],
    ["총 Epoch", str(total_epochs)],
    ["최종 Train Loss", f"{final_train_loss:.4f}"],
    ["최종 Val Loss", f"{final_val_loss:.4f}"],
    ["최종 Val Accuracy", f"{final_val_acc:.1f}%"],
]

for label, val in topk_data.items():
    table_data.append([label, val])

fig, ax = plt.subplots(figsize=(8, max(4, len(table_data) * 0.5 + 1)), facecolor="white")
ax.axis("off")
ax.set_title("범용 전이 확률 모델 — 성능 요약", fontsize=16, fontweight="bold", pad=20)

col_labels = ["항목", "값"]
table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colWidths=[0.5, 0.4],
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.0, 1.8)

# 헤더 스타일
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#1565C0")
    cell.set_text_props(color="white", fontweight="bold")

# 데이터 행 스타일 (짝수/홀수 교대)
for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_facecolor("#F5F5F5" if i % 2 == 0 else "white")
        cell.set_edgecolor("#E0E0E0")

fig.tight_layout()
summary_path = os.path.join(OUT_DIR, "performance_summary.png")
fig.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> {summary_path}")

print(f"\n완료! {OUT_DIR}/ 에 3개 차트 생성:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".png"):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f}  ({size:,} bytes)")
