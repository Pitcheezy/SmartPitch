"""
W&B 런 데이터를 로컬에 내려받아 분석용 CSV/JSON으로 저장합니다.
대상 런: pitcheezy/SmartPitch-Portfolio/cuafju1e
"""

import wandb
import pandas as pd
import json
import os

ENTITY  = "pitcheezy"
PROJECT = "SmartPitch-Portfolio"
RUN_ID  = "cuafju1e"
OUT_DIR = "wandb_export"

os.makedirs(OUT_DIR, exist_ok=True)

api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

print(f"런 이름  : {run.name}")
print(f"상태     : {run.state}")
print(f"생성일   : {run.created_at}")
print()

# ── 1. 하이퍼파라미터 config ──────────────────────────────────────────────
print("=== [1] Config (하이퍼파라미터) ===")
config = dict(run.config)
for k, v in config.items():
    print(f"  {k}: {v}")
with open(f"{OUT_DIR}/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"  → {OUT_DIR}/config.json 저장 완료\n")

# ── 2. 학습 메트릭 히스토리 ───────────────────────────────────────────────
print("=== [2] 학습 메트릭 히스토리 ===")
history = run.history(samples=10000)  # 최대 10,000 스텝
print(f"  총 {len(history)} 행, 컬럼: {list(history.columns)}")
history.to_csv(f"{OUT_DIR}/metrics_history.csv", index=False)
print(f"  → {OUT_DIR}/metrics_history.csv 저장 완료\n")

# ── 3. 최종 Summary (최고/최종 지표) ─────────────────────────────────────
print("=== [3] 최종 Summary ===")
summary = dict(run.summary)
for k, v in summary.items():
    if not k.startswith("_"):
        print(f"  {k}: {v}")
with open(f"{OUT_DIR}/summary.json", "w", encoding="utf-8") as f:
    # wandb 내부 객체 직렬화 불가 항목 제거
    clean = {k: v for k, v in summary.items()
             if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, dict, type(None)))}
    json.dump(clean, f, indent=2, ensure_ascii=False)
print(f"  → {OUT_DIR}/summary.json 저장 완료\n")

# ── 4. W&B Table (정책 테이블 등) ────────────────────────────────────────
print("=== [4] Logged Tables (정책 / 구종 분포 등) ===")
try:
    artifacts = run.logged_artifacts()
    for artifact in artifacts:
        print(f"  Artifact: {artifact.name} (type={artifact.type})")
except Exception as e:
    print(f"  Artifact 목록 조회 실패: {e}")

# 테이블 형태로 로깅된 데이터는 history에서 추출
table_cols = [c for c in history.columns if "table" in c.lower() or "policy" in c.lower() or "pitch" in c.lower()]
print(f"  테이블 관련 컬럼: {table_cols}")

# ── 5. 주요 메트릭 빠른 요약 ─────────────────────────────────────────────
print("\n=== [5] 주요 메트릭 빠른 요약 ===")
metric_groups = {
    "MLP 학습": ["train_loss", "val_loss", "val_accuracy", "epoch"],
    "DQN 학습": ["dqn/episode_reward", "dqn/episode_length", "dqn/exploration_rate"],
    "DQN 평가": ["dqn_eval/mean_reward", "dqn_eval/std_reward"],
}

for group, cols in metric_groups.items():
    available = [c for c in cols if c in history.columns]
    if available:
        print(f"\n  [{group}]")
        subset = history[available].dropna()
        if not subset.empty:
            print(subset.describe().round(4).to_string())
            subset.to_csv(f"{OUT_DIR}/{group.replace(' ', '_')}.csv", index=False)

print(f"\n모든 데이터가 '{OUT_DIR}/' 폴더에 저장되었습니다.")
print("파일 목록:")
for f in os.listdir(OUT_DIR):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f}  ({size:,} bytes)")
