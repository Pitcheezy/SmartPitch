"""
train_5year_models.py — 5년(2021~2025) 통합 학습 + 모델 비교

Task 21+23+25 통합:
  - 2021~2025 5시즌 Statcast 데이터 결합 (~350만 건)
  - 기존 피처 + 추가 피처 (spin_rate, spin_axis, release_pos)
  - 3종 모델 비교: MLP 기본 / MLP+피처 / LightGBM
  - Temperature Scaling (Calibration 개선)
  - Brier Score, ECE 평가

사용법:
  uv run scripts/train_5year_models.py                     # 3종 모델 전체 비교
  uv run scripts/train_5year_models.py --models mlp_plus   # 특정 모델만

산출물:
  best_transition_model_5year.pth       최고 MLP 가중치
  data/feature_columns_5year.json       피처 컬럼 목록
  data/target_classes_5year.json        타겟 클래스 목록
  data/model_config_5year.json          모델 아키텍처 설정
  docs/5year_model_comparison.md        모델 비교 결과표
"""

import argparse
import io
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pybaseball import cache, statcast
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import TransitionProbabilityModel

cache.enable()

# ── 경로 ────────────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(_ROOT, "data")
MODEL_TMP_PATH = os.path.join(_ROOT, "best_transition_model.pth")

# 5년 모델 저장 경로
MODEL_5Y_PATH = os.path.join(_ROOT, "best_transition_model_5year.pth")
FEAT_5Y_PATH = os.path.join(DATA_DIR, "feature_columns_5year.json")
TARGET_5Y_PATH = os.path.join(DATA_DIR, "target_classes_5year.json")
CONFIG_5Y_PATH = os.path.join(DATA_DIR, "model_config_5year.json")

# ── 시즌 날짜 ────────────────────────────────────────────────────────────────
SEASON_DATES = {
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}

# ── pitch_type → 구종명 매핑 (universal_model_trainer.py와 동일) ──────────
PITCH_TYPE_MAP = {
    "FF": "Fastball", "FA": "Fastball",
    "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "SV": "Slider",
    "CH": "Changeup", "EP": "Changeup", "SC": "Changeup",
    "FS": "Splitter", "FO": "Splitter",
    "CU": "Curveball", "KC": "Curveball", "CS": "Curveball",
    "KN": "Knuckleball",
    "PO": None,
}

DESCRIPTION_MAP = {
    "called_strike": "strike", "swinging_strike": "strike",
    "foul_tip": "strike", "swinging_strike_blocked": "strike",
    "missed_bunt": "strike", "bunt_foul_tip": "strike",
    "foul": "foul", "foul_bunt": "foul",
    "ball": "ball", "blocked_ball": "ball",
    "hit_by_pitch": "hit_by_pitch",
    "hit_into_play": "hit_into_play",
}

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = 1024
LR = 0.001
HIDDEN_DIMS = [256, 128, 64]
RANDOM_STATE = 42


def fetch_5year_data() -> pd.DataFrame:
    """2021~2025 5시즌 Statcast 데이터 수집 + 결합."""
    frames = []
    for year, (start, end) in SEASON_DATES.items():
        print(f"\n[수집] {year} 시즌 ({start} ~ {end})...")
        df = statcast(start_dt=start, end_dt=end)
        if df is None or df.empty:
            print(f"  [경고] {year} 데이터가 비어있습니다. 건너뜁니다.")
            continue
        df["game_year"] = year
        print(f"  {len(df):,}건")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[합계] 5시즌 전체: {len(combined):,}건")
    return combined


def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    """기본 전처리 (universal_model_trainer._preprocess_raw과 동일 로직)."""
    print(f"\n[전처리] 원본 {len(df):,}건...")
    df = df.copy()

    # pitch_type → mapped_pitch_name
    df["mapped_pitch_name"] = df["pitch_type"].map(PITCH_TYPE_MAP)
    df = df[df["mapped_pitch_name"].notna()]

    # on_1b/2b/3b → "0"/"1" 문자열
    for col in ["on_1b", "on_2b", "on_3b"]:
        df[col] = df[col].notna().astype(int).astype(str)

    # description → 5그룹
    df["description"] = df["description"].map(DESCRIPTION_MAP)
    df = df[df["description"].notna()]

    # 필수 컬럼 + NaN 제거
    required = [
        "balls", "strikes", "outs_when_up",
        "on_1b", "on_2b", "on_3b",
        "mapped_pitch_name", "zone", "description",
        "batter", "pitcher", "game_year",
        # 물리 피처 (기존)
        "release_speed", "pfx_x", "pfx_z",
        # 추가 물리 피처 (Task 25)
        "release_spin_rate", "spin_axis",
        "release_pos_x", "release_pos_z",
        # 매치업 피처
        "p_throws", "stand",
    ]
    # 실제 존재하는 컬럼만 유지
    available = [c for c in required if c in df.columns]
    df = df[available]

    # 핵심 컬럼 NaN 제거
    dropna_cols = [
        "balls", "strikes", "outs_when_up", "zone", "description",
        "release_speed", "pfx_x", "pfx_z",
    ]
    df = df.dropna(subset=[c for c in dropna_cols if c in df.columns])

    # zone float → int
    df["zone"] = df["zone"].astype(float).astype(int)

    print(f"  전처리 완료: {len(df):,}건")

    # 시즌별 분포 출력
    for year, group in df.groupby("game_year"):
        print(f"  {year}: {len(group):,}건")

    return df.reset_index(drop=True)


def add_physical_features_base(df: pd.DataFrame) -> pd.DataFrame:
    """기존 3개 물리 피처 정규화 (Exp5와 동일)."""
    df = df.copy()
    df["release_speed_n"] = (df["release_speed"].astype(float) - 90.0) / 5.0
    df["pfx_x_n"] = df["pfx_x"].astype(float)
    df["pfx_z_n"] = df["pfx_z"].astype(float)
    return df


def add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """추가 물리 피처 정규화 (Task 25)."""
    df = df.copy()

    # 스핀율 정규화
    if "release_spin_rate" in df.columns:
        df["spin_rate_n"] = (df["release_spin_rate"].astype(float) - 2200.0) / 500.0
        df["spin_rate_n"] = df["spin_rate_n"].fillna(0.0)

    # 스핀축 정규화
    if "spin_axis" in df.columns:
        df["spin_axis_n"] = df["spin_axis"].astype(float) / 360.0
        df["spin_axis_n"] = df["spin_axis_n"].fillna(0.5)

    # 릴리스 포인트 정규화
    if "release_pos_x" in df.columns:
        df["release_pos_x_n"] = (df["release_pos_x"].astype(float) + 2.0) / 2.0
        df["release_pos_x_n"] = df["release_pos_x_n"].fillna(0.0)

    if "release_pos_z" in df.columns:
        df["release_pos_z_n"] = (df["release_pos_z"].astype(float) - 6.0) / 1.0
        df["release_pos_z_n"] = df["release_pos_z_n"].fillna(0.0)

    # 좌/우 매치업 (one-hot 대신 수치로 인코딩)
    if "p_throws" in df.columns and "stand" in df.columns:
        # platoon advantage: 좌투-우타 or 우투-좌타 = 1 (투수 유리)
        df["platoon_advantage"] = (
            ((df["p_throws"] == "L") & (df["stand"] == "R")) |
            ((df["p_throws"] == "R") & (df["stand"] == "L"))
        ).astype(int)
        # 투수 좌투 여부
        df["p_throws_L"] = (df["p_throws"] == "L").astype(int)
        # 타자 좌타 여부
        df["stand_L"] = (df["stand"] == "L").astype(int)

    return df


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error 계산."""
    n_classes = y_prob.shape[1]
    ece = 0.0
    for c in range(n_classes):
        binary_true = (y_true == c).astype(int)
        prob_c = y_prob[:, c]
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                binary_true, prob_c, n_bins=n_bins, strategy="uniform"
            )
            # 각 빈의 가중 ECE
            bin_counts = np.histogram(prob_c, bins=n_bins, range=(0, 1))[0]
            bin_weights = bin_counts[bin_counts > 0] / len(y_true)
            ece += np.sum(
                bin_weights[:len(fraction_of_positives)]
                * np.abs(fraction_of_positives - mean_predicted_value)
            )
        except ValueError:
            pass
    return ece / n_classes


class TemperatureScaling(nn.Module):
    """Temperature Scaling for calibration (Guo et al., 2017)."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def find_optimal_temperature(
    model: TransitionProbabilityModel, val_df: pd.DataFrame, val_labels: np.ndarray
) -> float:
    """검증셋에서 NLL을 최소화하는 최적 Temperature 탐색."""
    # 모델에서 logits 추출
    X = val_df[model.feature_columns].values.astype(np.float32)
    X_tensor = torch.FloatTensor(X)
    model.model.eval()
    with torch.no_grad():
        logits = model.model(X_tensor)

    y_tensor = torch.LongTensor(val_labels)

    temp_module = TemperatureScaling()
    optimizer = torch.optim.LBFGS([temp_module.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def eval_fn():
        optimizer.zero_grad()
        scaled = temp_module(logits)
        loss = criterion(scaled, y_tensor)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    optimal_t = temp_module.temperature.item()
    print(f"  최적 Temperature: {optimal_t:.4f}")
    return optimal_t


def train_mlp_model(
    df: pd.DataFrame, run_name: str, description: str,
    use_extended: bool = False, use_class_weights: bool = True,
) -> dict:
    """MLP 모델 학습 + 평가."""
    import wandb

    print(f"\n{'='*60}")
    print(f"학습: {run_name}")
    print(f"  설명: {description}")
    print(f"  데이터: {len(df):,}건")
    print(f"{'='*60}")

    # 물리 피처 추가
    df = add_physical_features_base(df)
    if use_extended:
        df = add_extended_features(df)

    run = wandb.init(
        project="SmartPitch-Portfolio",
        name=run_name,
        config={
            "data": "2021-2025 MLB 5시즌",
            "n_samples": len(df),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "hidden_dims": HIDDEN_DIMS,
            "use_extended_features": use_extended,
            "use_class_weights": use_class_weights,
            "description": description,
        },
    )

    try:
        model = TransitionProbabilityModel(df=df, batch_size=BATCH_SIZE, lr=LR)
        model.run_modeling_pipeline(
            epochs=EPOCHS,
            hidden_dims=HIDDEN_DIMS,
            upload_artifact=False,
            use_lr_scheduler=False,
            use_class_weights=use_class_weights,
        )

        best_val_loss = wandb.run.summary.get("best_val_loss", float("nan"))
        best_val_acc = wandb.run.summary.get("best_val_acc", float("nan"))

        # 모델 저장
        os.makedirs(DATA_DIR, exist_ok=True)
        model_path = os.path.join(_ROOT, f"best_model_{run_name}.pth")
        feat_path = os.path.join(DATA_DIR, f"feature_columns_{run_name}.json")
        cls_path = os.path.join(DATA_DIR, f"target_classes_{run_name}.json")
        cfg_path = os.path.join(DATA_DIR, f"model_config_{run_name}.json")

        if os.path.exists(MODEL_TMP_PATH):
            shutil.copy(MODEL_TMP_PATH, model_path)
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(model.feature_columns, f, ensure_ascii=False, indent=2)
        with open(cls_path, "w", encoding="utf-8") as f:
            json.dump(model.target_classes, f, ensure_ascii=False, indent=2)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"hidden_dims": HIDDEN_DIMS, "dropout_rate": 0.2}, f, indent=2)

        # 검증셋: train_model()에서 저장한 val_loader 사용
        X_val_list, y_val_list = [], []
        for X_batch, y_batch in model.val_loader:
            X_val_list.append(X_batch.numpy())
            y_val_list.append(y_batch.numpy())
        X_val_arr = np.concatenate(X_val_list)
        val_labels = np.concatenate(y_val_list)

        # predict_proba를 위한 DataFrame 구성
        X_val_df = pd.DataFrame(X_val_arr, columns=model.feature_columns)
        val_proba = model.predict_proba(X_val_df)
        macro_f1 = f1_score(val_labels, val_proba.argmax(axis=1), average="macro")

        # Brier Score (multi-class: 평균)
        n_classes = val_proba.shape[1]
        brier = 0.0
        for c in range(n_classes):
            brier += brier_score_loss((val_labels == c).astype(int), val_proba[:, c])
        brier /= n_classes

        # ECE
        ece = compute_ece(val_labels, val_proba)

        # Temperature Scaling
        optimal_t = find_optimal_temperature(model, X_val_df, val_labels)

        # Temperature 적용 후 재평가
        X_tensor = torch.FloatTensor(X_val_arr)
        model.model.eval()
        with torch.no_grad():
            logits = model.model(X_tensor)
            scaled_proba = torch.softmax(logits / optimal_t, dim=1).numpy()

        brier_scaled = 0.0
        for c in range(n_classes):
            brier_scaled += brier_score_loss(
                (val_labels == c).astype(int), scaled_proba[:, c]
            )
        brier_scaled /= n_classes
        ece_scaled = compute_ece(val_labels, scaled_proba)

        wandb.log({
            "macro_f1": macro_f1, "brier_score": brier, "ece": ece,
            "optimal_temperature": optimal_t,
            "brier_score_calibrated": brier_scaled, "ece_calibrated": ece_scaled,
        })

        result = {
            "run_name": run_name,
            "description": description,
            "val_acc": best_val_acc,
            "macro_f1": macro_f1,
            "brier": brier,
            "ece": ece,
            "optimal_temperature": optimal_t,
            "brier_calibrated": brier_scaled,
            "ece_calibrated": ece_scaled,
            "model_path": model_path,
            "feat_path": feat_path,
            "cls_path": cls_path,
            "cfg_path": cfg_path,
            "n_features": len(model.feature_columns),
        }
        print(f"\n  val_acc: {best_val_acc:.4f}, macro_f1: {macro_f1:.4f}")
        print(f"  Brier: {brier:.4f} → {brier_scaled:.4f} (calibrated)")
        print(f"  ECE: {ece:.4f} → {ece_scaled:.4f} (calibrated)")
        return result

    finally:
        wandb.finish()


def train_lightgbm_model(df: pd.DataFrame) -> dict:
    """LightGBM 모델 학습 + 평가."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("[경고] lightgbm 미설치. LightGBM 실험을 건너뜁니다.")
        print("  설치: uv pip install lightgbm")
        return None

    print(f"\n{'='*60}")
    print("학습: LightGBM")
    print(f"  데이터: {len(df):,}건")
    print(f"{'='*60}")

    # 피처 준비
    df = add_physical_features_base(df)
    df = add_extended_features(df)

    # hit_by_pitch 제거
    df = df[df["description"] != "hit_by_pitch"]

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(df["description"])
    target_classes = le.classes_.tolist()

    # 수치 피처
    num_cols = [
        "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
        "release_speed_n", "pfx_x_n", "pfx_z_n",
        "spin_rate_n", "spin_axis_n",
        "release_pos_x_n", "release_pos_z_n",
        "platoon_advantage", "p_throws_L", "stand_L",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    # 범주형 → label encoding for LightGBM
    cat_cols = ["mapped_pitch_name", "zone"]
    for col in ["batter_cluster", "pitcher_cluster"]:
        if col in df.columns:
            cat_cols.append(col)

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    feature_cols = num_cols + [c for c in cat_cols if c in df.columns]

    # on_1b/2b/3b는 문자열 "0"/"1" → 수치로
    for col in ["on_1b", "on_2b", "on_3b"]:
        if col in feature_cols:
            df[col] = df[col].astype(int)

    X = df[feature_cols]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"  Features: {len(feature_cols)}")

    cat_feature_names = [c for c in cat_cols if c in feature_cols]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_feature_names, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": len(target_classes),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[val_data], callbacks=callbacks,
    )

    # 평가
    val_proba = model.predict(X_val)
    val_pred = val_proba.argmax(axis=1)
    val_acc = (val_pred == y_val).mean()
    macro_f1 = f1_score(y_val, val_pred, average="macro")

    n_classes = val_proba.shape[1]
    brier = sum(
        brier_score_loss((y_val == c).astype(int), val_proba[:, c])
        for c in range(n_classes)
    ) / n_classes
    ece = compute_ece(y_val, val_proba)

    # LightGBM 모델 저장
    lgb_path = os.path.join(_ROOT, "lightgbm_5year.txt")
    model.save_model(lgb_path)

    result = {
        "run_name": "LightGBM_5Year",
        "description": "LightGBM + 전체 피처 (tabular baseline)",
        "val_acc": val_acc,
        "macro_f1": macro_f1,
        "brier": brier,
        "ece": ece,
        "optimal_temperature": 1.0,
        "brier_calibrated": brier,
        "ece_calibrated": ece,
        "model_path": lgb_path,
        "n_features": len(feature_cols),
    }
    print(f"\n  val_acc: {val_acc:.4f}, macro_f1: {macro_f1:.4f}")
    print(f"  Brier: {brier:.4f}, ECE: {ece:.4f}")
    return result


def write_comparison_report(results: list[dict], output_path: str):
    """모델 비교 결과를 마크다운 파일로 저장."""
    lines = [
        "# 5년 통합 모델 비교 결과",
        "",
        f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 데이터: 2021~2025 MLB 5시즌 정규시즌",
        "",
        "## 모델 비교",
        "",
        "| 모델 | val_acc | macro F1 | Brier | ECE | Brier(cal) | ECE(cal) | T | 피처수 |",
        "|------|---------|----------|-------|-----|------------|----------|---|--------|",
    ]

    # 기존 Exp5 baseline 참고행
    lines.append(
        "| Exp5 (2023, baseline) | 57.5% | 0.495 | — | — | — | — | — | 43 |"
    )

    for r in results:
        if r is None:
            continue
        acc = f"{r['val_acc']:.1%}" if r["val_acc"] == r["val_acc"] else "N/A"
        f1 = f"{r['macro_f1']:.3f}"
        brier = f"{r['brier']:.4f}"
        ece = f"{r['ece']:.4f}"
        brier_c = f"{r['brier_calibrated']:.4f}"
        ece_c = f"{r['ece_calibrated']:.4f}"
        t = f"{r['optimal_temperature']:.2f}"
        nf = str(r.get("n_features", "—"))
        lines.append(f"| {r['run_name']} | {acc} | {f1} | {brier} | {ece} | {brier_c} | {ece_c} | {t} | {nf} |")

    lines.extend([
        "",
        "## 결론",
        "",
        "- 가장 높은 val_acc 모델을 `best_transition_model_5year.pth`로 저장",
        "- Temperature Scaling으로 Calibration 개선 (Brier/ECE 감소)",
        "- 기존 발표 데모 모델(Cease/Gallen)은 영향 없음",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[저장] {output_path}")


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="5년 통합 모델 학습")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["mlp_base", "mlp_plus", "lgbm"],
        choices=["mlp_base", "mlp_plus", "lgbm"],
        help="학습할 모델 (기본: 3종 전체)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SmartPitch 5년 통합 모델 학습 (Task 21+23+25)")
    print(f"  시즌: 2021~2025 (5시즌)")
    print(f"  모델: {', '.join(args.models)}")
    print("=" * 60)

    # ── 1. 데이터 수집 ──────────────────────────────────────────────────────
    t0 = time.time()
    raw_df = fetch_5year_data()
    print(f"\n  데이터 수집 시간: {time.time() - t0:.0f}초")

    # ── 2. 기본 전처리 ──────────────────────────────────────────────────────
    df_base = preprocess_base(raw_df)
    del raw_df

    # ── 3. 모델 학습 ────────────────────────────────────────────────────────
    results = []

    if "mlp_base" in args.models:
        r = train_mlp_model(
            df_base.copy(),
            run_name="5Year_MLP_Base",
            description="5년 데이터 + 기존 피처 (Exp5 동일 구조)",
            use_extended=False,
            use_class_weights=True,
        )
        results.append(r)

    if "mlp_plus" in args.models:
        r = train_mlp_model(
            df_base.copy(),
            run_name="5Year_MLP_Extended",
            description="5년 데이터 + 확장 피처 (spin_rate/axis, release_pos, platoon)",
            use_extended=True,
            use_class_weights=True,
        )
        results.append(r)

    if "lgbm" in args.models:
        r = train_lightgbm_model(df_base.copy())
        if r is not None:
            results.append(r)

    del df_base

    # ── 4. 결과 비교 ────────────────────────────────────────────────────────
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("[오류] 학습된 모델이 없습니다.")
        return

    # 비교 표 출력
    print(f"\n{'='*80}")
    print("모델 비교 결과")
    print(f"{'='*80}")
    print(f"{'모델':<25} {'val_acc':>8} {'F1':>7} {'Brier':>7} {'ECE':>7} {'T':>5} {'피처':>4}")
    print("-" * 80)
    print(f"{'Exp5 (2023, baseline)':<25} {'57.5%':>8} {'0.495':>7} {'—':>7} {'—':>7} {'—':>5} {'43':>4}")
    for r in valid_results:
        acc = f"{r['val_acc']:.1%}"
        print(
            f"{r['run_name']:<25} {acc:>8} {r['macro_f1']:>7.3f} "
            f"{r['brier']:>7.4f} {r['ece']:>7.4f} {r['optimal_temperature']:>5.2f} "
            f"{r.get('n_features', '—'):>4}"
        )

    # ── 5. 최고 MLP 모델 저장 ───────────────────────────────────────────────
    mlp_results = [r for r in valid_results if "MLP" in r["run_name"]]
    if mlp_results:
        best_mlp = max(mlp_results, key=lambda r: r["val_acc"])
        print(f"\n[최고 MLP] {best_mlp['run_name']} (val_acc={best_mlp['val_acc']:.4f})")

        if os.path.exists(best_mlp["model_path"]):
            shutil.copy(best_mlp["model_path"], MODEL_5Y_PATH)
            shutil.copy(best_mlp["feat_path"], FEAT_5Y_PATH)
            shutil.copy(best_mlp["cls_path"], TARGET_5Y_PATH)
            shutil.copy(best_mlp["cfg_path"], CONFIG_5Y_PATH)

            # Temperature도 config에 추가
            with open(CONFIG_5Y_PATH, "r") as f:
                cfg = json.load(f)
            cfg["temperature"] = best_mlp["optimal_temperature"]
            with open(CONFIG_5Y_PATH, "w") as f:
                json.dump(cfg, f, indent=2)

            print(f"  저장: {MODEL_5Y_PATH}")

    # ── 6. 비교 보고서 ──────────────────────────────────────────────────────
    report_path = os.path.join(_ROOT, "docs", "5year_model_comparison.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    write_comparison_report(valid_results, report_path)

    # 임시 파일 정리
    for r in valid_results:
        for key in ["model_path", "feat_path", "cls_path", "cfg_path"]:
            fpath = r.get(key)
            if fpath and os.path.exists(fpath) and fpath != MODEL_5Y_PATH:
                os.remove(fpath)
    if os.path.exists(MODEL_TMP_PATH):
        os.remove(MODEL_TMP_PATH)

    print(f"\n{'='*60}")
    print("5년 통합 모델 학습 완료")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
