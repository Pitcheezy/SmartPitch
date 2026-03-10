"""
SmartPitch 통합 파이프라인

1) 01 preprocess, 02 embedding, 03 profiles, 04 matchup (데이터 파이프라인)
2) 상태 전이 확률 예측 모델 학습 → MDP 최적 볼배합 도출 → DQN 강화학습 에이전트 학습
"""
from __future__ import annotations

import os
import sys
import io
from pathlib import Path

# Windows 콘솔 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

# 프로젝트 루트 경로
_proj_root = Path(__file__).resolve().parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import pandas as pd
import wandb

# 데이터 파이프라인 모듈
from src.io_utils import get_paths, log, save_parquet, read_parquet
from src.preprocess import preprocess_statcast
from src.embedding import run_pitcher_umap_cluster, EmbeddingConfig
from src.profiles import build_pitcher_profiles, build_batter_profiles
from src.matchup import build_matchup_tables, MatchupConfig
from src.fetch import fetch_statcast_by_date, FetchConfig

# 모델 및 RL 모듈
from src.model import TransitionProbabilityModel
from src.mdp_solver import MDPOptimizer
from src.pitch_env import PitchEnv
from src.rl_trainer import DQNTrainer


def get_pitcher_id(first_name: str, last_name: str) -> int:
    """pybaseball으로 투수 MLBAM ID 조회"""
    from pybaseball import playerid_lookup

    lookup = playerid_lookup(last_name, first_name)
    if lookup is None or len(lookup) == 0:
        raise ValueError(f"투수를 찾을 수 없습니다: {first_name} {last_name}")
    # 가장 최근 활동한 선수 우선 (key_mlbam 사용)
    pid = lookup.iloc[0]["key_mlbam"]
    if pd.isna(pid):
        raise ValueError(f"MLBAM ID를 찾을 수 없습니다: {first_name} {last_name}")
    return int(pid)


def prepare_df_for_model(df_pitcher: pd.DataFrame) -> pd.DataFrame:
    """
    embedding 출력을 TransitionProbabilityModel 입력 형식으로 변환.
    - mapped_pitch_name: pitch_cluster_id (또는 pitch_type fallback)
    - on_1b, on_2b, on_3b: count_state용 "0"/"1" 문자열
    """
    df = df_pitcher.copy()

    # pitch_cluster_id가 없으면 pitch_type으로 대체
    if "pitch_cluster_id" not in df.columns:
        df["pitch_cluster_id"] = df["pitcher"].astype(str) + "_" + df["pitch_type"].fillna("UNK").astype(str)
    df["mapped_pitch_name"] = df["pitch_cluster_id"].astype(str)

    # count_state용 on_1b, on_2b, on_3b를 "0"/"1" 문자열로
    for c in ["on_1b", "on_2b", "on_3b"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int).astype(str)

    # zone NaN 제거 (모델에서 필요)
    df = df.dropna(subset=["zone"]).copy()
    df["zone"] = pd.to_numeric(df["zone"], errors="coerce")
    df = df.dropna(subset=["zone"]).copy()

    return df


def run_data_pipeline(start_date: str, end_date: str, project_root: Path) -> tuple[pd.DataFrame, Path]:
    """
    00 fetch → 01 preprocess → 02 embedding → 03 profiles → 04 matchup 실행.
    반환: (df_umap_cluster, processed_dir)
    """
    tag = f"{start_date}_to_{end_date}"
    paths = get_paths(project_root)

    raw_csv = paths.raw_csv_range(start_date, end_date)
    out_clean = paths.processed_pitch_clean_range(start_date, end_date)
    out_embed = paths.processed_pitch_umap_cluster_range(start_date, end_date)
    out_pitcher_prof = paths.processed_pitcher_profiles_range(start_date, end_date)
    out_batter_prof = paths.processed_batter_profiles_range(start_date, end_date)
    out_pitch_level = paths.processed_matchup_pitch_level_range(start_date, end_date)
    out_pair_level = paths.processed_matchup_pair_level_range(start_date, end_date)
    out_summary = paths.processed_dir / f"pitcher_cluster_summary_{tag}.csv"

    # 00 fetch
    if not raw_csv.exists():
        log("00 fetch: Statcast 데이터 수집 중...")
        df_raw = fetch_statcast_by_date(start_date, end_date, FetchConfig())
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(raw_csv, index=False, encoding="utf-8-sig")
        log(f"00 fetch 완료: {raw_csv}")
    else:
        log(f"00 fetch: 기존 파일 사용 ({raw_csv})")
        df_raw = pd.read_csv(raw_csv, low_memory=False)

    # 01 preprocess
    log("01 preprocess: 데이터 정제 중...")
    df_clean = preprocess_statcast(df_raw)
    save_parquet(df_clean, out_clean)
    log(f"01 preprocess 완료: {out_clean}")

    # 02 embedding
    log("02 embedding: UMAP·HDBSCAN 클러스터링 중...")
    cfg = EmbeddingConfig()
    df_emb, summary = run_pitcher_umap_cluster(df_clean, cfg)
    save_parquet(df_emb, out_embed)
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    log(f"02 embedding 완료: {out_embed}")

    # 03 profiles
    log("03 profiles: 투수·타자 프로필 생성 중...")
    pitcher_profiles = build_pitcher_profiles(df_emb, summary)
    batter_profiles = build_batter_profiles(df_emb)
    save_parquet(pitcher_profiles, out_pitcher_prof)
    save_parquet(batter_profiles, out_batter_prof)
    log(f"03 profiles 완료")

    # 04 matchup
    log("04 matchup: 매치업 테이블 생성 중...")
    pitch_level, pair_level, _ = build_matchup_tables(
        df_emb, pitcher_profiles, batter_profiles, MatchupConfig(topk=5)
    )
    save_parquet(pitch_level, out_pitch_level)
    save_parquet(pair_level, out_pair_level)
    log(f"04 matchup 완료")

    return df_emb, paths.processed_dir


def main(player_first_name: str, player_last_name: str, start_date: str, end_date: str) -> None:
    project_root = _proj_root

    print("\n[System] 파이프라인 실행을 시작합니다...")
    print("=" * 60)
    print("SmartPitch 파이프라인 실행 시작")
    print("=" * 60)

    # 투수 ID 조회
    log(f"투수 조회: {player_first_name} {player_last_name}")
    pitcher_id = get_pitcher_id(player_first_name, player_last_name)
    log(f"  → MLBAM ID: {pitcher_id}")

    # -------------------------------------------------------------
    # 1. W&B 초기화
    # -------------------------------------------------------------
    run = wandb.init(
        project="SmartPitch-Portfolio",
        name=f"{player_first_name.capitalize()}_{player_last_name.capitalize()}_Pipeline",
        config={
            "pitcher": f"{player_first_name} {player_last_name}",
            "pitcher_id": pitcher_id,
            "season": start_date[:4],
            "model_type": "PyTorch MLP",
            "epochs": 5,
            "hidden_dims": [128, 64],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "dqn_total_timesteps": 300_000,
            "dqn_buffer_size": 100_000,
            "dqn_learning_rate": 1e-4,
            "dqn_exploration_fraction": 0.30,
            "dqn_exploration_final_eps": 0.05,
            "dqn_gamma": 0.99,
        },
    )

    try:
        # -------------------------------------------------------------
        # 2. 01~04 데이터 파이프라인 실행
        # -------------------------------------------------------------
        print("\n[단계 1] 01 preprocess, 02 embedding, 03 profiles, 04 matchup 실행")
        df_emb, _ = run_data_pipeline(start_date, end_date, project_root)

        # 해당 투수만 필터 (임베딩 결과에 없으면 preprocess에서 pitch_type으로 fallback)
        paths = get_paths(project_root)
        df_pitcher = df_emb[df_emb["pitcher"] == pitcher_id].copy()
        if len(df_pitcher) == 0:
            log("  → 임베딩에 해당 투수 없음. preprocess 데이터로 fallback (pitch_type 기반)")
            df_clean = read_parquet(paths.processed_pitch_clean_range(start_date, end_date))
            df_pitcher = df_clean[df_clean["pitcher"] == pitcher_id].copy()
            df_pitcher["pitch_cluster_id"] = (
                df_pitcher["pitcher"].astype(str) + "_" + df_pitcher["pitch_type"].fillna("UNK").astype(str)
            )
            df_pitcher["pitch_cluster_local"] = -1
            if len(df_pitcher) == 0:
                raise ValueError(
                    f"해당 기간에 투수 {player_first_name} {player_last_name} (ID={pitcher_id})의 투구 데이터가 없습니다. "
                    "기간을 넓히거나 다른 투수를 선택하세요."
                )
        log(f"  → 해당 투수 투구 수: {len(df_pitcher):,}")

        # 모델 입력 형식으로 변환
        df_for_model = prepare_df_for_model(df_pitcher)

        # W&B Artifact: 전처리 데이터 업로드
        if wandb.run:
            artifact = wandb.Artifact(name="pitch_processed", type="dataset")
            tmp_path = project_root / "data" / "processed" / f"pitcher_{pitcher_id}_for_model.parquet"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            save_parquet(df_for_model, tmp_path)
            artifact.add_file(str(tmp_path))
            wandb.log_artifact(artifact)
            log("  → W&B Artifact 업로드 완료")

        # -------------------------------------------------------------
        # 3. 상태 전이 확률 예측 모델 학습
        # -------------------------------------------------------------
        print("\n[단계 2] 상태 전이 확률 예측(MLP) 모델 학습")
        config = wandb.config
        model_module = TransitionProbabilityModel(
            df=df_for_model,
            batch_size=64,
            lr=config.learning_rate,
        )
        model_module.run_modeling_pipeline(
            epochs=config.epochs,
            upload_artifact=True,
        )

        feature_cols = model_module.feature_columns
        target_classes = model_module.target_classes

        # 구종: noise(-1) 제외
        pitch_names = (
            df_pitcher[df_pitcher["pitch_cluster_local"] >= 0]["pitch_cluster_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        if len(pitch_names) == 0:
            pitch_names = df_pitcher["pitch_cluster_id"].astype(str).unique().tolist()
        print(f"식별된 구종 수: {len(pitch_names)}")

        zones = sorted([float(z) for z in df_for_model["zone"].dropna().unique()])

        # -------------------------------------------------------------
        # 4. MDP 기반 최적 볼배합 도출
        # -------------------------------------------------------------
        print("\n[단계 3] 벨만 방정식 기반 최적 볼배합(Policy) 도출")
        optimizer = MDPOptimizer(
            transition_model=model_module,
            feature_columns=feature_cols,
            target_classes=target_classes,
            pitch_names=pitch_names,
            zones=zones,
        )
        optimal_policy = optimizer.run_optimizer()

        # -------------------------------------------------------------
        # 5. DQN 강화학습 에이전트 학습
        # -------------------------------------------------------------
        print("\n[단계 4] DQN Model-Free 강화학습 에이전트 학습")

        train_env = PitchEnv(
            transition_model=model_module,
            pitch_names=pitch_names,
            zones=zones,
        )
        eval_env = PitchEnv(
            transition_model=model_module,
            pitch_names=pitch_names,
            zones=zones,
        )

        dqn_config = wandb.config
        trainer = (
            DQNTrainer(env=train_env, eval_env=eval_env)
            .build(
                learning_rate=dqn_config.dqn_learning_rate,
                buffer_size=dqn_config.dqn_buffer_size,
                exploration_fraction=dqn_config.dqn_exploration_fraction,
                exploration_final_eps=dqn_config.dqn_exploration_final_eps,
                gamma=dqn_config.dqn_gamma,
            )
        )

        trainer.train(
            total_timesteps=dqn_config.dqn_total_timesteps,
            use_wandb=True,
        )

        trainer.evaluate(n_episodes=100)
        trainer.print_policy_sample(train_env)

        print("\n" + "=" * 60)
        print("모든 SmartPitch 파이프라인이 성공적으로 완료되었습니다!")
        print("=" * 60)

    except Exception as e:
        print(f"\n파이프라인 실행 중 오류가 발생했습니다: {e}")
        raise e

    finally:
        wandb.finish()


if __name__ == "__main__":
    print("=" * 60)
    print("⚾ SmartPitch AI: 메이저리그 투수 볼배합 최적화 파이프라인 ⚾")
    print("=" * 60)
    print("\n[데이터 수집 안내]")
    print("- 메이저리그 스탯캐스트(Statcast) 트래킹 데이터 기반입니다.")
    print("- 권장 데이터 기간: 2015년 ~ 2024년 정규시즌 (2015년 이전은 결측치가 많을 수 있습니다.)")
    print("\n[🎯 AI 성능 테스트를 위한 추천 투수 3인방]")
    print("1. Gerrit Cole   : 4구종(직구/슬/커/체)을 바탕으로 한 정석적인 투구 정책 베이스라인 테스트")
    print("2. Yu Darvish    : 10개 이상의 다양한 구종을 던지는 투수의 한계 군집화(Clustering) 테스트")
    print("3. Clayton Kershaw: 좌완 레전드의 예리한 슬라이더가 RE24 실점 억제에 미치는 영향 분석")
    print("\n" + "-" * 60)

    player_name = input("▶ 분석할 투수의 영문 이름을 입력하세요 (기본값: Gerrit Cole): ").strip()
    if not player_name:
        player_name = "Gerrit Cole"

    start_date = input("▶ 데이터 시작일을 입력하세요 (기본값: 2019-03-28): ").strip()
    if not start_date:
        start_date = "2019-03-28"

    end_date = input("▶ 데이터 종료일을 입력하세요 (기본값: 2019-09-29): ").strip()
    if not end_date:
        end_date = "2019-09-29"

    name_parts = player_name.split()
    if len(name_parts) >= 2:
        player_first_name = name_parts[0]
        player_last_name = " ".join(name_parts[1:])
    else:
        player_first_name = player_name
        player_last_name = ""

    main(player_first_name, player_last_name, start_date, end_date)
