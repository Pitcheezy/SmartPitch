"""
evaluate_baselines.py — DQN과 비교할 베이스라인 3종 평가 스크립트

평가 환경:
    - 범용 전이 모델(best_transition_model_universal.pth) 로드
    - PitchEnv(pitcher_cluster=0): Cole 2019 DQN 평가와 동일 조건
    - 1000 에피소드 × 동일 seed로 공정 비교

베이스라인:
    1. RandomAgent       : 균등 랜덤
    2. MostFrequentAgent : Cole 2019 실제 투구 최빈 (pitch, zone) 고정
                           실패 시 2023 리그 최빈 조합 fallback
    3. FrequencyAgent    : 2023 리그 (pitch, zone) 분포로 샘플링
                           Cole 2019 분포가 가능하면 별도 평가 추가

DQN 레퍼런스:
    Cole 2019, 평균 보상 +0.436 ± 1.255 (W&B run h4n3o0di)

산출물:
    docs/baseline_comparison.md   비교 표
    docs/baseline_comparison.png  막대그래프 (gitignored)

실행:
    uv run src/evaluate_baselines.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 프로젝트 루트 등록 (main.py 패턴)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import TransitionProbabilityModel
from src.pitch_env import PitchEnv, get_valid_pitches
from src.mdp_solver import MDPOptimizer
from src.universal_model_trainer import PITCH_TYPE_MAP, _preprocess_raw

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_BASE, '..'))
_DATA_DIR = os.path.join(_ROOT, 'data')
_DOCS_DIR = os.path.join(_ROOT, 'docs')

# Gerrit Cole MLBAM ID (Statcast pitcher ID)
COLE_MLBAM_ID = 543037
COLE_START = "2019-03-28"
COLE_END   = "2019-09-29"

LEAGUE_START = "2023-03-30"
LEAGUE_END   = "2023-10-01"

N_EPISODES = 1000
SEED_BASE  = 0

# DQN 레퍼런스 (W&B run h4n3o0di — Cole 2019)
DQN_REFERENCE = {
    "agent":             "DQN (Cole 2019 ref)",
    "mean_reward":       0.436,
    "std_reward":        1.255,
    "pitch_entropy":     float("nan"),
    "mean_pitches_per_ep": float("nan"),
    "action_space":      "~52 (Cole 식별 4구종 × 13존)",
    "notes":             "W&B run h4n3o0di",
}


# ────────────────────────────────────────────────────────────────────────────
# 환경 / 모델 로드
# ────────────────────────────────────────────────────────────────────────────

def _build_env():
    """범용 모델을 로드하고 main.py와 동일 설정의 PitchEnv를 생성."""
    model = TransitionProbabilityModel.load_from_checkpoint(
        model_path=os.path.join(_ROOT, "best_transition_model_universal.pth"),
        feature_columns_path=os.path.join(_DATA_DIR, "feature_columns_universal.json"),
        target_classes_path=os.path.join(_DATA_DIR, "target_classes_universal.json"),
        model_config_path=os.path.join(_DATA_DIR, "model_config_universal.json"),
    )

    # feature_columns에서 mapped_pitch_name_*, zone_* 접두사 파싱
    pitch_names = sorted({
        col[len("mapped_pitch_name_"):]
        for col in model.feature_columns
        if col.startswith("mapped_pitch_name_")
    })
    zones = sorted({
        int(col[len("zone_"):])
        for col in model.feature_columns
        if col.startswith("zone_")
    })

    # 군집 0 유효 구종 필터링 (Task 18)
    valid_pitches_0 = get_valid_pitches(0, pitch_names)

    env = PitchEnv(
        transition_model=model,
        pitch_names=valid_pitches_0,
        zones=zones,
        pitcher_cluster=0,
    )

    print(f"[env] all pitch_names({len(pitch_names)})={pitch_names}")
    print(f"[env] valid pitches cluster 0({len(valid_pitches_0)})={valid_pitches_0}")
    print(f"[env] zones({len(zones)})={zones}")
    print(f"[env] action_space={env.action_space.n}")
    return env, pitch_names, valid_pitches_0, zones, model


# ────────────────────────────────────────────────────────────────────────────
# 데이터 수집 (Cole / 2023 리그)
# ────────────────────────────────────────────────────────────────────────────

def _collect_cole_pitches() -> "pd.DataFrame | None":
    """pybaseball.statcast_pitcher로 Cole 2019 투구 데이터를 가져옴 (cache 사용)."""
    try:
        from pybaseball import statcast_pitcher, cache
        cache.enable()
        print(f"[Cole] statcast_pitcher({COLE_START} ~ {COLE_END}, {COLE_MLBAM_ID}) 호출 중...")
        df = statcast_pitcher(COLE_START, COLE_END, COLE_MLBAM_ID)
        if df is None or df.empty:
            print("[Cole] 데이터 비어있음")
            return None
        print(f"[Cole] 수집 완료: {len(df):,}건")
        return df
    except Exception as e:
        print(f"[Cole] 수집 실패: {e}")
        return None


def _collect_league_2023() -> "pd.DataFrame | None":
    """pybaseball.statcast로 2023 시즌 전체 데이터를 가져옴 (cache 사용)."""
    try:
        from pybaseball import statcast, cache
        cache.enable()
        print(f"[League] statcast({LEAGUE_START} ~ {LEAGUE_END}) 호출 중... (캐시 활용)")
        df = statcast(start_dt=LEAGUE_START, end_dt=LEAGUE_END)
        if df is None or df.empty:
            print("[League] 데이터 비어있음")
            return None
        print(f"[League] 수집 완료: {len(df):,}건")
        return df
    except Exception as e:
        print(f"[League] 수집 실패: {e}")
        return None


def _df_to_pitch_zone_counts(raw_df: pd.DataFrame, pitch_names, zones) -> "pd.Series | None":
    """
    raw statcast df → (mapped_pitch_name, zone) 빈도 시리즈
    Cole/리그 모두 동일 처리. env의 (pitch_names × zones) 격자에 들어가는 조합만 집계.
    """
    df = raw_df.copy()
    df['mapped_pitch_name'] = df['pitch_type'].map(PITCH_TYPE_MAP)
    df = df[df['mapped_pitch_name'].notna()]
    df = df[df['zone'].notna()]
    df['zone'] = df['zone'].astype(float).astype(int)

    pitch_set = set(pitch_names)
    zone_set  = set(int(z) for z in zones)
    df = df[df['mapped_pitch_name'].isin(pitch_set) & df['zone'].isin(zone_set)]

    if df.empty:
        return None
    return df.groupby(['mapped_pitch_name', 'zone']).size()


def _counts_to_action_probs(counts: pd.Series, pitch_names, zones) -> np.ndarray:
    """(pitch, zone) 빈도 → action_idx 별 확률 벡터."""
    n_pitches = len(pitch_names)
    n_zones   = len(zones)
    probs = np.zeros(n_pitches * n_zones, dtype=np.float64)
    pitch_to_idx = {p: i for i, p in enumerate(pitch_names)}
    zone_to_idx  = {int(z): i for i, z in enumerate(zones)}
    for (pitch, zone), cnt in counts.items():
        if pitch in pitch_to_idx and int(zone) in zone_to_idx:
            action = pitch_to_idx[pitch] * n_zones + zone_to_idx[int(zone)]
            probs[action] = float(cnt)
    s = probs.sum()
    if s == 0:
        return None
    return probs / s


def _counts_to_top_action(counts: pd.Series, pitch_names, zones) -> "int | None":
    """(pitch, zone) 빈도 → 최빈 1쌍의 action 인덱스."""
    n_zones = len(zones)
    pitch_to_idx = {p: i for i, p in enumerate(pitch_names)}
    zone_to_idx  = {int(z): i for i, z in enumerate(zones)}

    sorted_counts = counts.sort_values(ascending=False)
    for (pitch, zone), _ in sorted_counts.items():
        if pitch in pitch_to_idx and int(zone) in zone_to_idx:
            return pitch_to_idx[pitch] * n_zones + zone_to_idx[int(zone)]
    return None


# ────────────────────────────────────────────────────────────────────────────
# 베이스라인 에이전트
# ────────────────────────────────────────────────────────────────────────────

class RandomAgent:
    name = "Random"

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def act(self, obs, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.n_actions))


class FixedActionAgent:
    """MostFrequent: 학습된 단일 action을 항상 반환."""
    def __init__(self, name: str, action: int, label: str):
        self.name = name
        self.action = int(action)
        self.label = label

    def act(self, obs, rng: np.random.Generator) -> int:
        return self.action


class CategoricalAgent:
    """Frequency: 사전 계산된 확률 분포로 샘플링."""
    def __init__(self, name: str, probs: np.ndarray):
        self.name = name
        self.probs = probs.astype(np.float64)
        self.probs /= self.probs.sum()
        self.n_actions = len(probs)

    def act(self, obs, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_actions, p=self.probs))


class MDPPolicyAgent:
    """
    MDP 최적 정책 lookup. PitchEnv 8D obs를 state key로 변환해
    optimal_policy[state]['pitch','zone']을 action 인덱스로 매핑.
    정책에 없는 상태는 균등 랜덤 fallback.
    """
    name = "MDPPolicy"

    def __init__(self, optimal_policy: dict, pitch_names, zones):
        self.policy = optimal_policy
        self.pitch_names = pitch_names
        self.zones = [int(z) for z in zones]
        self.n_zones = len(self.zones)
        self.n_actions = len(pitch_names) * self.n_zones
        self.pitch_to_idx = {p: i for i, p in enumerate(pitch_names)}
        self.zone_to_idx  = {int(z): i for i, z in enumerate(self.zones)}
        self.miss_count = 0

    def _obs_to_state_key(self, obs) -> str:
        balls   = int(obs[0])
        strikes = int(obs[1])
        outs    = int(obs[2])
        on1, on2, on3 = int(obs[3]), int(obs[4]), int(obs[5])
        bc = int(obs[6])
        pc = int(obs[7])
        return f"{balls}-{strikes}_{outs}_{on1}{on2}{on3}_{bc}_{pc}"

    def act(self, obs, rng: np.random.Generator) -> int:
        key = self._obs_to_state_key(obs)
        action_info = self.policy.get(key)
        if action_info is None:
            self.miss_count += 1
            return int(rng.integers(0, self.n_actions))
        pitch = action_info['pitch']
        zone  = int(action_info['zone'])
        if pitch not in self.pitch_to_idx or zone not in self.zone_to_idx:
            self.miss_count += 1
            return int(rng.integers(0, self.n_actions))
        return self.pitch_to_idx[pitch] * self.n_zones + self.zone_to_idx[zone]


# ────────────────────────────────────────────────────────────────────────────
# MDP solve / load (pickle cache)
# ────────────────────────────────────────────────────────────────────────────

MDP_POLICY_CACHE = os.path.join(_DATA_DIR, "mdp_optimal_policy.pkl")

def solve_or_load_mdp_policy(model, pitch_names, zones,
                             pitcher_clusters=("0", "1", "2", "3"),
                             valid_pitches_by_cluster=None) -> dict:
    """
    MDPOptimizer.solve_mdp()로 9216 상태 정책을 계산하고 pickle 캐시.
    재실행 시 캐시가 있으면 즉시 로드 (10~30분 학습 스킵).

    valid_pitches_by_cluster가 지정되면 군집별 유효 구종만 탐색 (Task 18).
    ⚠ 캐시 사용 시, 이전 캐시가 다른 action space로 생성되었을 수 있으므로
       action space 변경 후에는 반드시 캐시 삭제 필요.
    """
    import pickle

    if os.path.exists(MDP_POLICY_CACHE):
        try:
            with open(MDP_POLICY_CACHE, "rb") as f:
                cached = pickle.load(f)
            print(f"[MDP] cache 로드: {MDP_POLICY_CACHE} ({len(cached)} states)")
            return cached
        except Exception as e:
            print(f"[MDP] cache 로드 실패: {e} → 재계산")

    print(f"[MDP] solve_mdp() 실행 — 상태 수 = "
          f"{12*3*8*8*len(pitcher_clusters)} (CPU에서 10~30분 소요)")
    optimizer = MDPOptimizer(
        transition_model=model,
        feature_columns=model.feature_columns,
        target_classes=model.target_classes,
        pitch_names=pitch_names,
        zones=zones,
        pitcher_clusters=list(pitcher_clusters),
        valid_pitches_by_cluster=valid_pitches_by_cluster,
    )
    optimizer.solve_mdp()
    policy = optimizer.optimal_policy

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(MDP_POLICY_CACHE, "wb") as f:
        pickle.dump(policy, f)
    print(f"[MDP] cache 저장: {MDP_POLICY_CACHE} ({len(policy)} states)")
    return policy


# ────────────────────────────────────────────────────────────────────────────
# 평가 루프
# ────────────────────────────────────────────────────────────────────────────

def evaluate_agent(env: PitchEnv, agent, n_episodes: int, seed_base: int,
                   pitch_names, zones) -> dict:
    n_actions = env.action_space.n
    n_pitches = len(pitch_names)
    n_zones   = len(zones)

    pitch_counts = np.zeros(n_pitches, dtype=np.int64)
    episode_rewards = np.zeros(n_episodes, dtype=np.float64)
    pitches_per_ep  = np.zeros(n_episodes, dtype=np.int64)

    rng = np.random.default_rng(seed_base + 12345)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        terminated = False
        truncated = False
        ep_reward = 0.0
        n_pitches_in_ep = 0
        while not (terminated or truncated):
            action = agent.act(obs, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            pitch_counts[action // n_zones] += 1
            n_pitches_in_ep += 1
        episode_rewards[ep] = ep_reward
        pitches_per_ep[ep]  = n_pitches_in_ep

    # 구종 엔트로피 (자연로그, 0 마스킹)
    total = pitch_counts.sum()
    if total > 0:
        p = pitch_counts / total
        p_nz = p[p > 0]
        entropy = float(-(p_nz * np.log(p_nz)).sum())
    else:
        entropy = 0.0

    return {
        "agent":             agent.name,
        "mean_reward":       float(episode_rewards.mean()),
        "std_reward":        float(episode_rewards.std()),
        "pitch_entropy":     entropy,
        "mean_pitches_per_ep": float(pitches_per_ep.mean()),
        "action_space":      str(n_actions),
        "notes":             "",
    }


# ────────────────────────────────────────────────────────────────────────────
# 출력 (Markdown / PNG)
# ────────────────────────────────────────────────────────────────────────────

def _format_markdown(rows) -> str:
    headers = ["Agent", "Mean Reward ± Std", "Pitch Entropy", "Mean Pitches/Ep", "Action Space", "Notes"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        mean = r["mean_reward"]
        std  = r["std_reward"]
        if np.isnan(mean):
            mean_str = "—"
        else:
            mean_str = f"{mean:+.3f} ± {std:.3f}"
        ent = r["pitch_entropy"]
        ent_str = "—" if (isinstance(ent, float) and np.isnan(ent)) else f"{ent:.3f}"
        mpe = r["mean_pitches_per_ep"]
        mpe_str = "—" if (isinstance(mpe, float) and np.isnan(mpe)) else f"{mpe:.2f}"
        notes = r.get("notes", "") or ""
        lines.append(
            f"| {r['agent']} | {mean_str} | {ent_str} | {mpe_str} | {r['action_space']} | {notes} |"
        )
    return "\n".join(lines)


def _save_markdown(rows, out_path: str, action_space_n: int, pitch_names, zones):
    md = []
    md.append("# Baseline Comparison — SmartPitch")
    md.append("")
    md.append(f"- 환경: `PitchEnv(pitcher_cluster=0)` + 범용 전이 모델")
    md.append(f"  (`best_transition_model_universal.pth`, 4클래스: ball/foul/hit_into_play/strike)")
    md.append(f"- 평가: 각 에이전트 **{N_EPISODES}** 에피소드, 동일 seed (`{SEED_BASE}~{SEED_BASE+N_EPISODES-1}`)")
    md.append(f"- Baseline action space: **{action_space_n}** "
              f"= {len(pitch_names)} 구종 × {len(zones)} 존")
    md.append(f"  - 구종: {', '.join(pitch_names)}")
    md.append(f"  - 존:   {', '.join(str(z) for z in zones)}")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append(_format_markdown(rows))
    md.append("")
    md.append("## 비고")
    md.append("")
    md.append("- **DQN (Cole 2019 ref)**: W&B run `h4n3o0di`의 100 에피소드 평가 결과(평균 보상 +0.436 ± 1.255)이며")
    md.append("  본 스크립트가 직접 재실행한 값이 아닙니다. DQN은 `clustering.PitchClustering`로")
    md.append("  Cole 본인이 던진 구종(보통 4종: Fastball/Slider/Curveball/Changeup)만으로 학습되었으므로")
    md.append("  action space ≈ 4 × 13 = 52로, 본 베이스라인의 117과 다릅니다.")
    md.append("- 베이스라인은 universal 모델의 9개 구종 전체를 후보로 가지므로 탐색 공간이 더 큽니다.")
    md.append("  → DQN과의 직접 비교 시 \"DQN은 더 작은 action space에서 학습됨\"을 감안해야 합니다.")
    md.append("- `release_speed_n / pfx_x_n / pfx_z_n` 물리 피처는 `PitchEnv._sample_outcome`에서 0으로 입력")
    md.append("  되며, 이는 DQN 평가 시점과 동일한 조건입니다 (공정 비교).")
    md.append("")
    md.append("## 재실행")
    md.append("")
    md.append("```bash")
    md.append("uv run src/evaluate_baselines.py")
    md.append("```")
    md.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[save] markdown → {out_path}")


def _save_plot(rows, out_path: str):
    names = [r["agent"] for r in rows]
    means = [r["mean_reward"] for r in rows]
    stds  = [r["std_reward"] for r in rows]
    colors = ["#888" if "DQN" in n else "#3a7bd5" for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel("Mean episode reward (RE24-based)")
    ax.set_title(f"Baseline vs DQN — {N_EPISODES} episodes (PitchEnv, pitcher_cluster=0)")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[save] plot → {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Per-cluster 평가
# ────────────────────────────────────────────────────────────────────────────

def _cluster_characteristics(df_filtered: pd.DataFrame) -> str:
    """군집의 평균 구속 + 상위 3 구종 비중을 한 줄 문자열로 반환."""
    df = df_filtered
    if df.empty:
        return "—"

    # 평균 구속
    if 'release_speed' in df.columns:
        avg_speed = df['release_speed'].astype(float).dropna().mean()
        speed_str = f"avg {avg_speed:.1f} mph"
    else:
        speed_str = ""

    # 상위 3 구종 비중
    pitch_share = df['mapped_pitch_name'].value_counts(normalize=True).head(3)
    top_pitches = ", ".join(f"{name} {share*100:.0f}%" for name, share in pitch_share.items())

    return f"{speed_str} · {top_pitches}".strip(" ·")


def run_per_cluster_evaluation(model, pitch_names, zones,
                               league_raw: "pd.DataFrame | None",
                               mdp_policy: "dict | None" = None) -> None:
    """
    pitcher_cluster 0~3 각각에 대해 PitchEnv를 새로 만들고
    Random / MostFrequent(군집 최빈) / Frequency(군집 분포) 평가.
    결과를 docs/baseline_by_cluster.md로 저장.
    """
    print("\n" + "=" * 60)
    print("Per-Pitcher-Cluster Baseline Evaluation")
    print("=" * 60)

    cluster_csv = os.path.join(_DATA_DIR, "pitcher_clusters_2023.csv")
    if not os.path.exists(cluster_csv):
        print(f"[per-cluster] {cluster_csv} 없음 → skip")
        return
    if league_raw is None:
        print("[per-cluster] 2023 리그 raw 데이터 없음 → skip")
        return

    df_clusters = pd.read_csv(cluster_csv)  # [pitcher_id, cluster]
    pitcher_to_cluster = dict(zip(df_clusters['pitcher_id'], df_clusters['cluster']))

    # raw 전처리: pitch_type → mapped_pitch_name, zone int, cluster join
    raw = league_raw.copy()
    raw['mapped_pitch_name'] = raw['pitch_type'].map(PITCH_TYPE_MAP)
    raw = raw[raw['mapped_pitch_name'].notna()]
    raw = raw[raw['zone'].notna()]
    raw['zone'] = raw['zone'].astype(float).astype(int)
    raw['p_cluster'] = raw['pitcher'].map(pitcher_to_cluster)
    raw = raw[raw['p_cluster'].notna()]
    raw['p_cluster'] = raw['p_cluster'].astype(int)
    print(f"[per-cluster] cluster 매핑된 투구 수: {len(raw):,}건")

    cluster_ids = sorted(raw['p_cluster'].unique())
    rows_table = []           # 비교표 로우
    cluster_meta = []         # (cluster_id, top_action_label, characteristics)

    for cid in cluster_ids:
        print(f"\n--- pitcher_cluster={cid} ---")
        df_c = raw[raw['p_cluster'] == cid]
        characteristics = _cluster_characteristics(df_c)
        print(f"  특성: {characteristics}")
        print(f"  투구 수: {len(df_c):,}건")

        # 군집별 유효 구종 필터링 (Task 18)
        valid_pitches = get_valid_pitches(int(cid), pitch_names)
        print(f"  유효 구종: {valid_pitches} ({len(valid_pitches)}종)")

        counts = _df_to_pitch_zone_counts(df_c, valid_pitches, zones)
        if counts is None or counts.empty:
            print(f"  유효 (pitch,zone) 조합 없음 → skip")
            continue

        top_action = _counts_to_top_action(counts, valid_pitches, zones)
        probs      = _counts_to_action_probs(counts, valid_pitches, zones)

        n_zones_local = len(zones)
        top_pitch_idx = top_action // n_zones_local
        top_zone_idx  = top_action %  n_zones_local
        top_label = f"{valid_pitches[top_pitch_idx]} / Zone {zones[top_zone_idx]}"
        print(f"  최빈 조합: {top_label}")

        # 군집별 환경 생성 (유효 구종만 사용)
        env = PitchEnv(
            transition_model=model,
            pitch_names=valid_pitches,
            zones=zones,
            pitcher_cluster=int(cid),
        )

        agents = [
            RandomAgent(env.action_space.n),
            FixedActionAgent(name="MostFrequent", action=top_action, label=top_label),
            CategoricalAgent(name="Frequency", probs=probs),
        ]
        if mdp_policy is not None:
            agents.append(MDPPolicyAgent(mdp_policy, valid_pitches, zones))

        for agent in agents:
            r = evaluate_agent(env, agent, N_EPISODES, SEED_BASE, valid_pitches, zones)
            extra = ""
            if isinstance(agent, MDPPolicyAgent) and agent.miss_count > 0:
                extra = f"  [policy miss={agent.miss_count}]"
            print(f"  {agent.name:<14} mean={r['mean_reward']:+.4f} ± {r['std_reward']:.4f}  "
                  f"entropy={r['pitch_entropy']:.3f}  pitches/ep={r['mean_pitches_per_ep']:.2f}{extra}")
            rows_table.append({
                "cluster":   cid,
                "agent":     agent.name,
                "mean":      r['mean_reward'],
                "std":       r['std_reward'],
                "entropy":   r['pitch_entropy'],
                "mean_pitches": r['mean_pitches_per_ep'],
            })

        # DQN 참고값 행 — 군집 0만 학습된 값 보유, 나머지는 미학습 표시
        if int(cid) == 0:
            rows_table.append({
                "cluster":   cid,
                "agent":     "DQN (Cole 2019 ref)",
                "mean":      0.436,
                "std":       1.255,
                "entropy":   float("nan"),
                "mean_pitches": float("nan"),
            })
        else:
            rows_table.append({
                "cluster":   cid,
                "agent":     "DQN",
                "mean":      float("nan"),
                "std":       float("nan"),
                "entropy":   float("nan"),
                "mean_pitches": float("nan"),
            })

        cluster_meta.append((int(cid), top_label, characteristics, len(df_c)))

    # ── Markdown 저장 ────────────────────────────────────────────────────────
    out_path = os.path.join(_DOCS_DIR, "baseline_by_cluster.md")
    md = []
    md.append("# Per-Pitcher-Cluster Baseline Comparison")
    md.append("")
    md.append(f"- 환경: `PitchEnv(pitcher_cluster=K)` + 범용 전이 모델")
    md.append(f"- 평가: 각 (군집 × 에이전트) **{N_EPISODES}** 에피소드, 동일 seed")
    md.append(f"- Action space: 군집별 유효 구종 × {len(zones)} 존 (1% 미만 구종 제외)")
    md.append("")
    md.append("## 군집 정보")
    md.append("")
    md.append("| Cluster | 투구 수 (2023) | 최빈 (pitch / zone) | 특성 (avg 구속 · top3 구종) |")
    md.append("|---|---|---|---|")
    for cid, top_label, characteristics, n_pitches in cluster_meta:
        md.append(f"| {cid} | {n_pitches:,} | {top_label} | {characteristics} |")
    md.append("")
    md.append("## 평가 결과")
    md.append("")
    md.append("| Cluster | Agent | Mean ± Std | Entropy | Pitches/Ep |")
    md.append("|---|---|---|---|---|")
    for r in rows_table:
        if isinstance(r['mean'], float) and np.isnan(r['mean']):
            mean_str = "미학습"
        else:
            mean_str = f"{r['mean']:+.3f} ± {r['std']:.3f}"
        ent_str = "—" if (isinstance(r['entropy'], float) and np.isnan(r['entropy'])) else f"{r['entropy']:.3f}"
        mpe_str = "—" if (isinstance(r['mean_pitches'], float) and np.isnan(r['mean_pitches'])) else f"{r['mean_pitches']:.2f}"
        md.append(
            f"| {r['cluster']} | {r['agent']} | {mean_str} | {ent_str} | {mpe_str} |"
        )
    md.append("")
    md.append("## 비고")
    md.append("")
    md.append("- 군집 정의는 `data/pitcher_clusters_2023.csv` (K=4) — `pitcher_clustering.py` 산출물")
    md.append("- MostFrequent의 최빈 조합은 해당 군집 내 2023 시즌 (pitch_type → mapped_pitch_name, zone) value_counts 1위")
    md.append("- Frequency는 동일 군집 내 (pitch, zone) 빈도 분포로 매 step 샘플링")
    md.append("- MDPPolicy는 `MDPOptimizer.solve_mdp()`로 9,216개 상태 전체에 대해 한 번만 풀고")
    md.append(f"  `data/mdp_optimal_policy.pkl`로 캐시. obs(8D) → state key `\"b-s_outs_runners_bc_pc\"` 변환 후 lookup.")
    md.append("  모든 군집에서 동일한 정책을 공유하지만 PitchEnv 시뮬레이션 시 `pitcher_cluster`만 달라짐.")
    md.append("- DQN 행: 군집 0(Cole 2019)만 W&B run `h4n3o0di`의 평가값(+0.436 ± 1.255). 군집 1~3은 학습된 적 없음.")
    md.append("- 동일 seed로 환경 reset(`seed=0..999`) → 군집/에이전트 간 공정 비교")
    md.append("")
    md.append("## 재실행")
    md.append("")
    md.append("```bash")
    md.append("uv run src/evaluate_baselines.py")
    md.append("```")
    md.append("")

    os.makedirs(_DOCS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n[save] markdown → {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SmartPitch Baseline Evaluation")
    print("=" * 60)

    env, all_pitch_names, pitch_names, zones, model = _build_env()
    n_actions = env.action_space.n

    # ── 데이터 수집 ──────────────────────────────────────────────────────────
    cole_raw   = _collect_cole_pitches()
    league_raw = _collect_league_2023()

    cole_counts   = _df_to_pitch_zone_counts(cole_raw,   pitch_names, zones) if cole_raw   is not None else None
    league_counts = _df_to_pitch_zone_counts(league_raw, pitch_names, zones) if league_raw is not None else None

    if cole_counts is not None:
        print(f"[Cole]   유효 (pitch,zone) 조합 수: {len(cole_counts)}")
    if league_counts is not None:
        print(f"[League] 유효 (pitch,zone) 조합 수: {len(league_counts)}")

    # ── 에이전트 구성 ────────────────────────────────────────────────────────
    agents = []

    agents.append(RandomAgent(n_actions))

    # MostFrequent: Cole 우선, 실패 시 League fallback
    most_freq_action = None
    most_freq_label = None
    if cole_counts is not None:
        a = _counts_to_top_action(cole_counts, pitch_names, zones)
        if a is not None:
            most_freq_action, most_freq_label = a, "Cole 2019"
    if most_freq_action is None and league_counts is not None:
        a = _counts_to_top_action(league_counts, pitch_names, zones)
        if a is not None:
            most_freq_action, most_freq_label = a, "2023 League fallback"

    if most_freq_action is not None:
        pitch_idx = most_freq_action // len(zones)
        zone_idx  = most_freq_action %  len(zones)
        print(f"[MostFrequent] source={most_freq_label}, "
              f"action={most_freq_action} ({pitch_names[pitch_idx]} / Zone {zones[zone_idx]})")
        agents.append(FixedActionAgent(
            name=f"MostFrequent ({most_freq_label})",
            action=most_freq_action,
            label=f"{pitch_names[pitch_idx]} / Zone {zones[zone_idx]}",
        ))
    else:
        print("[MostFrequent] 모든 데이터 수집 실패 → skip")

    # Frequency (League 2023)
    if league_counts is not None:
        probs = _counts_to_action_probs(league_counts, pitch_names, zones)
        if probs is not None:
            agents.append(CategoricalAgent("Frequency (2023 League)", probs))

    # Frequency (Cole 2019, 가능 시)
    if cole_counts is not None:
        probs = _counts_to_action_probs(cole_counts, pitch_names, zones)
        if probs is not None:
            agents.append(CategoricalAgent("Frequency (Cole 2019)", probs))

    # ── 평가 ──────────────────────────────────────────────────────────────────
    results = []
    for agent in agents:
        print(f"\n[evaluate] {agent.name} — {N_EPISODES} episodes...")
        r = evaluate_agent(env, agent, N_EPISODES, SEED_BASE, pitch_names, zones)
        print(f"  mean={r['mean_reward']:+.4f} ± {r['std_reward']:.4f}  "
              f"entropy={r['pitch_entropy']:.3f}  pitches/ep={r['mean_pitches_per_ep']:.2f}")
        results.append(r)

    # DQN 레퍼런스 행 추가 (action_space 메모 갱신)
    results.append(DQN_REFERENCE)

    # ── 표 출력 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Comparison Table")
    print("=" * 60)
    print(_format_markdown(results))

    # ── 저장 ──────────────────────────────────────────────────────────────────
    os.makedirs(_DOCS_DIR, exist_ok=True)
    _save_markdown(results, os.path.join(_DOCS_DIR, "baseline_comparison.md"),
                   n_actions, pitch_names, zones)
    _save_plot(results, os.path.join(_DOCS_DIR, "baseline_comparison.png"))

    # ── MDP 최적 정책 (9216 상태) — pickle 캐시 사용 ─────────────────────────
    # 군집별 유효 구종 딕셔너리 구성 (Task 18)
    valid_pitches_by_cluster = {}
    for cid in range(4):
        valid_pitches_by_cluster[str(cid)] = get_valid_pitches(cid, all_pitch_names)
    mdp_policy = solve_or_load_mdp_policy(model, all_pitch_names, zones,
                                          pitcher_clusters=("0", "1", "2", "3"),
                                          valid_pitches_by_cluster=valid_pitches_by_cluster)

    # ── Per-pitcher-cluster 평가 (MDPPolicy 포함) ────────────────────────────
    run_per_cluster_evaluation(model, all_pitch_names, zones, league_raw, mdp_policy=mdp_policy)

    print("\n[done]")


if __name__ == "__main__":
    main()
