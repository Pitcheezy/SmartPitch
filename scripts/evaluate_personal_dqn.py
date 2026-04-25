"""
evaluate_personal_dqn.py — Cease/Gallen 개인 DQN 베이스라인 비교 + 통계 분석

5종 에이전트(Random, MostFrequent, Frequency, MDPPolicy, DQN)를
각 투수의 개인화 action space에서 1000 에피소드 평가하고,
통계적 유의성 검정(Welch t-test, Cohen's d)까지 수행.

산출물:
    docs/baseline_cease.md          Cease 5-agent 비교
    docs/baseline_gallen.md         Gallen 5-agent 비교
    data/dqn_cease_2024_2025_eval.json    Cease 평가 결과 JSON
    data/dqn_gallen_2024_2025_eval.json   Gallen 평가 결과 JSON

실행:
    uv run scripts/evaluate_personal_dqn.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from scipy import stats

# 프로젝트 루트 등록
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from stable_baselines3 import DQN
from src.model import TransitionProbabilityModel
from src.pitch_env import PitchEnv
from src.mdp_solver import MDPOptimizer

_DATA_DIR = os.path.join(_ROOT, 'data')
_DOCS_DIR = os.path.join(_ROOT, 'docs')

N_EPISODES = 1000
SEED_BASE = 0

# ── 투수별 설정 ──────────────────────────────────────────────────────────────

PITCHERS = {
    "cease": {
        "name": "Dylan Cease",
        "mlbam_id": 656302,
        "pitcher_cluster": 0,
        "seasons": "2024+2025",
        "n_pitches_total": 5400,
        "pitch_names": ["Fastball", "Slider", "Changeup"],  # clustering 결과 순서 (구속 내림차순)
        "dqn_model_path": os.path.join(_ROOT, "dqn_cease_2024_2025.zip"),
        "doc_path": os.path.join(_DOCS_DIR, "baseline_cease.md"),
        "json_path": os.path.join(_DATA_DIR, "dqn_cease_2024_2025_eval.json"),
        # MostFrequent: 학습 결과에서 가장 많이 선택된 구종+존
        # Cease DQN 평가: Fastball 82.6% → 가장 많이 사용된 존은 zone 9 (1606회)
        "most_freq_pitch": "Fastball",
        "most_freq_zone": 9,
        # Frequency: 학습된 DQN이 아닌 실제 Statcast 투구 분포 사용
        # 2024+2025 합산: FF 43%, SL 44%, KC 7.5%, 나머지 소량
        # clustering 결과 3구종 매핑: Fastball(FF)=43%, Slider(SL+ST+KC)=55%, Changeup(CH)=2%
        # → 존 분포는 균등 가정 (실측 데이터 없으므로)
        "freq_probs": None,  # 아래에서 동적 계산
    },
    "gallen": {
        "name": "Zac Gallen",
        "mlbam_id": 668678,
        "pitcher_cluster": 0,
        "seasons": "2024+2025",
        "n_pitches_total": 4572,
        "pitch_names": ["Fastball", "Slider", "Changeup", "Curveball"],  # clustering 결과 순서 (구속 내림차순)
        "dqn_model_path": os.path.join(_ROOT, "dqn_gallen_2024_2025.zip"),
        "doc_path": os.path.join(_DOCS_DIR, "baseline_gallen.md"),
        "json_path": os.path.join(_DATA_DIR, "dqn_gallen_2024_2025_eval.json"),
        # Gallen DQN 평가: Fastball 35.3%, Curveball 33.6% → zone 9 (1578회)
        "most_freq_pitch": "Fastball",
        "most_freq_zone": 9,
        "freq_probs": None,
    },
}


# ── 환경 / 모델 로드 ────────────────────────────────────────────────────────

def _load_universal_model():
    """범용 전이 모델 로드."""
    model = TransitionProbabilityModel.load_from_checkpoint(
        model_path=os.path.join(_ROOT, "best_transition_model_universal.pth"),
        feature_columns_path=os.path.join(_DATA_DIR, "feature_columns_universal.json"),
        target_classes_path=os.path.join(_DATA_DIR, "target_classes_universal.json"),
        model_config_path=os.path.join(_DATA_DIR, "model_config_universal.json"),
    )
    return model


def _build_env(model, pitch_names, pitcher_cluster=0):
    """투수별 PitchEnv 생성."""
    zones = sorted({
        int(col[len("zone_"):])
        for col in model.feature_columns
        if col.startswith("zone_")
    })
    env = PitchEnv(
        transition_model=model,
        pitch_names=pitch_names,
        zones=zones,
        pitcher_cluster=pitcher_cluster,
    )
    return env, zones


def _build_freq_probs_from_statcast(pitcher_id, pitch_names, zones):
    """Statcast에서 실제 투구 분포를 가져와 action 확률 벡터 생성."""
    try:
        from pybaseball import statcast_pitcher, cache
        from src.universal_model_trainer import PITCH_TYPE_MAP
        cache.enable()

        # 2024+2025 데이터
        df1 = statcast_pitcher("2024-03-28", "2024-10-31", player_id=pitcher_id)
        df2 = statcast_pitcher("2025-03-27", "2025-07-31", player_id=pitcher_id)
        df = pd.concat([df1, df2], ignore_index=True)

        df['mapped_pitch_name'] = df['pitch_type'].map(PITCH_TYPE_MAP)
        df = df[df['mapped_pitch_name'].notna()]
        df = df[df['zone'].notna()]
        df['zone'] = df['zone'].astype(float).astype(int)

        pitch_set = set(pitch_names)
        zone_set = set(int(z) for z in zones)
        df = df[df['mapped_pitch_name'].isin(pitch_set) & df['zone'].isin(zone_set)]

        if df.empty:
            return None

        n_pitches = len(pitch_names)
        n_zones = len(zones)
        probs = np.zeros(n_pitches * n_zones, dtype=np.float64)
        pitch_to_idx = {p: i for i, p in enumerate(pitch_names)}
        zone_to_idx = {int(z): i for i, z in enumerate(zones)}

        for (pitch, zone), cnt in df.groupby(['mapped_pitch_name', 'zone']).size().items():
            if pitch in pitch_to_idx and int(zone) in zone_to_idx:
                action = pitch_to_idx[pitch] * n_zones + zone_to_idx[int(zone)]
                probs[action] = float(cnt)

        s = probs.sum()
        if s == 0:
            return None
        return probs / s
    except Exception as e:
        print(f"  [Frequency] Statcast 분포 수집 실패: {e}")
        return None


# ── 베이스라인 에이전트 ──────────────────────────────────────────────────────

class RandomAgent:
    name = "Random"

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def act(self, obs, rng):
        return int(rng.integers(0, self.n_actions))


class FixedActionAgent:
    def __init__(self, name, action, label):
        self.name = name
        self.action = int(action)
        self.label = label

    def act(self, obs, rng):
        return self.action


class CategoricalAgent:
    def __init__(self, name, probs):
        self.name = name
        self.probs = probs.astype(np.float64)
        self.probs /= self.probs.sum()
        self.n_actions = len(probs)

    def act(self, obs, rng):
        return int(rng.choice(self.n_actions, p=self.probs))


class MDPPolicyAgent:
    name = "MDPPolicy"

    def __init__(self, optimal_policy, pitch_names, zones):
        self.policy = optimal_policy
        self.pitch_names = pitch_names
        self.zones = [int(z) for z in zones]
        self.n_zones = len(self.zones)
        self.n_actions = len(pitch_names) * self.n_zones
        self.pitch_to_idx = {p: i for i, p in enumerate(pitch_names)}
        self.zone_to_idx = {int(z): i for i, z in enumerate(self.zones)}
        self.miss_count = 0

    def _obs_to_state_key(self, obs):
        balls = int(obs[0])
        strikes = int(obs[1])
        outs = int(obs[2])
        on1, on2, on3 = int(obs[3]), int(obs[4]), int(obs[5])
        bc = int(obs[6])
        pc = int(obs[7])
        return f"{balls}-{strikes}_{outs}_{on1}{on2}{on3}_{bc}_{pc}"

    def act(self, obs, rng):
        key = self._obs_to_state_key(obs)
        action_info = self.policy.get(key)
        if action_info is None:
            self.miss_count += 1
            return int(rng.integers(0, self.n_actions))
        pitch = action_info['pitch']
        zone = int(action_info['zone'])
        if pitch not in self.pitch_to_idx or zone not in self.zone_to_idx:
            self.miss_count += 1
            return int(rng.integers(0, self.n_actions))
        return self.pitch_to_idx[pitch] * self.n_zones + self.zone_to_idx[zone]


class DQNAgent:
    def __init__(self, name, model_path, env):
        self.name = name
        self.model = DQN.load(model_path, env=env)

    def act(self, obs, rng):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


# ── 평가 루프 ────────────────────────────────────────────────────────────────

def evaluate_agent(env, agent, n_episodes, seed_base, pitch_names, zones):
    """에이전트를 n_episodes만큼 평가하고 에피소드별 보상 배열도 반환."""
    n_zones = len(zones)
    n_pitches_cnt = len(pitch_names)
    pitch_counts = np.zeros(n_pitches_cnt, dtype=np.int64)
    episode_rewards = np.zeros(n_episodes, dtype=np.float64)
    pitches_per_ep = np.zeros(n_episodes, dtype=np.int64)

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
        pitches_per_ep[ep] = n_pitches_in_ep

    # 구종 엔트로피
    total = pitch_counts.sum()
    if total > 0:
        p = pitch_counts / total
        p_nz = p[p > 0]
        entropy = float(-(p_nz * np.log(p_nz)).sum())
    else:
        entropy = 0.0

    # 구종 비율
    pitch_dist = {}
    if total > 0:
        for i, pn in enumerate(pitch_names):
            pitch_dist[pn] = round(float(pitch_counts[i] / total * 100), 1)

    mean_r = float(episode_rewards.mean())
    std_r = float(episode_rewards.std())
    sem = std_r / np.sqrt(n_episodes)
    ci_95_low = mean_r - 1.96 * sem
    ci_95_high = mean_r + 1.96 * sem

    return {
        "agent": agent.name,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "sem": sem,
        "ci_95": [round(ci_95_low, 4), round(ci_95_high, 4)],
        "pitch_entropy": entropy,
        "mean_pitches_per_ep": float(pitches_per_ep.mean()),
        "pitch_distribution": pitch_dist,
        "episode_rewards": episode_rewards,  # 통계 검정용
    }


# ── MDP 풀이 (개인별) ────────────────────────────────────────────────────────

def solve_mdp_for_pitcher(model, pitch_names, zones):
    """해당 투수의 action space로 MDP 풀이 (캐시 없이 fresh 계산)."""
    # 투수 군집 0만 사용하되, 전체 군집 상태 생성 (범용 모델이므로)
    all_clusters = ["0", "1", "2", "3"]
    valid_pitches_by_cluster = {c: pitch_names for c in all_clusters}

    optimizer = MDPOptimizer(
        transition_model=model,
        feature_columns=model.feature_columns,
        target_classes=model.target_classes,
        pitch_names=pitch_names,
        zones=zones,
        pitcher_clusters=all_clusters,
        valid_pitches_by_cluster=valid_pitches_by_cluster,
    )
    optimizer.solve_mdp()
    return optimizer.optimal_policy


# ── 통계 검정 ────────────────────────────────────────────────────────────────

def statistical_tests(results):
    """DQN vs 각 베이스라인: Welch t-test + Cohen's d."""
    dqn_result = None
    for r in results:
        if "DQN" in r["agent"]:
            dqn_result = r
            break
    if dqn_result is None:
        return []

    dqn_rewards = dqn_result["episode_rewards"]
    tests = []

    for r in results:
        if "DQN" in r["agent"]:
            continue
        other_rewards = r["episode_rewards"]

        # Welch's t-test (unequal variance)
        t_stat, p_value = stats.ttest_ind(dqn_rewards, other_rewards, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt((dqn_rewards.std()**2 + other_rewards.std()**2) / 2)
        cohens_d = (dqn_rewards.mean() - other_rewards.mean()) / pooled_std if pooled_std > 0 else 0

        # Effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect = "negligible"
        elif abs_d < 0.5:
            effect = "small"
        elif abs_d < 0.8:
            effect = "medium"
        else:
            effect = "large"

        tests.append({
            "comparison": f"DQN vs {r['agent']}",
            "dqn_mean": round(dqn_rewards.mean(), 4),
            "other_mean": round(other_rewards.mean(), 4),
            "diff": round(dqn_rewards.mean() - other_rewards.mean(), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_value),
            "p_value_str": f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
            "significant_005": bool(p_value < 0.05),
            "significant_001": bool(p_value < 0.01),
            "cohens_d": round(float(cohens_d), 4),
            "effect_size": effect,
        })

    return tests


# ── 출력 ─────────────────────────────────────────────────────────────────────

def save_markdown(pitcher_key, pitcher_info, results, stat_tests, out_path):
    """투수별 baseline 비교 마크다운 저장."""
    md = []
    md.append(f"# Baseline Comparison — {pitcher_info['name']}")
    md.append("")
    md.append(f"- 투수: **{pitcher_info['name']}** (MLBAM {pitcher_info['mlbam_id']}, 군집 {pitcher_info['pitcher_cluster']})")
    md.append(f"- 학습 데이터: {pitcher_info['seasons']} ({pitcher_info['n_pitches_total']:,} 투구)")
    md.append(f"- 환경: `PitchEnv(pitcher_cluster={pitcher_info['pitcher_cluster']})` + 범용 전이 모델")
    md.append(f"- 평가: 각 에이전트 **{N_EPISODES}** 에피소드, 동일 seed (`{SEED_BASE}~{SEED_BASE+N_EPISODES-1}`)")

    n_pitches = len(pitcher_info['pitch_names'])
    n_zones = 13
    md.append(f"- Action space: **{n_pitches * n_zones}** = {n_pitches}구종 × {n_zones}존")
    md.append(f"  - 구종: {', '.join(pitcher_info['pitch_names'])}")
    md.append("")

    # 결과 테이블
    md.append("## Results")
    md.append("")
    md.append("| Agent | Mean Reward | Std | SEM | 95% CI | Pitch Entropy | Pitches/Ep |")
    md.append("|-------|------------|-----|-----|--------|---------------|------------|")
    for r in results:
        ci = r['ci_95']
        md.append(
            f"| {r['agent']} "
            f"| {r['mean_reward']:+.4f} "
            f"| {r['std_reward']:.4f} "
            f"| {r['sem']:.4f} "
            f"| [{ci[0]:+.4f}, {ci[1]:+.4f}] "
            f"| {r['pitch_entropy']:.3f} "
            f"| {r['mean_pitches_per_ep']:.2f} |"
        )
    md.append("")

    # 구종 분포
    md.append("## 구종 분포")
    md.append("")
    md.append("| Agent | " + " | ".join(pitcher_info['pitch_names']) + " |")
    md.append("|-------|" + "|".join(["------"] * n_pitches) + "|")
    for r in results:
        dist = r.get('pitch_distribution', {})
        cols = " | ".join(f"{dist.get(p, 0):.1f}%" for p in pitcher_info['pitch_names'])
        md.append(f"| {r['agent']} | {cols} |")
    md.append("")

    # 통계 검정
    md.append("## 통계적 유의성 검정")
    md.append("")
    md.append("Welch's t-test (양측검정, 불등분산), Cohen's d 효과크기")
    md.append("")
    md.append("| 비교 | 차이 | t-통계량 | p-value | 유의(α=0.05) | Cohen's d | 효과크기 |")
    md.append("|------|------|---------|---------|-------------|-----------|---------|")
    for t in stat_tests:
        sig = "**Yes**" if t['significant_005'] else "No"
        md.append(
            f"| {t['comparison']} "
            f"| {t['diff']:+.4f} "
            f"| {t['t_statistic']:.4f} "
            f"| {t['p_value_str']} "
            f"| {sig} "
            f"| {t['cohens_d']:+.4f} "
            f"| {t['effect_size']} |"
        )
    md.append("")

    md.append("## 재실행")
    md.append("")
    md.append("```bash")
    md.append("uv run scripts/evaluate_personal_dqn.py")
    md.append("```")
    md.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  [save] {out_path}")


def save_json(pitcher_info, results, stat_tests, out_path):
    """평가 결과 JSON 저장 (episode_rewards 제외)."""
    data = {
        "pitcher": {
            "name": pitcher_info["name"],
            "mlbam_id": pitcher_info["mlbam_id"],
            "pitcher_cluster": pitcher_info["pitcher_cluster"],
            "seasons": pitcher_info["seasons"],
            "n_pitches_total": pitcher_info["n_pitches_total"],
            "pitch_names": pitcher_info["pitch_names"],
            "action_space": len(pitcher_info["pitch_names"]) * 13,
        },
        "evaluation": {
            "n_episodes": N_EPISODES,
            "seed_base": SEED_BASE,
        },
        "results": [
            {k: v for k, v in r.items() if k != "episode_rewards"}
            for r in results
        ],
        "statistical_tests": stat_tests,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [save] {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def evaluate_pitcher(pitcher_key, pitcher_info, model):
    """한 투수에 대해 5-agent 평가 + 통계 검정."""
    print(f"\n{'='*60}")
    print(f"  {pitcher_info['name']} — 개인 DQN 베이스라인 평가")
    print(f"{'='*60}")

    pitch_names = pitcher_info['pitch_names']
    env, zones = _build_env(model, pitch_names, pitcher_info['pitcher_cluster'])
    n_actions = env.action_space.n
    n_zones = len(zones)

    print(f"  구종: {pitch_names}")
    print(f"  Action space: {n_actions} ({len(pitch_names)}구종 × {n_zones}존)")

    agents = []

    # 1. Random
    agents.append(RandomAgent(n_actions))

    # 2. MostFrequent
    pitch_to_idx = {p: i for i, p in enumerate(pitch_names)}
    zone_to_idx = {int(z): i for i, z in enumerate(zones)}
    mf_pitch = pitcher_info['most_freq_pitch']
    mf_zone = pitcher_info['most_freq_zone']
    if mf_pitch in pitch_to_idx and int(mf_zone) in zone_to_idx:
        mf_action = pitch_to_idx[mf_pitch] * n_zones + zone_to_idx[int(mf_zone)]
        agents.append(FixedActionAgent(
            "MostFrequent", mf_action,
            f"{mf_pitch} zone {mf_zone}"
        ))
    else:
        print(f"  [WARNING] MostFrequent 구종/존 매핑 실패, 스킵")

    # 3. Frequency (Statcast 실제 분포)
    print(f"  Statcast에서 실제 투구 분포 수집 중...")
    freq_probs = _build_freq_probs_from_statcast(
        pitcher_info['mlbam_id'], pitch_names, zones
    )
    if freq_probs is not None:
        agents.append(CategoricalAgent("Frequency", freq_probs))
        print(f"  → Frequency 에이전트 생성 완료")
    else:
        print(f"  [WARNING] Frequency 분포 수집 실패, 스킵")

    # 4. MDPPolicy (개인 action space로 fresh 계산)
    print(f"  MDP 최적 정책 계산 중 (개인 action space {n_actions})...")
    mdp_policy = solve_mdp_for_pitcher(model, pitch_names, zones)
    mdp_agent = MDPPolicyAgent(mdp_policy, pitch_names, zones)
    agents.append(mdp_agent)

    # 5. DQN (학습된 모델)
    dqn_path = pitcher_info['dqn_model_path']
    if os.path.exists(dqn_path):
        dqn_agent = DQNAgent(f"DQN ({pitcher_info['name']})", dqn_path, env)
        agents.append(dqn_agent)
    else:
        print(f"  [ERROR] DQN 모델 없음: {dqn_path}")

    # 평가
    results = []
    for agent in agents:
        print(f"\n  [{agent.name}] 평가 중 ({N_EPISODES} 에피소드)...")
        result = evaluate_agent(env, agent, N_EPISODES, SEED_BASE, pitch_names, zones)
        ci = result['ci_95']
        print(f"    → 평균 보상: {result['mean_reward']:+.4f} ± {result['std_reward']:.4f}")
        print(f"    → SEM: {result['sem']:.4f}, 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        if hasattr(agent, 'miss_count') and agent.miss_count > 0:
            print(f"    → MDP miss (fallback): {agent.miss_count}회")
        results.append(result)

    # 통계 검정
    print(f"\n  통계적 유의성 검정 (Welch t-test + Cohen's d)...")
    stat_tests = statistical_tests(results)
    for t in stat_tests:
        sig_mark = "***" if t['significant_001'] else ("**" if t['significant_005'] else "")
        print(f"    {t['comparison']}: diff={t['diff']:+.4f}, "
              f"p={t['p_value_str']}{sig_mark}, d={t['cohens_d']:+.4f} ({t['effect_size']})")

    # 저장
    save_markdown(pitcher_key, pitcher_info, results, stat_tests, pitcher_info['doc_path'])
    save_json(pitcher_info, results, stat_tests, pitcher_info['json_path'])

    return results, stat_tests


def main():
    print("=" * 60)
    print("  SmartPitch — 개인 DQN 베이스라인 비교 평가")
    print("=" * 60)

    model = _load_universal_model()

    all_results = {}
    all_tests = {}

    for key, info in PITCHERS.items():
        results, tests = evaluate_pitcher(key, info, model)
        all_results[key] = results
        all_tests[key] = tests

    # 요약 출력
    print(f"\n{'='*60}")
    print("  전체 요약")
    print(f"{'='*60}")
    for key, results in all_results.items():
        name = PITCHERS[key]['name']
        print(f"\n  [{name}]")
        for r in results:
            print(f"    {r['agent']:20s}: {r['mean_reward']:+.4f} ± {r['std_reward']:.4f} "
                  f"(SEM {r['sem']:.4f}, 95% CI [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}])")

    print(f"\n{'='*60}")
    print("  평가 완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
