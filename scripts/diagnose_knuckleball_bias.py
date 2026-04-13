"""
diagnose_knuckleball_bias.py — Knuckleball MLP calibration 문제 진단

MLP가 Knuckleball에 부여하는 확률 분포를 다른 구종과 비교하여
DQN/MDP의 Knuckleball 편중 원인을 정량적으로 분석합니다.

분석 항목:
  1. 구종별 평균 predict_proba() 분포 (strike/foul/ball/hit_into_play)
  2. 구종별 "투수 유리도" 점수 (strike+foul 비율)
  3. 2023 MLB 학습 데이터에서 Knuckleball 실제 샘플 수
  4. 물리 피처 lookup 값 비교

실행:
    uv run scripts/diagnose_knuckleball_bias.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import TransitionProbabilityModel

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DATA = os.path.join(_ROOT, 'data')


def main():
    # 모델 로드
    model = TransitionProbabilityModel.load_from_checkpoint(
        model_path=os.path.join(_ROOT, "best_transition_model_universal.pth"),
        feature_columns_path=os.path.join(_DATA, "feature_columns_universal.json"),
        target_classes_path=os.path.join(_DATA, "target_classes_universal.json"),
        model_config_path=os.path.join(_DATA, "model_config_universal.json"),
    )

    # 피처 파싱
    pitch_names = [c.replace("mapped_pitch_name_", "")
                   for c in model.feature_columns if c.startswith("mapped_pitch_name_")]
    zones = [float(c.replace("zone_", ""))
             for c in model.feature_columns if c.startswith("zone_")]

    # 물리 피처 lookup 로드
    phys_lookup = {}
    lookup_csv = os.path.join(_DATA, "physical_feature_lookup.csv")
    if os.path.exists(lookup_csv):
        df_lk = pd.read_csv(lookup_csv)
        for _, row in df_lk.iterrows():
            key = (str(int(row['pitcher_cluster'])), row['mapped_pitch_name'])
            phys_lookup[key] = {
                'release_speed_n': float(row['release_speed_n']),
                'pfx_x_n': float(row['pfx_x_n']),
                'pfx_z_n': float(row['pfx_z_n']),
                'count': int(row['count']),
            }

    print("=" * 70)
    print("Knuckleball MLP Calibration 진단")
    print("=" * 70)

    target_classes = model.target_classes
    print(f"\n출력 클래스: {target_classes}")
    # ball=0, foul=1, hit_into_play=2, strike=3 (알파벳 순)

    # ── 1. 구종별 평균 predict_proba ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print("1. 구종별 평균 predict_proba() (모든 군집 × 존 평균)")
    print(f"{'─'*70}")

    results = {}
    for pitch in sorted(pitch_names):
        all_probas = []
        for pc in range(4):
            for zone in zones:
                input_df = pd.DataFrame(
                    np.zeros((1, len(model.feature_columns))),
                    columns=model.feature_columns,
                )
                # 기본 상태: 0-0 카운트, 0아웃, 주자 없음
                input_df['balls'] = 0
                input_df['strikes'] = 0
                input_df['outs'] = 0

                # 물리 피처
                pk = (str(pc), pitch)
                if pk in phys_lookup:
                    for col in ['release_speed_n', 'pfx_x_n', 'pfx_z_n']:
                        if col in input_df.columns:
                            input_df[col] = phys_lookup[pk][col]

                # one-hot
                col_pitch = f"mapped_pitch_name_{pitch}"
                if col_pitch in input_df.columns:
                    input_df[col_pitch] = 1.0
                col_zone = f"zone_{zone:.0f}"
                if col_zone in input_df.columns:
                    input_df[col_zone] = 1.0
                col_pc = f"pitcher_cluster_{pc}"
                if col_pc in input_df.columns:
                    input_df[col_pc] = 1.0

                proba = model.predict_proba(input_df)[0]
                all_probas.append(proba)

        avg_proba = np.mean(all_probas, axis=0)
        results[pitch] = avg_proba

    # 표 출력
    header = f"{'구종':<14}" + "".join(f"{c:<16}" for c in target_classes) + f"{'strike+foul':>12}"
    print(header)
    print("-" * len(header))

    sorted_pitches = sorted(results.keys(),
                            key=lambda p: results[p][target_classes.index('strike')] + results[p][target_classes.index('foul')],
                            reverse=True)

    for pitch in sorted_pitches:
        proba = results[pitch]
        sf = proba[target_classes.index('strike')] + proba[target_classes.index('foul')]
        line = f"{pitch:<14}"
        for p in proba:
            line += f"{p:.4f}          "
        line += f"{sf:.4f}"
        if pitch == "Knuckleball":
            line += "  ◀ BIAS"
        print(line)

    # ── 2. 물리 피처 비교 ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("2. 물리 피처 lookup 값 비교 (pitcher_cluster=0 기준)")
    print(f"{'─'*70}")
    print(f"{'구종':<14} {'release_speed_n':>16} {'pfx_x_n':>10} {'pfx_z_n':>10} {'count':>8}")
    print("-" * 60)
    for pitch in sorted(pitch_names):
        pk = ("0", pitch)
        if pk in phys_lookup:
            v = phys_lookup[pk]
            print(f"{pitch:<14} {v['release_speed_n']:>16.3f} {v['pfx_x_n']:>10.3f} {v['pfx_z_n']:>10.3f} {v['count']:>8}")
        else:
            print(f"{pitch:<14} {'(없음)':>16}")

    # ── 3. 학습 데이터 Knuckleball 비율 ──────────────────────────────────
    print(f"\n{'─'*70}")
    print("3. Knuckleball 물리 피처 lookup 전 군집 비교")
    print(f"{'─'*70}")
    kb_counts = []
    for pc in range(4):
        pk = (str(pc), "Knuckleball")
        if pk in phys_lookup:
            kb_counts.append((pc, phys_lookup[pk]))
    for pc, v in kb_counts:
        print(f"  cluster {pc}: speed_n={v['release_speed_n']:.3f}, "
              f"pfx_x={v['pfx_x_n']:.3f}, pfx_z={v['pfx_z_n']:.3f}, "
              f"count={v['count']}")

    total_all = sum(phys_lookup[k]['count'] for k in phys_lookup)
    total_kb = sum(phys_lookup[k]['count'] for k in phys_lookup if k[1] == "Knuckleball")
    print(f"\n  전체 lookup 투구 수: {total_all:,}")
    print(f"  Knuckleball 투구 수: {total_kb:,} ({total_kb/total_all*100:.2f}%)")

    # ── 4. 구종별 "투수 유리도" 점수 비교 ────────────────────────────────
    print(f"\n{'─'*70}")
    print("4. 구종별 투수 유리도 = P(strike) + P(foul) - P(hit_into_play)")
    print("   (높을수록 DQN/MDP가 선호)")
    print(f"{'─'*70}")

    si = target_classes.index('strike')
    fi = target_classes.index('foul')
    hi = target_classes.index('hit_into_play')

    scores = {}
    for pitch in sorted_pitches:
        p = results[pitch]
        score = p[si] + p[fi] - p[hi]
        scores[pitch] = score

    for pitch in sorted(scores.keys(), key=lambda x: scores[x], reverse=True):
        bar = "█" * int(scores[pitch] * 50)
        marker = " ◀ BIAS" if pitch == "Knuckleball" else ""
        print(f"  {pitch:<14} {scores[pitch]:.4f}  {bar}{marker}")

    print(f"\n{'='*70}")
    print("진단 완료")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
