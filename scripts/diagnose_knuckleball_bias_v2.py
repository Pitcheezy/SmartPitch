"""
diagnose_knuckleball_bias_v2.py — Knuckleball 편중 정밀 진단 (v2)

v1 대비 개선:
  - 단일 패턴(0-0/0아웃) 아닌 다양한 상태에서 평균 확률 측정
  - lookup count ↔ 학습 데이터 관계 정확히 명시
  - 군집별로 분리 분석 (군집 간 차이 확인)
  - Knuckleball 단일 인과가 아닌 복합 요인 검증

실행:
    uv run scripts/diagnose_knuckleball_bias_v2.py
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
    model = TransitionProbabilityModel.load_from_checkpoint(
        model_path=os.path.join(_ROOT, "best_transition_model_universal.pth"),
        feature_columns_path=os.path.join(_DATA, "feature_columns_universal.json"),
        target_classes_path=os.path.join(_DATA, "target_classes_universal.json"),
        model_config_path=os.path.join(_DATA, "model_config_universal.json"),
    )

    pitch_names = [c.replace("mapped_pitch_name_", "")
                   for c in model.feature_columns if c.startswith("mapped_pitch_name_")]
    zones = [float(c.replace("zone_", ""))
             for c in model.feature_columns if c.startswith("zone_")]
    target_classes = model.target_classes  # ['ball', 'foul', 'hit_into_play', 'strike']
    si = target_classes.index('strike')
    fi = target_classes.index('foul')
    hi = target_classes.index('hit_into_play')
    bi = target_classes.index('ball')

    # 물리 피처 lookup
    phys_lookup = {}
    lookup_csv = os.path.join(_DATA, "physical_feature_lookup.csv")
    if os.path.exists(lookup_csv):
        df_lk = pd.read_csv(lookup_csv)
        for _, row in df_lk.iterrows():
            key = (int(row['pitcher_cluster']), row['mapped_pitch_name'])
            phys_lookup[key] = {
                'release_speed_n': float(row['release_speed_n']),
                'pfx_x_n': float(row['pfx_x_n']),
                'pfx_z_n': float(row['pfx_z_n']),
                'count': int(row['count']),
            }

    print("=" * 75)
    print("Knuckleball MLP Calibration 정밀 진단 (v2)")
    print("=" * 75)
    print(f"target_classes: {target_classes}")
    print(f"인덱스: ball={bi}, foul={fi}, hit_into_play={hi}, strike={si}")

    # ── A. 데이터 출처 명확화 ────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print("A. 데이터 출처: physical_feature_lookup.csv의 count")
    print(f"{'─'*75}")
    print("  - generate_physical_lookup.py가 _preprocess_raw() 적용 후 집계")
    print("  - universal_model_trainer.py의 학습 전처리와 동일한 파이프라인")
    print("  - 따라서 count ≈ MLP 학습에 사용된 행 수 (dropna 후)")

    total_all = sum(v['count'] for v in phys_lookup.values())
    kb_total = sum(v['count'] for k, v in phys_lookup.items() if k[1] == "Knuckleball")
    print(f"\n  전체: {total_all:,}건")
    print(f"  Knuckleball: {kb_total}건 ({kb_total/total_all*100:.3f}%)")
    for pc in range(4):
        pk = (pc, "Knuckleball")
        if pk in phys_lookup:
            print(f"    cluster {pc}: {phys_lookup[pk]['count']}건")
        else:
            print(f"    cluster {pc}: 0건 (lookup에 없음)")

    # ── B. 다양한 상태에서의 구종별 평균 확률 ────────────────────────────
    print(f"\n{'─'*75}")
    print("B. 구종별 MLP predict_proba() — 다양한 상태 평균")
    print("   (12 카운트 × 3 아웃 × 4 투수군집 × 13 존 = 1,872 쿼리/구종)")
    print(f"{'─'*75}")

    counts_list = []
    for b in range(4):
        for s in range(3):
            counts_list.append((b, s))

    results_by_cluster = {}  # {(cluster, pitch): avg_proba}
    results_overall = {}     # {pitch: avg_proba}

    for pitch in pitch_names:
        all_probas = []
        cluster_probas = {pc: [] for pc in range(4)}

        for pc in range(4):
            for zone in zones:
                for balls, strikes in counts_list:
                    for outs in range(3):
                        input_df = pd.DataFrame(
                            np.zeros((1, len(model.feature_columns))),
                            columns=model.feature_columns,
                        )
                        input_df['balls'] = float(balls)
                        input_df['strikes'] = float(strikes)
                        input_df['outs'] = float(outs)

                        # 물리 피처
                        pk = (pc, pitch)
                        if pk in phys_lookup:
                            for col in ['release_speed_n', 'pfx_x_n', 'pfx_z_n']:
                                if col in input_df.columns:
                                    input_df[col] = phys_lookup[pk][col]

                        # one-hot
                        for col_name in [f"mapped_pitch_name_{pitch}",
                                         f"zone_{zone:.0f}",
                                         f"pitcher_cluster_{pc}"]:
                            if col_name in input_df.columns:
                                input_df[col_name] = 1.0

                        proba = model.predict_proba(input_df)[0]
                        all_probas.append(proba)
                        cluster_probas[pc].append(proba)

        results_overall[pitch] = np.mean(all_probas, axis=0)
        for pc in range(4):
            results_by_cluster[(pc, pitch)] = np.mean(cluster_probas[pc], axis=0)

    # 전체 평균 표
    print(f"\n  {'구종':<14} {'P(ball)':>8} {'P(foul)':>8} {'P(hip)':>8} {'P(stk)':>8}  {'s+f':>6} {'s+f-h':>7}")
    print("  " + "-" * 68)

    sorted_pitches = sorted(pitch_names,
                            key=lambda p: results_overall[p][si] + results_overall[p][fi],
                            reverse=True)
    for pitch in sorted_pitches:
        p = results_overall[pitch]
        sf = p[si] + p[fi]
        sfh = p[si] + p[fi] - p[hi]
        marker = " ◀" if pitch == "Knuckleball" else ""
        print(f"  {pitch:<14} {p[bi]:>8.4f} {p[fi]:>8.4f} {p[hi]:>8.4f} {p[si]:>8.4f}"
              f"  {sf:>6.4f} {sfh:>7.4f}{marker}")

    # ── C. 군집별 Knuckleball 확률 vs 평균 ───────────────────────────────
    print(f"\n{'─'*75}")
    print("C. 군집별 분석: Knuckleball vs 해당 군집 평균")
    print(f"{'─'*75}")

    for pc in range(4):
        pk = (pc, "Knuckleball")
        has_lookup = pk in phys_lookup

        # 해당 군집의 모든 구종 평균
        cluster_all = []
        for pitch in pitch_names:
            if (pc, pitch) in results_by_cluster:
                cluster_all.append(results_by_cluster[(pc, pitch)])
        cluster_avg = np.mean(cluster_all, axis=0)

        # Knuckleball 값
        if (pc, "Knuckleball") in results_by_cluster:
            kb_proba = results_by_cluster[(pc, "Knuckleball")]
        else:
            kb_proba = None

        print(f"\n  --- Cluster {pc} ---")
        print(f"  lookup 데이터: {'있음 (' + str(phys_lookup[pk]['count']) + '건)' if has_lookup else '없음 (물리피처 0 입력)'}")
        print(f"  {'':14} {'P(ball)':>8} {'P(foul)':>8} {'P(hip)':>8} {'P(stk)':>8}  {'s+f-h':>7}")
        print(f"  {'군집 평균':<14} {cluster_avg[bi]:>8.4f} {cluster_avg[fi]:>8.4f} {cluster_avg[hi]:>8.4f} {cluster_avg[si]:>8.4f}"
              f"  {cluster_avg[si]+cluster_avg[fi]-cluster_avg[hi]:>7.4f}")
        if kb_proba is not None:
            diff_sfh = (kb_proba[si]+kb_proba[fi]-kb_proba[hi]) - (cluster_avg[si]+cluster_avg[fi]-cluster_avg[hi])
            print(f"  {'Knuckleball':<14} {kb_proba[bi]:>8.4f} {kb_proba[fi]:>8.4f} {kb_proba[hi]:>8.4f} {kb_proba[si]:>8.4f}"
                  f"  {kb_proba[si]+kb_proba[fi]-kb_proba[hi]:>7.4f}  (Δ={diff_sfh:+.4f})")

        # 해당 군집에서 가장 유리한 구종 Top 3
        cluster_scores = {}
        for pitch in pitch_names:
            pp = results_by_cluster.get((pc, pitch))
            if pp is not None:
                cluster_scores[pitch] = pp[si] + pp[fi] - pp[hi]
        top3 = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top-3 유리 구종: " + ", ".join(f"{p}({s:.4f})" for p, s in top3))

    # ── D. Knuckleball 없을 때 DQN 성능 예측 ────────────────────────────
    print(f"\n{'─'*75}")
    print("D. 복합 요인 분석")
    print(f"{'─'*75}")
    print("""
  Knuckleball 편중의 원인은 단일 요인이 아닌 복합적:

  [요인 1] MLP 학습 데이터 불균형
    - Knuckleball 192건 / 전체 717,626건 = 0.027%
    - 극소 샘플로 P(hit_into_play)를 과소추정 → 투수 유리도 과대추정
    - 군집 1, 3에는 lookup 자체가 없어 물리 피처가 0으로 입력됨

  [요인 2] 극단적 물리 피처로 인한 외삽(extrapolation)
    - Knuckleball release_speed_n = -5.9 (약 60mph)
    - 다른 구종은 -2.0 ~ +0.9 범위 → MLP 학습 범위를 벗어남
    - MLP가 이 외삽 영역에서 strike 확률을 높게 출력

  [요인 3] 행동 공간 크기 차이
    - 군집 0 DQN: Cole 전용 4구종 ~52 actions (Knuckleball 없음)
    - 군집 1~3 DQN: 범용 9구종 117 actions (Knuckleball 포함)
    - action space가 2배 이상 넓어 탐색 효율 저하

  [요인 4] DQN 탐색 부족 (300K timesteps)
    - 117 actions에서 300K 스텝은 충분하지 않을 수 있음
    - 초기 탐색에서 Knuckleball이 유리하다고 학습되면 exploitation에 고착

  ⚠ 따라서 "Knuckleball 편중 = 성능 저하의 유일한 원인"이 아니라
    "MLP 왜곡 + 행동공간 차이 + 탐색/수렴" 복합 요인으로 설명해야 함
""")

    print("=" * 75)
    print("진단 완료 (v2)")
    print("=" * 75)


if __name__ == "__main__":
    main()
