"""
BIP (Ball In Play) 확률 계산 스크립트.

Statcast play-by-play 데이터에서 인플레이 타구 결과(아웃/1루타/2루타/3루타/홈런)
비율을 시즌별로 집계한다.

현재 PitchEnv/MDPSolver는 아웃 70%, 1루타 15%, 2루타 10%, 홈런 5%로 하드코딩되어 있다.
이 스크립트로 실제 MLB 비율을 계산하여 data/bip_probabilities_{YYYY}.json으로 저장한다.

사용법:
  uv run scripts/compute_bip_probabilities.py              # 2021~2025 전체
  uv run scripts/compute_bip_probabilities.py 2024          # 특정 시즌만
  uv run scripts/compute_bip_probabilities.py --dry-run     # 저장 없이 출력만

출력:
  data/bip_probabilities_{YYYY}.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
from pybaseball import cache, statcast

# ── 시즌 날짜 (compute_re24_per_season.py와 동일) ──────────────────────────
SEASON_DATES = {
    2019: ("2019-03-28", "2019-09-29"),
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}

# ── Statcast events → BIP 카테고리 매핑 ────────────────────────────────────
BIP_EVENT_MAP = {
    # 아웃
    "field_out": "out",
    "force_out": "out",
    "grounded_into_double_play": "out",
    "fielders_choice": "out",
    "fielders_choice_out": "out",
    "sac_fly": "out",
    "sac_bunt": "out",
    "double_play": "out",
    "triple_play": "out",
    "sac_fly_double_play": "out",
    "sac_bunt_double_play": "out",
    # 안타
    "single": "single",
    "field_error": "single",  # 에러는 1루타와 유사한 결과
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
}


def _get_season_dates(year: int) -> tuple[str, str]:
    """시즌 시작/종료 날짜 반환."""
    if year in SEASON_DATES:
        return SEASON_DATES[year]
    print(f"[경고] {year} 시즌 날짜가 미등록. 3/28 ~ 9/29로 추정합니다.")
    return (f"{year}-03-28", f"{year}-09-29")


def compute_bip(year: int) -> dict:
    """시즌별 BIP 확률 계산.

    Returns:
        {
            "season": 2024,
            "source": "pybaseball statcast (computed)",
            "n_total_bip": 12345,
            "n_categorized": 12300,
            "probabilities": {"out": 0.68, "single": 0.16, ...},
            "counts": {"out": 8400, "single": 1900, ...}
        }
    """
    cache.enable()
    start_dt, end_dt = _get_season_dates(year)

    print(f"\n[수집] {year} 시즌 Statcast 데이터 ({start_dt} ~ {end_dt})...")
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    if df is None or df.empty:
        print(f"[오류] {year} 데이터가 비어있습니다.", file=sys.stderr)
        sys.exit(1)
    print(f"       수집 완료: {len(df):,}건")

    # hit_into_play 이벤트만 필터링
    # description == "hit_into_play" 인 행의 events 컬럼 사용
    bip = df[df["description"] == "hit_into_play"].copy()
    print(f"[필터] hit_into_play: {len(bip):,}건")

    # events → BIP 카테고리 매핑
    bip["category"] = bip["events"].map(BIP_EVENT_MAP)

    # 매핑 안 된 이벤트 확인
    unmapped = bip[bip["category"].isna()]["events"].value_counts()
    if len(unmapped) > 0:
        print(f"[주의] 매핑 안 된 이벤트 {len(unmapped)}종:")
        for evt, cnt in unmapped.items():
            print(f"       {evt}: {cnt}")

    bip_valid = bip[bip["category"].notna()]
    print(f"[유효] 카테고리 매핑 완료: {len(bip_valid):,}건")

    # 확률 계산
    counts = bip_valid["category"].value_counts().to_dict()
    total = len(bip_valid)
    probabilities = {k: round(v / total, 4) for k, v in counts.items()}

    # 결과 정리 (순서 고정)
    categories = ["out", "single", "double", "triple", "home_run"]
    result = {
        "season": year,
        "source": "pybaseball statcast play-by-play (computed)",
        "computed_at": datetime.now().astimezone().isoformat(),
        "n_total_bip": len(bip),
        "n_categorized": total,
        "probabilities": {c: probabilities.get(c, 0.0) for c in categories},
        "counts": {c: counts.get(c, 0) for c in categories},
    }

    # 요약 출력
    print(f"\n{'='*50}")
    print(f"BIP 확률 — {year} 시즌")
    print(f"{'='*50}")
    print(f"{'카테고리':<12} {'건수':>8} {'비율':>8}")
    print("-" * 30)
    for c in categories:
        cnt = counts.get(c, 0)
        pct = probabilities.get(c, 0.0)
        print(f"{c:<12} {cnt:>8,} {pct:>8.1%}")
    print(f"{'합계':<12} {total:>8,} {'100.0%':>8}")

    # 현재 하드코딩 값과 비교
    hardcoded = {"out": 0.70, "single": 0.15, "double": 0.10, "triple": 0.0, "home_run": 0.05}
    print(f"\n[비교] 현재 하드코딩 vs 실측:")
    for c in categories:
        hc = hardcoded.get(c, 0)
        actual = probabilities.get(c, 0)
        diff = actual - hc
        print(f"  {c:<12} 하드코딩 {hc:.1%} → 실측 {actual:.1%}  (Δ {diff:+.1%})")

    return result


def save_bip_json(result: dict, output_dir: str) -> str:
    """BIP 확률을 JSON 파일로 저장."""
    year = result["season"]
    output_path = os.path.join(output_dir, f"bip_probabilities_{year}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Statcast 데이터에서 BIP(Ball In Play) 확률 계산",
    )
    parser.add_argument(
        "years",
        type=int,
        nargs="*",
        default=[2021, 2022, 2023, 2024, 2025],
        help="계산할 시즌 연도 (기본: 2021~2025)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="계산만 하고 파일 저장하지 않음",
    )
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    results = []
    for year in args.years:
        result = compute_bip(year)
        results.append(result)
        if not args.dry_run:
            save_bip_json(result, data_dir)

    # 전체 시즌 비교 요약
    if len(results) > 1:
        print(f"\n\n{'='*60}")
        print("시즌별 BIP 확률 비교")
        print(f"{'='*60}")
        categories = ["out", "single", "double", "triple", "home_run"]
        header = f"{'시즌':>6}" + "".join(f"{c:>10}" for c in categories)
        print(header)
        print("-" * len(header))
        for r in results:
            row = f"{r['season']:>6}"
            for c in categories:
                row += f"{r['probabilities'].get(c, 0):>10.1%}"
            print(row)
        # 5시즌 평균
        print("-" * len(header))
        row = f"{'평균':>6}"
        for c in categories:
            avg = sum(r["probabilities"].get(c, 0) for r in results) / len(results)
            row += f"{avg:>10.1%}"
        print(row)

    print("\n완료.")


if __name__ == "__main__":
    main()
