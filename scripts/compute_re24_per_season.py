"""
RE24 (Run Expectancy by 24 base-out states) 계산 스크립트.

Statcast play-by-play 데이터로부터 RE24 매트릭스를 직접 계산한다.
data/re24_{YYYY}.json 파일을 재현하거나 검증하는 용도로 사용.

알고리즘:
  1. pybaseball.statcast()로 전체 시즌 데이터 수집
  2. 타석 완료 이벤트(events != NaN)만 필터링하여 plate appearance 단위로 변환
  3. game_pk + inning + inning_topbot 조합으로 하프이닝 식별
  4. 각 하프이닝 내에서 타석별 base-out 상태와 잔여 득점(이후 발생 득점) 계산
  5. 24개 상태별 평균 잔여 득점 = RE24

사용법:
  uv run scripts/compute_re24_per_season.py              # 2024 시즌 (기본값)
  uv run scripts/compute_re24_per_season.py 2023         # 2023 시즌
  uv run scripts/compute_re24_per_season.py 2024 --dry-run  # 저장 없이 출력만

출력:
  data/re24_{YYYY}.json (기존 RE24 JSON 포맷과 동일)
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
from pybaseball import cache, statcast

# ── 상수 ──────────────────────────────────────────────────────────────────────

# MLB 시즌 시작/종료 기본 날짜 (정규시즌 기준, 포스트시즌 제외)
SEASON_DATES = {
    2019: ("2019-03-28", "2019-09-29"),
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}

# RE24 24개 상태 키 목록
ALL_STATE_KEYS = [
    f"{outs}_{on_1b}{on_2b}{on_3b}"
    for outs in range(3)
    for on_1b in range(2)
    for on_2b in range(2)
    for on_3b in range(2)
]


def _get_season_dates(year: int) -> tuple[str, str]:
    """시즌 시작/종료 날짜 반환. 미등록 시즌은 일반적인 날짜로 추정."""
    if year in SEASON_DATES:
        return SEASON_DATES[year]
    # 미등록 시즌 — 일반적인 MLB 정규시즌 범위로 추정
    print(f"[경고] {year} 시즌 날짜가 미등록. 3/28 ~ 9/29로 추정합니다.")
    return (f"{year}-03-28", f"{year}-09-29")


def fetch_season_data(year: int) -> pd.DataFrame:
    """시즌 전체 Statcast 데이터 수집.

    pybaseball 캐시를 활성화하여 재실행 시 빠르게 로드.
    """
    cache.enable()
    start_dt, end_dt = _get_season_dates(year)

    print(f"[수집] {year} 시즌 Statcast 데이터 ({start_dt} ~ {end_dt})...")
    print("       (첫 실행 시 10~30분 소요, 캐시 이후 ~1분)")

    df = statcast(start_dt=start_dt, end_dt=end_dt)
    if df is None or df.empty:
        print("[오류] 데이터가 비어있습니다. 날짜 범위를 확인하세요.", file=sys.stderr)
        sys.exit(1)

    print(f"       수집 완료: {len(df):,}건")
    return df


def compute_re24(df: pd.DataFrame) -> dict[str, float]:
    """Statcast play-by-play 데이터에서 RE24 매트릭스를 계산.

    접근법:
      - 각 하프이닝(game_pk + inning + inning_topbot)을 독립 단위로 처리
      - 타석 완료 이벤트(events 컬럼이 NaN이 아닌 행)만 사용
      - 각 타석의 base-out 상태 기록 + 해당 시점 이후 하프이닝 끝까지의 득점 계산
      - 상태별 평균 잔여 득점 = RE24 값

    Statcast 주요 컬럼:
      - bat_score: 타석 시작 시점 타격팀 득점
      - post_bat_score: 타석 완료 후 타격팀 득점
      - outs_when_up: 타석 시작 시점 아웃 수 (0/1/2)
      - on_1b/on_2b/on_3b: 주자 유무 (NaN=빈 베이스, 선수ID=주자 있음)

    Returns:
        24개 상태 키 → 기대 득점 딕셔너리
    """
    print("[전처리] 타석 완료 이벤트 필터링...")

    # 타석 완료 이벤트만 (events가 NaN이 아닌 행 = plate appearance 결과)
    pa = df.dropna(subset=["events"]).copy()
    print(f"         전체 {len(df):,}건 → 타석 완료 {len(pa):,}건")

    # 필수 컬럼 확인
    required_cols = [
        "game_pk", "inning", "inning_topbot", "outs_when_up",
        "on_1b", "on_2b", "on_3b", "bat_score", "post_bat_score",
    ]
    missing = [c for c in required_cols if c not in pa.columns]
    if missing:
        print(f"[오류] 필수 컬럼 누락: {missing}", file=sys.stderr)
        sys.exit(1)

    # 주자 유무를 0/1로 변환 (NaN=빈 베이스→0, 선수ID 있음→1)
    for base_col in ["on_1b", "on_2b", "on_3b"]:
        pa[base_col] = pa[base_col].notna().astype(int)

    # outs_when_up을 정수로 변환
    pa["outs_when_up"] = pa["outs_when_up"].astype(int)

    # 하프이닝 그룹 키 생성
    pa["half_inning"] = (
        pa["game_pk"].astype(str) + "_"
        + pa["inning"].astype(str) + "_"
        + pa["inning_topbot"].astype(str)
    )

    # at_bat_number 또는 pitch_number 기준 정렬 (시간순)
    # Statcast는 역순으로 반환될 수 있으므로 정렬 필수
    sort_cols = ["game_pk", "inning", "at_bat_number"]
    if "at_bat_number" not in pa.columns:
        # fallback: pitch_number 사용
        sort_cols = ["game_pk", "inning", "pitch_number"]
    pa = pa.sort_values(sort_cols).reset_index(drop=True)

    # base-out 상태 키 생성
    pa["state_key"] = (
        pa["outs_when_up"].astype(str) + "_"
        + pa["on_1b"].astype(str)
        + pa["on_2b"].astype(str)
        + pa["on_3b"].astype(str)
    )

    print("[계산] 하프이닝별 잔여 득점 계산...")

    # 각 하프이닝의 총 득점 (마지막 타석의 post_bat_score - 첫 타석의 bat_score)
    # 방법: 하프이닝 그룹별로 마지막 post_bat_score를 구한 뒤,
    #        각 타석의 잔여 득점 = 하프이닝 최종 득점 - 타석 시작 시점 득점
    hi_groups = pa.groupby("half_inning", sort=False)

    # 하프이닝별 최종 득점 (마지막 타석의 post_bat_score)
    hi_final_score = hi_groups["post_bat_score"].transform("last")

    # 잔여 득점: 해당 타석 시점부터 하프이닝 끝까지 추가로 들어온 점수
    pa["runs_from_here"] = hi_final_score - pa["bat_score"]

    # 잔여 득점이 음수인 경우 제거 (데이터 오류 또는 정정 이벤트)
    neg_count = (pa["runs_from_here"] < 0).sum()
    if neg_count > 0:
        print(f"  [주의] 잔여 득점 < 0인 행 {neg_count}건 제거")
        pa = pa[pa["runs_from_here"] >= 0]

    # 3아웃 상태 제거 (이닝 종료 후 기록되는 것이므로 RE24 계산에 불필요)
    pa = pa[pa["outs_when_up"] < 3]

    print(f"         유효 타석: {len(pa):,}건")

    # ── 상태별 평균 잔여 득점 계산 ──────────────────────────────────────────
    print("[집계] 24개 base-out 상태별 평균 기대 득점...")

    state_stats = pa.groupby("state_key")["runs_from_here"].agg(["mean", "count"])
    state_stats = state_stats.rename(columns={"mean": "re24", "count": "n"})

    # 결과 딕셔너리 구성
    re24_matrix = {}
    for key in ALL_STATE_KEYS:
        if key in state_stats.index:
            re24_matrix[key] = round(state_stats.loc[key, "re24"], 2)
        else:
            # 데이터 부족으로 관측되지 않은 상태 (정상 시즌에서는 발생하지 않아야 함)
            print(f"  [경고] 상태 '{key}'에 대한 관측치 없음 — 0.00으로 설정")
            re24_matrix[key] = 0.00

    # ── 요약 출력 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RE24 매트릭스 계산 결과")
    print("=" * 60)

    # 표 형식 출력
    print(f"\n{'상태':<10} {'RE24':>8} {'관측수':>10}")
    print("-" * 30)
    for key in ALL_STATE_KEYS:
        n = int(state_stats.loc[key, "n"]) if key in state_stats.index else 0
        print(f"{key:<10} {re24_matrix[key]:>8.2f} {n:>10,}")

    total_pa = len(pa)
    print(f"\n총 유효 타석: {total_pa:,}")
    print(f"주루공 0아웃 평균: {re24_matrix['0_000']:.2f}")
    print(f"만루 0아웃 평균:   {re24_matrix['0_111']:.2f}")
    print(f"주루공 2아웃 평균: {re24_matrix['2_000']:.2f}")

    return re24_matrix


def save_re24_json(
    matrix: dict[str, float],
    year: int,
    output_dir: str,
) -> str:
    """RE24 매트릭스를 JSON 파일로 저장.

    기존 data/re24_{YYYY}.json과 동일한 포맷으로 저장한다.
    """
    output_path = os.path.join(output_dir, f"re24_{year}.json")

    payload = {
        "season": year,
        "source": "pybaseball statcast play-by-play (computed)",
        "computed_at": datetime.now().astimezone().isoformat(),
        "partial": False,
        "matrix": matrix,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[저장] {output_path}")
    return output_path


def main():
    """CLI 진입점."""
    parser = argparse.ArgumentParser(
        description="Statcast play-by-play 데이터로 RE24 매트릭스 계산",
    )
    parser.add_argument(
        "year",
        type=int,
        nargs="?",
        default=2024,
        help="MLB 시즌 연도 (기본값: 2024)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="매트릭스를 출력만 하고 파일 저장하지 않음",
    )
    args = parser.parse_args()

    year = args.year
    print(f"\n{'='*60}")
    print(f"RE24 계산: {year} 시즌")
    print(f"{'='*60}\n")

    # 1. 데이터 수집
    df = fetch_season_data(year)

    # 2. RE24 계산
    matrix = compute_re24(df)

    # 3. 저장 (--dry-run이 아닌 경우)
    if args.dry_run:
        print("\n[dry-run] 파일 저장을 건너뜁니다.")
        print("\nJSON 미리보기:")
        payload = {
            "season": year,
            "source": "pybaseball statcast play-by-play (computed)",
            "computed_at": datetime.now().astimezone().isoformat(),
            "partial": False,
            "matrix": matrix,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        save_re24_json(matrix, year, data_dir)

    print("\n완료.")


if __name__ == "__main__":
    main()
