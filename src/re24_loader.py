"""
RE24 (Run Expectancy 24-state) 시즌별 로더 모듈.

시즌별 RE24 매트릭스를 JSON에서 로드하고, PitchEnv / MDPOptimizer 등
소비자 코드에 일관된 인터페이스를 제공한다.

키 포맷: "{outs}_{on_1b}{on_2b}{on_3b}" (예: "0_000", "2_111")
"""

import json
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SEASON = 2024
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_state_key(outs: int, on_1b: int, on_2b: int, on_3b: int) -> str:
    """RE24 상태 키 생성. PitchEnv / MDPOptimizer 공용."""
    return f"{outs}_{on_1b}{on_2b}{on_3b}"


@lru_cache(maxsize=8)
def load(season: Optional[int] = None) -> dict[str, float]:
    """시즌별 RE24 매트릭스를 JSON에서 로드.

    Args:
        season: MLB 시즌 연도 (예: 2024). None이면 DEFAULT_SEASON 사용 + 경고.

    Returns:
        24개 상태의 기대 실점 딕셔너리.

    Raises:
        FileNotFoundError: 해당 시즌 JSON이 없는 경우.
    """
    if season is None:
        logger.warning(
            "RE24 season 미지정 — 기본값 %d 사용. "
            "명시적으로 season을 전달하세요.",
            DEFAULT_SEASON,
        )
        season = DEFAULT_SEASON

    json_path = os.path.join(_DATA_DIR, f"re24_{season}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"RE24 JSON not found: {json_path}. "
            f"사용 가능한 시즌: {list_available_seasons()}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrix = data["matrix"]

    # 24개 상태 검증
    expected_keys = {
        get_state_key(outs, on_1b, on_2b, on_3b)
        for outs in range(3)
        for on_1b in range(2)
        for on_2b in range(2)
        for on_3b in range(2)
    }
    actual_keys = set(matrix.keys())
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        raise ValueError(
            f"RE24 JSON ({season}) 키 불일치. "
            f"누락: {missing}, 초과: {extra}"
        )

    logger.info("RE24 매트릭스 로드: season=%d, 상태 수=%d", season, len(matrix))
    return matrix


def list_available_seasons() -> list[int]:
    """data/ 디렉토리에서 사용 가능한 RE24 시즌 목록 반환."""
    seasons = []
    if os.path.isdir(_DATA_DIR):
        for fname in os.listdir(_DATA_DIR):
            if fname.startswith("re24_") and fname.endswith(".json"):
                try:
                    year = int(fname.replace("re24_", "").replace(".json", ""))
                    seasons.append(year)
                except ValueError:
                    pass
    return sorted(seasons)


def load_matrices_for_years(years: list[int]) -> dict[int, dict[str, float]]:
    """여러 시즌의 RE24 매트릭스를 한 번에 로드 (캐시 활용).

    Args:
        years: 시즌 연도 리스트. 예: [2023, 2024]

    Returns:
        {year: matrix} 딕셔너리.
    """
    return {year: load(year) for year in sorted(set(years))}
