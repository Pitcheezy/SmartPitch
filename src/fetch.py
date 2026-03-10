"""
Statcast 데이터 수집 (pybaseball)

지정 기간(YYYY-MM-DD)의 MLB Statcast pitch-by-pitch 데이터를 수집합니다.

참고: Baseball Savant API가 간헐적으로 잘못된 CSV를 반환하여
      pandas ParserError가 발생할 수 있음. 이 경우 on_bad_lines='skip'으로 우회.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class FetchConfig:
    """수집 설정 (향후 확장용)"""
    use_cache: bool = True  # pybaseball 캐시 사용


def _apply_statcast_csv_patch():
    """
    pybaseball statcast CSV 파싱 시 ParserError 방지.
    API가 반환하는 불완전한 CSV 행을 건너뛰도록 patch.
    """
    import pybaseball.datasources.statcast as statcast_ds

    _original = statcast_ds.get_statcast_data_from_csv

    def _patched(csv_content: str, null_replacement=None, known_percentages=None):
        if null_replacement is None:
            null_replacement = __import__("numpy").nan
        if known_percentages is None:
            known_percentages = []
        data = pd.read_csv(io.StringIO(csv_content), on_bad_lines="skip")
        return statcast_ds.postprocessing.try_parse_dataframe(
            data,
            parse_numerics=False,
            null_replacement=null_replacement,
            known_percentages=known_percentages,
        )

    statcast_ds.get_statcast_data_from_csv = _patched


def fetch_statcast_by_date(start_date: str, end_date: str, cfg: FetchConfig) -> pd.DataFrame:
    """
    기간별 Statcast pitch-by-pitch 데이터 수집

    Args:
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        cfg: 수집 설정

    Returns:
        Statcast pitch-by-pitch DataFrame (pybaseball 원본 컬럼)

    Raises:
        ValueError: 해당 기간에 데이터가 없을 경우
    """
    from pybaseball import statcast, cache

    _apply_statcast_csv_patch()

    if cfg.use_cache:
        cache.enable()
    df = statcast(start_dt=start_date, end_dt=end_date)

    if df is None or len(df) == 0:
        raise ValueError(f"해당 기간에 Statcast 데이터 없음: {start_date} ~ {end_date}")

    return df
