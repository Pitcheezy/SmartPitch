"""
bip_loader.py — BIP (Ball In Play) 확률 시즌별 JSON 로더.

re24_loader.py와 동일한 패턴으로, data/bip_probabilities_{YYYY}.json에서
인플레이 타구 결과 확률을 로드한다.

사용법:
    from src.bip_loader import load as load_bip

    probs = load_bip(2024)
    # => {"out": 0.68, "single": 0.16, "double": 0.05, "triple": 0.005, "home_run": 0.035}

    probs = load_average([2021, 2022, 2023, 2024, 2025])
    # => 5시즌 평균 확률

현재 하드코딩 값 (pitch_env.py / mdp_solver.py):
    out=0.70, single=0.15, double=0.10, home_run=0.05 (triple 없음)

실측 데이터로 교체하면 2루타/홈런 과대추정이 해소됨.
"""

import json
import os
from functools import lru_cache
from typing import Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# BIP 확률 필수 카테고리
REQUIRED_CATEGORIES = ["out", "single", "double", "triple", "home_run"]

# 기존 하드코딩 fallback (triple 포함 버전)
HARDCODED_FALLBACK = {
    "out": 0.70,
    "single": 0.15,
    "double": 0.10,
    "triple": 0.00,
    "home_run": 0.05,
}


@lru_cache(maxsize=8)
def load(season: Optional[int] = None) -> dict[str, float]:
    """시즌별 BIP 확률 로드.

    Args:
        season: MLB 시즌 연도. None이면 하드코딩 fallback 반환.

    Returns:
        {"out": 0.68, "single": 0.16, "double": 0.05, "triple": 0.005, "home_run": 0.035}

    Raises:
        FileNotFoundError: 해당 시즌 JSON 파일이 없을 때
        ValueError: 필수 카테고리가 누락되었거나 확률 합이 1에서 벗어날 때
    """
    if season is None:
        return HARDCODED_FALLBACK.copy()

    json_path = os.path.join(_DATA_DIR, f"bip_probabilities_{season}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"BIP 확률 파일 없음: {json_path}\n"
            f"  생성 방법: uv run scripts/compute_bip_probabilities.py {season}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    probs = data["probabilities"]

    # 필수 카테고리 검증
    missing = [c for c in REQUIRED_CATEGORIES if c not in probs]
    if missing:
        raise ValueError(f"BIP 확률에 누락된 카테고리: {missing} (파일: {json_path})")

    # 확률 합 검증 (±0.01 허용)
    total = sum(probs[c] for c in REQUIRED_CATEGORIES)
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"BIP 확률 합이 1.0에서 벗어남: {total:.4f} (파일: {json_path})"
        )

    return {c: probs[c] for c in REQUIRED_CATEGORIES}


def load_average(seasons: Optional[list[int]] = None) -> dict[str, float]:
    """여러 시즌의 BIP 확률 평균 계산.

    Args:
        seasons: 평균을 낼 시즌 목록. None이면 사용 가능한 전체 시즌.

    Returns:
        카테고리별 평균 확률 딕셔너리
    """
    if seasons is None:
        seasons = list_available_seasons()

    if not seasons:
        return HARDCODED_FALLBACK.copy()

    all_probs = [load(s) for s in seasons]

    avg = {}
    for c in REQUIRED_CATEGORIES:
        avg[c] = round(sum(p[c] for p in all_probs) / len(all_probs), 4)

    return avg


def list_available_seasons() -> list[int]:
    """사용 가능한 BIP 시즌 목록 반환 (오름차순)."""
    seasons = []
    if os.path.isdir(_DATA_DIR):
        for f in os.listdir(_DATA_DIR):
            if f.startswith("bip_probabilities_") and f.endswith(".json"):
                try:
                    year = int(f.replace("bip_probabilities_", "").replace(".json", ""))
                    seasons.append(year)
                except ValueError:
                    pass
    return sorted(seasons)
