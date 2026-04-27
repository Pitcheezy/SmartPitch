"""
tests/test_re24_loader.py — RE24 시즌별 로더 유닛 테스트

검증 항목:
  1. 존재하는 시즌 JSON 정상 로드 (24개 키, 값 범위)
  2. 존재하지 않는 시즌 → FileNotFoundError
  3. get_state_key() 포맷 검증
  4. lru_cache 동작 확인 (동일 객체 반환)
  5. list_available_seasons() 반환값 검증
  6. PitchEnv / MDPOptimizer 와 동일한 키 형식 사용 확인
"""

import pytest
from src.re24_loader import load, get_state_key, list_available_seasons, load_matrices_for_years


class TestGetStateKey:
    """get_state_key() 키 포맷 검증."""

    def test_basic_format(self):
        assert get_state_key(0, 0, 0, 0) == "0_000"
        assert get_state_key(2, 1, 1, 1) == "2_111"
        assert get_state_key(1, 1, 0, 1) == "1_101"

    def test_all_24_keys_unique(self):
        keys = {
            get_state_key(outs, b1, b2, b3)
            for outs in range(3)
            for b1 in range(2)
            for b2 in range(2)
            for b3 in range(2)
        }
        assert len(keys) == 24


class TestLoad:
    """load() 함수 검증."""

    def test_load_2024(self):
        matrix = load(2024)
        assert len(matrix) == 24
        # 0_000 값 범위 확인 (0.3 ~ 0.7 사이여야 합리적)
        assert 0.3 <= matrix["0_000"] <= 0.7
        # 2_111 값은 0_000보다 커야 함 (주자 만루가 기대 실점 높음)
        assert matrix["2_111"] > matrix["2_000"]

    def test_load_2023(self):
        matrix = load(2023)
        assert len(matrix) == 24
        assert 0.3 <= matrix["0_000"] <= 0.7

    def test_load_2019(self):
        matrix = load(2019)
        assert len(matrix) == 24
        assert 0.3 <= matrix["0_000"] <= 0.7

    def test_load_nonexistent_season_raises(self):
        with pytest.raises(FileNotFoundError, match="RE24 JSON not found"):
            load(1900)

    def test_load_cache_returns_same_object(self):
        """lru_cache가 동일 시즌에 대해 같은 객체를 반환하는지 확인."""
        load.cache_clear()  # 테스트 격리
        m1 = load(2024)
        m2 = load(2024)
        assert m1 is m2  # 캐시된 동일 객체

    def test_all_values_non_negative(self):
        """RE24 값은 모두 0 이상이어야 함."""
        matrix = load(2024)
        for key, val in matrix.items():
            assert val >= 0.0, f"{key}={val} is negative"

    def test_monotonicity_outs(self):
        """아웃 수가 증가하면 같은 주자 상태에서 RE24가 감소해야 함."""
        matrix = load(2024)
        for runners in ["000", "100", "010", "001", "110", "101", "011", "111"]:
            for outs in range(2):
                key_now = get_state_key(outs, int(runners[0]), int(runners[1]), int(runners[2]))
                key_next = get_state_key(outs + 1, int(runners[0]), int(runners[1]), int(runners[2]))
                assert matrix[key_now] >= matrix[key_next], (
                    f"RE24({key_now})={matrix[key_now]} < RE24({key_next})={matrix[key_next]}"
                )


class TestListAvailableSeasons:
    """list_available_seasons() 검증."""

    def test_returns_sorted_list(self):
        seasons = list_available_seasons()
        assert isinstance(seasons, list)
        assert seasons == sorted(seasons)

    def test_known_seasons_present(self):
        seasons = list_available_seasons()
        assert 2024 in seasons
        assert 2023 in seasons
        assert 2019 in seasons


class TestLoadMatricesForYears:
    """load_matrices_for_years() 검증."""

    def test_load_multiple(self):
        result = load_matrices_for_years([2023, 2024])
        assert 2023 in result
        assert 2024 in result
        assert len(result[2023]) == 24
        assert len(result[2024]) == 24

    def test_different_seasons_have_different_values(self):
        """다른 시즌은 (최소 일부) 다른 값을 가져야 함."""
        result = load_matrices_for_years([2023, 2024])
        # 완전히 동일할 수는 없음
        differences = sum(
            1 for k in result[2023]
            if abs(result[2023][k] - result[2024][k]) > 0.001
        )
        assert differences > 0, "2023과 2024 RE24가 완전 동일 — 데이터 오류 의심"
