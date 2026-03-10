"""
경로 관리 및 I/O 유틸리티

- ProjectPaths: 기간별 raw/processed 파일 경로 제공
- save_parquet / read_parquet: Parquet 저장·로드 (Windows Arrow 호환 처리)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ProjectPaths:
    """프로젝트 데이터 경로 (기간별 파일명 사용)"""

    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path

    def raw_csv_range(self, start: str, end: str) -> Path:
        """수집된 Statcast 원시 CSV 경로"""
        return self.raw_dir / f"statcast_{start}_to_{end}.csv"

    def processed_pitch_clean_range(self, start: str, end: str) -> Path:
        """전처리된 pitch 데이터 (parquet)"""
        return self.processed_dir / f"pitch_clean_{start}_to_{end}.parquet"

    def processed_pitch_umap_cluster_range(self, start: str, end: str) -> Path:
        """UMAP+HDBSCAN 임베딩 결과 (parquet)"""
        return self.processed_dir / f"pitch_umap_cluster_{start}_to_{end}.parquet"

    def processed_pitcher_profiles_range(self, start: str, end: str) -> Path:
        """투수 프로필 (parquet)"""
        return self.processed_dir / f"pitcher_profiles_{start}_to_{end}.parquet"

    def processed_batter_profiles_range(self, start: str, end: str) -> Path:
        """타자 프로필 (parquet)"""
        return self.processed_dir / f"batter_profiles_{start}_to_{end}.parquet"

    def processed_matchup_pitch_level_range(self, start: str, end: str) -> Path:
        """투수-타자 매치업 (투구 단위, parquet)"""
        return self.processed_dir / f"matchup_pitch_level_{start}_to_{end}.parquet"

    def processed_matchup_pair_level_range(self, start: str, end: str) -> Path:
        """투수-타자 매치업 (쌍 단위, parquet)"""
        return self.processed_dir / f"matchup_pair_level_{start}_to_{end}.parquet"


def find_project_root(start: Path, max_up: int = 7) -> Path:
    """data/processed, src 디렉터리가 있는 상위 폴더를 프로젝트 루트로 탐색"""
    cur = start.resolve()
    for _ in range(max_up):
        if (cur / "data" / "processed").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    # fallback: start의 상위
    return start.resolve().parent


def get_paths(project_root: Path | None = None) -> ProjectPaths:
    """ProjectPaths 인스턴스 생성 (project_root 미지정 시 자동 탐색)"""
    if project_root is None:
        project_root = find_project_root(Path(os.getcwd()))
    project_root = project_root.resolve()
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )


def log(msg: str) -> None:
    """타임스탬프와 함께 로그 출력"""
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}")


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    DataFrame을 Parquet으로 저장 (Windows + Arrow 호환 처리)

    PyArrow 문자열 타입과 fastparquet 조합에서 발생하는 'Unable to avoid copy' /
    UTF8 변환 오류를 방지하기 위해, 문자열/카테고리 컬럼을 Python object로 변환 후 저장.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df2 = df.copy()

    # 1) 모든 컬럼을 검토하며 PyArrow 문자열 컬럼을 순수 Python object로 변환
    for c in df2.columns:
        dt_str = str(df2[c].dtype).lower()

        # 문자열 계열(arrow 포함) / object / category는 모두 처리
        if ("string" in dt_str) or (dt_str == "object") or ("category" in dt_str) or ("arrow" in dt_str):
            # PyArrow 백킹을 완전히 제거하기 위해 to_numpy() + copy 사용
            try:
                # PyArrow 컬럼인 경우 먼저 numpy로 변환해서 PyArrow 스칼라 제거
                arr = df2[c].to_numpy(dtype=object, na_value=None, copy=True)
                # 리스트로 변환 후 다시 numpy array로 변환 (순수 Python 객체만 포함)
                converted = [None if v is pd.NA or pd.isna(v) else str(v) for v in arr]
                df2[c] = np.array(converted, dtype=object)
            except Exception:
                # 혹시 위 방법이 실패하면 기존 방식 사용
                s = df2[c].astype("object")
                s = s.fillna(None)
                s = s.apply(lambda v: None if v is None else str(v))
                df2[c] = np.array(s.to_list(), dtype=object)

    # 2) 저장 - pyarrow 엔진으로 시도, 실패하면 fastparquet 사용
    try:
        df2.to_parquet(path, index=False, engine="pyarrow")
    except Exception:
        df2.to_parquet(path, index=False, engine="fastparquet")


def read_parquet(path: Path) -> pd.DataFrame:
    """Parquet 파일 로드 (fastparquet 엔진 사용)"""
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Parquet file is 0 bytes: {path}")
    return pd.read_parquet(path, engine="fastparquet")
