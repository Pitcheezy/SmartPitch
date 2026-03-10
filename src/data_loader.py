import pandas as pd
from pybaseball import statcast_pitcher, playerid_lookup
import wandb
import os

class PitchDataLoader:
    """
    pybaseball을 이용해 특정 투수의 데이터를 수집하고 전처리한 뒤,
    결과물을 W&B Artifact로 저장 및 제공하는 클래스
    """

    def __init__(self, first_name: str, last_name: str, start_date: str, end_date: str, wandb_config: dict = None):
        """
        초기화 메서드
        :param first_name: 투수 이름 (예: 'gerrit')
        :param last_name: 투수 성 (예: 'cole')
        :param start_date: 수집 시작일 (예: '2019-03-28')
        :param end_date: 수집 종료일 (예: '2019-09-29')
        :param wandb_config: W&B 초기화 및 업로드에 필요한 설정 정보 딕셔너리
        """
        self.first_name = first_name
        self.last_name = last_name
        self.start_date = start_date
        self.end_date = end_date
        self.wandb_config = wandb_config
        self.raw_data = None
        self.processed_data = None

    def _fetch_data(self) -> pd.DataFrame:
        """
        [내부 메서드] playerid_lookup 및 statcast_pitcher를 사용해 원본 데이터를 다운로드
        """
        print(f"[{self.first_name.capitalize()} {self.last_name.capitalize()}] 선수 ID 검색 중...")
        player_info = playerid_lookup(self.last_name, self.first_name)
        if player_info.empty:
            raise ValueError(f"선수 정보를 찾을 수 없습니다: {self.first_name} {self.last_name}")
            
        mlbam_id = player_info['key_mlbam'].values[0]
        
        print(f"[{self.start_date} ~ {self.end_date}] 투구 데이터 수집 중...")
        df = statcast_pitcher(self.start_date, self.end_date, player_id=mlbam_id)
        
        if df.empty:
            raise ValueError("수집된 데이터가 없습니다. 날짜나 선수 이름을 확인해주세요.")
            
        self.raw_data = df
        return df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [내부 메서드] 수집된 데이터의 결측치 제거, 주자 상태(NaN -> 0/1 문자열) 변환, 
        필요한 피처 추출 등 전처리를 수행
        """
        print("데이터 전처리 진행 중...")
        
        # 주자 상태 결측치(NaN)를 0과 1 문자열로 치환
        df['on_1b'] = df['on_1b'].notna().astype(int).astype(str)
        df['on_2b'] = df['on_2b'].notna().astype(int).astype(str)
        df['on_3b'] = df['on_3b'].notna().astype(int).astype(str)
        
        # 필요한 피처와 메타데이터 정의
        features = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'release_pos_x', 'release_pos_z']
        meta = ['balls', 'strikes', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b', 'description', 'zone', 'batter']
        
        # 필요한 컬럼만 추출 후 결측치 제거
        df_processed = df[features + meta].dropna()
        
        print(f"전처리 완료: 총 {len(df_processed)} 개의 투구 데이터 확보")
        self.processed_data = df_processed
        return df_processed

    def upload_to_wandb(self, artifact_name: str = "pitch_data", dataset_type: str = "dataset") -> None:
        """
        전처리가 완료된 데이터를 로컬 CSV로 임시 저장한 뒤, W&B Artifact로 업로드
        :param artifact_name: W&B에 등록될 Artifact 이름
        :param dataset_type: Artifact 타입
        """
        if self.processed_data is None:
            raise ValueError("업로드할 전처리된 데이터가 없습니다. load_and_prepare_data()를 먼저 실행하세요.")
            
        if not wandb.run:
            print("W&B run이 활성화되어 있지 않습니다. init을 시도합니다.")
            if self.wandb_config:
                wandb.init(**self.wandb_config)
            else:
                wandb.init(project="SmartPitch-Portfolio", job_type="upload-dataset")

        print("W&B Artifact 업로드 준비 중...")
        
        # 로컬 CSV 임시 저장
        csv_filename = f"{self.first_name}_{self.last_name}_processed.csv"
        self.processed_data.to_csv(csv_filename, index=False)
        
        # W&B Artifact 생성 및 파일 추가
        artifact = wandb.Artifact(name=artifact_name, type=dataset_type)
        artifact.add_file(csv_filename)
        
        # Artifact 로깅
        wandb.log_artifact(artifact)
        print(f"W&B Artifacts 서버에 데이터 업로드 완료: {artifact_name}")
        
        # 임시 파일 삭제 (선택적)
        if os.path.exists(csv_filename):
            os.remove(csv_filename)

    def load_and_prepare_data(self, upload_artifact: bool = False, artifact_name: str = "pitch_data") -> pd.DataFrame:
        """
        전체 데이터 로드 파이프라인 실행
        1. _fetch_data() 호출
        2. _preprocess_data() 호출
        3. 조건에 따라 upload_to_wandb() 호출
        :return: 전처리가 완료된 데이터프레임
        """
        df_raw = self._fetch_data()
        df_processed = self._preprocess_data(df_raw)
        
        if upload_artifact:
            self.upload_to_wandb(artifact_name=artifact_name)
            
        return df_processed
