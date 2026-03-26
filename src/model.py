"""
model.py — 투구 결과 전이 확률 예측 MLP 모델

역할:
    (볼카운트 + 구종 + 코스 + 타자유형 + 투수유형) → 투구 결과 확률 분포 예측
    MDP Solver와 RL 환경(PitchEnv)이 이 모델의 predict_proba()를 호출하여
    각 행동(구종+코스)의 기대 결과를 추정합니다.

입력 피처 (One-Hot Encoding):
    - count_state    : "3-2_2_111" 형식 (볼-스트라이크_아웃_주자상태), 12×3×8 = 288가지
    - mapped_pitch_name: 구종명 (Fastball/Slider 등, clustering.py 결과)
    - zone           : 투구 코스 (1~14, 존 번호)
    - batter_cluster : 타자 유형 (0~7, batter_clusters_2023.csv)
    - pitcher_cluster: 투수 유형 (0~K-1, pitcher_clusters_2023.csv, 현재 K=4)

출력:
    - 투구 결과 클래스 확률 (called_strike, ball, swinging_strike, hit_into_play 등 ~12종)
    - predict_proba(): softmax 확률 벡터 반환

MLP 구조:
    Input(가변) → Linear(128)-BN-ReLU-Dropout(0.2)
              → Linear(64) -BN-ReLU-Dropout(0.2)
              → Linear(output_dim)  ← 클래스 수만큼

저장 파일:
    best_transition_model.pth  ← val_loss 기준 최고 체크포인트

개선 필요 사항:
    - epochs=5는 너무 적음 → val_acc 47% 수준 (목표: 65%)
    - 20~30 epoch + EarlyStopping 권장
    - 범용 모델: 단일 투수 데이터 → 전 MLB 데이터로 재학습 필요
"""
import pandas as pd
import numpy as np
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any, List

class PitchDataset(Dataset):
    """PyTorch 학습용 데이터셋 래퍼 — numpy 배열을 Tensor로 변환"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    """
    전이 확률 예측을 위한 다층 퍼셉트론 (MLP) 모델
    (특정 볼카운트/아웃/주자 상황에서 특정 구종을 특정 코스에 던졌을 때의 결과 예측)
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.2):
        super(MLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        # 은닉층 추가
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        # 출력층
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TransitionProbabilityModel:
    """
    전처리 및 클러스터링이 완료된 데이터를 받아 PyTorch MLP 모델로 
    투구 결과(Transition) 확률을 예측하고 학습하는 클래스
    """
    def __init__(self, df: pd.DataFrame, batch_size: int = 64, lr: float = 0.001):
        self.df = df.copy()
        self.batch_size = batch_size
        self.lr = lr
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_classes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 기기: {self.device}")

    @classmethod
    def load_from_checkpoint(
        cls,
        model_path: str,
        feature_columns_path: str,
        target_classes_path: str,
    ) -> "TransitionProbabilityModel":
        """
        저장된 가중치(.pth) / 특징 컬럼(.json) / 결과 클래스(.json)를 로드하여
        추론 전용 인스턴스를 반환합니다.

        universal_model_trainer.py 가 생성한 아래 세 파일을 main.py에서
        USE_UNIVERSAL_MODEL=True 로 사용할 때 호출됩니다.

            best_transition_model_universal.pth
            data/feature_columns_universal.json
            data/target_classes_universal.json

        :param model_path:           .pth 파일 절대/상대 경로
        :param feature_columns_path: feature_columns_universal.json 경로
        :param target_classes_path:  target_classes_universal.json 경로
        :return: model.eval() 상태의 TransitionProbabilityModel 인스턴스
        """
        import json

        # 빈 DataFrame으로 인스턴스 생성 — _prepare_data() 호출 없이 추론만 사용
        instance = cls(df=pd.DataFrame(), batch_size=256)

        with open(feature_columns_path, 'r', encoding='utf-8') as f:
            instance.feature_columns = json.load(f)
        with open(target_classes_path, 'r', encoding='utf-8') as f:
            instance.target_classes = json.load(f)

        # LabelEncoder 재구성 — predict_proba 후 클래스 이름 조회에 사용
        instance.label_encoder.classes_ = np.array(instance.target_classes)

        input_dim  = len(instance.feature_columns)
        output_dim = len(instance.target_classes)

        instance.model = MLP(input_dim, output_dim).to(instance.device)
        instance.model.load_state_dict(
            torch.load(model_path, map_location=instance.device, weights_only=True)
        )
        instance.model.eval()

        print(f"[UniversalModel] 로드 완료 — input_dim={input_dim}, output_dim={output_dim}")
        return instance

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader, int, int]:
        """
        [내부 메서드] X(One-Hot Encoding)와 y(Label Encoding)를 만들고
        Train/Val DataLoader를 생성
        """
        print("데이터 전처리 및 인코딩 중...")

        # ── [타자 군집 merge] ────────────────────────────────────────────────────
        # batter_clusters_2023.csv: {batter_id, stand, cluster}
        # self.df의 'batter' 컬럼(MLBAM ID)으로 left join → 'batter_cluster' 컬럼 생성
        # 매칭 안 되면 0 (unknown batter → 평균적 타자로 취급)
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "batter_clusters_2023.csv")
        try:
            if os.path.exists(csv_path):
                df_clusters = pd.read_csv(csv_path)
                self.df = self.df.merge(df_clusters[['batter_id', 'cluster']], left_on='batter', right_on='batter_id', how='left')
                self.df['batter_cluster'] = self.df['cluster'].fillna(0).astype(int).astype(str)
                print(f"[Model] 타자 군집(batter_cluster) 병합 완료")
            else:
                self.df['batter_cluster'] = "0"
                print(f"[Model] Warning: '{csv_path}' 없음. 기본 군집(0) 할당.")
        except Exception as e:
            self.df['batter_cluster'] = "0"
            print(f"[Model] Error loading batter cluster csv: {e}")

        # ── [투수 군집 merge] ────────────────────────────────────────────────────
        # pitcher_clusters_2023.csv: {pitcher_id, cluster}
        # self.df의 'pitcher' 컬럼(MLBAM ID)으로 left join → 'pitcher_cluster' 컬럼 생성
        # 컬럼 충돌 방지를 위해 pitcher 쪽 cluster를 p_cluster로 rename 후 merge
        # 매칭 안 되면 0 (2023 시즌 500구 미만 투수 등)
        pitcher_csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "pitcher_clusters_2023.csv")
        try:
            if os.path.exists(pitcher_csv_path):
                df_pitcher_clusters = pd.read_csv(pitcher_csv_path)[['pitcher_id', 'cluster']].rename(
                    columns={'cluster': 'p_cluster'}  # batter merge의 'cluster' 컬럼과 충돌 방지
                )
                self.df = self.df.merge(df_pitcher_clusters, left_on='pitcher', right_on='pitcher_id', how='left')
                self.df['pitcher_cluster'] = self.df['p_cluster'].fillna(0).astype(int).astype(str)
                print(f"[Model] 투수 군집(pitcher_cluster) 병합 완료")
            else:
                self.df['pitcher_cluster'] = "0"
                print(f"[Model] Warning: '{pitcher_csv_path}' 없음. 기본 군집(0) 할당.")
        except Exception as e:
            self.df['pitcher_cluster'] = "0"
            print(f"[Model] Error loading pitcher cluster csv: {e}")

        # ── [count_state 생성] ───────────────────────────────────────────────────
        # 형식: "볼-스트라이크_아웃_123주자"  예) "3-2_2_111"
        # MDP 상태 키의 앞 3파트와 동일한 형식 → MDP와 모델이 동일한 상태 표현 사용
        self.df['count_state'] = (
            self.df['balls'].astype(int).astype(str) + "-" +
            self.df['strikes'].astype(int).astype(str) + "_" +
            self.df['outs_when_up'].astype(int).astype(str) + "_" +
            self.df['on_1b'] + self.df['on_2b'] + self.df['on_3b']
        )

        # ── [One-Hot Encoding] ───────────────────────────────────────────────────
        # 5개 카테고리 변수를 모두 원-핫으로 변환
        # 최종 입력 차원: count_state(최대288) + pitch_name(4~6) + zone(14) + batter(8) + pitcher(4) ≈ 320차원
        X_raw = self.df[['count_state', 'mapped_pitch_name', 'zone', 'batter_cluster', 'pitcher_cluster']]
        y_raw = self.df['description']

        X_encoded = pd.get_dummies(X_raw, columns=['count_state', 'mapped_pitch_name', 'zone', 'batter_cluster', 'pitcher_cluster'])
        self.feature_columns = X_encoded.columns.tolist()  # MDP/RL에서 동일 컬럼 순서로 입력 생성 시 사용

        # ── [Label Encoding] ─────────────────────────────────────────────────────
        # 투구 결과(description)를 정수 레이블로 변환
        # 예: "called_strike"→0, "ball"→1, "hit_into_play"→2, ...
        y_encoded = self.label_encoder.fit_transform(y_raw)
        self.target_classes = self.label_encoder.classes_.tolist()

        # ── [희귀 클래스 제거] ────────────────────────────────────────────────────
        # train_test_split의 stratify 옵션은 각 클래스가 최소 2개 이상의 샘플을 요구함
        # 1개뿐인 클래스(예: 매우 드문 결과 코드)는 stratify 오류를 일으키므로 제거
        class_counts = pd.Series(y_encoded).value_counts()
        valid_classes = class_counts[class_counts >= 2].index

        valid_mask = pd.Series(y_encoded).isin(valid_classes).values
        X_encoded_valid = X_encoded.iloc[valid_mask]
        y_encoded_valid = y_encoded[valid_mask]

        input_dim = len(self.feature_columns)
        output_dim = len(self.target_classes)
        print(f"입력 특징 차원: {input_dim}, 출력 클래스 수: {output_dim}")
        
        # 3. Train / Val 분리
        X_train, X_val, y_train, y_val = train_test_split(
            X_encoded_valid.values, y_encoded_valid, test_size=0.2, random_state=42, stratify=y_encoded_valid
        )
        
        # 4. DataLoader 생성
        train_dataset = PitchDataset(X_train, y_train)
        val_dataset = PitchDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, input_dim, output_dim

    def train_model(self, epochs: int = 50, hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.2):
        """
        PyTorch MLP 모델을 학습하고 검증 성능을 W&B에 로깅
        """
        train_loader, val_loader, input_dim, output_dim = self._prepare_data()
        
        self.model = MLP(input_dim, output_dim, hidden_dims, dropout_rate).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        print(f"모델 학습 시작 (총 {epochs} Epochs)...")
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
                    
            val_loss /= len(val_loader.dataset)
            val_acc = correct / total
            
            # --- W&B 로깅 ---
            if wandb.run:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                })
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                
            # 최고 성능 모델 가중치 임시 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_transition_model.pth")
                
        print("학습 완료! 최고 검증 손실(Best Val Loss):", f"{best_val_loss:.4f}")
        
        # 가장 좋았던 모델 로드
        self.model.load_state_dict(torch.load("best_transition_model.pth", weights_only=True))

    def upload_model_artifact(self, artifact_name: str = "transition_mlp_model"):
        """
        학습된 모델의 가중치 파일(.pth)을 W&B Artifact로 업로드
        """
        if not os.path.exists("best_transition_model.pth"):
            raise FileNotFoundError("저장된 모델 파일이 없습니다. train_model()을 먼저 실행하세요.")
            
        if wandb.run:
            print("W&B Artifact로 모델 업로드 중...")
            artifact = wandb.Artifact(name=artifact_name, type="model")
            artifact.add_file("best_transition_model.pth")
            wandb.log_artifact(artifact)
            print(f"모델 업로드 완료: {artifact_name}")
        else:
            print("W&B run이 활성화되어 있지 않아 모델 아티팩트를 업로드할 수 없습니다.")

    def run_modeling_pipeline(self, epochs: int = 50, hidden_dims: List[int] = None, upload_artifact: bool = True):
        """
        데이터 준비부터 모델 학습, W&B 로깅 및 아티팩트 업로드까지 일괄 수행
        hidden_dims가 None이면 train_model() 기본값([128, 64]) 사용
        """
        kwargs = {"epochs": epochs}
        if hidden_dims is not None:
            kwargs["hidden_dims"] = hidden_dims
        self.train_model(**kwargs)
        if upload_artifact:
            self.upload_model_artifact()
            
    def predict_proba(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        추후 8장 MDP 연산에서 특정 상황(1줄짜리 데이터)에 대한 결과를 예측할 때 사용하는 추론 메서드
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(input_df.values).to(self.device)
            outputs = self.model(X_tensor)
            # Softmax를 통과시켜 확률 값으로 변환
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs
