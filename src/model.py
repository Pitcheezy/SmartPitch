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
    """PyTorch 학습용 데이터셋"""
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

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader, int, int]:
        """
        [내부 메서드] X(One-Hot Encoding)와 y(Label Encoding)를 만들고
        Train/Val DataLoader를 생성
        """
        print("데이터 전처리 및 인코딩 중...")
        # 1. count_state 만들기 (볼-스트라이크_아웃_주자123)
        self.df['count_state'] = (
            self.df['balls'].astype(int).astype(str) + "-" + 
            self.df['strikes'].astype(int).astype(str) + "_" + 
            self.df['outs_when_up'].astype(int).astype(str) + "_" + 
            self.df['on_1b'] + self.df['on_2b'] + self.df['on_3b']
        )
        
        # 2. X, y 정의
        X_raw = self.df[['count_state', 'mapped_pitch_name', 'zone']]
        y_raw = self.df['description']
        
        # X: One-Hot Encoding
        X_encoded = pd.get_dummies(X_raw, columns=['count_state', 'mapped_pitch_name', 'zone'])
        self.feature_columns = X_encoded.columns.tolist()
        
        # y: Label Encoding
        y_encoded = self.label_encoder.fit_transform(y_raw)
        self.target_classes = self.label_encoder.classes_.tolist()
        
        # --- 수정된 부분: 빈도가 너무 적은(예: 1) 클래스가 포함된 데이터 샘플 제거 ---
        # stratify=y_encoded 를 위해 최소 2개 이상의 샘플을 가진 클래스만 남김
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

    def run_modeling_pipeline(self, epochs: int = 50, upload_artifact: bool = True):
        """
        데이터 준비부터 모델 학습, W&B 로깅 및 아티팩트 업로드까지 일괄 수행
        """
        self.train_model(epochs=epochs)
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
