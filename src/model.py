"""
model.py — 투구 결과 전이 확률 예측 MLP 모델

역할:
    (볼카운트 + 구종 + 코스 + 타자유형 + 투수유형) → 투구 결과 확률 분포 예측
    MDP Solver와 RL 환경(PitchEnv)이 이 모델의 predict_proba()를 호출하여
    각 행동(구종+코스)의 기대 결과를 추정합니다.

입력 피처 (총 ~40차원):
    수치 피처 6개 (정수, count_state one-hot 288차원 대체):
    - balls, strikes, outs     : 볼카운트/아웃 수 (0~3, 0~2, 0~2)
    - on_1b, on_2b, on_3b      : 주자 상태 (0/1)
    범주형 피처 One-Hot:
    - mapped_pitch_name: 구종명 (Fastball/Slider 등, clustering.py 결과)
    - zone           : 투구 코스 (1~14, 존 번호)
    - batter_cluster : 타자 유형 (0~7, batter_clusters_2023.csv)
    - pitcher_cluster: 투수 유형 (0~K-1, pitcher_clusters_2023.csv, 현재 K=4)

출력:
    - 투구 결과 클래스 확률 (ball/strike/foul/hit_into_play 4종, hit_by_pitch 제거됨)
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
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple, Dict, Any, List, Optional

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
        model_config_path: Optional[str] = None,
    ) -> "TransitionProbabilityModel":
        """
        저장된 가중치(.pth) / 특징 컬럼(.json) / 결과 클래스(.json)를 로드하여
        추론 전용 인스턴스를 반환합니다.

        universal_model_trainer.py 가 생성한 아래 파일들을 main.py에서
        USE_UNIVERSAL_MODEL=True 로 사용할 때 호출됩니다.

            best_transition_model_universal.pth
            data/feature_columns_universal.json
            data/target_classes_universal.json
            data/model_config_universal.json   ← hidden_dims/dropout_rate (선택)

        :param model_path:           .pth 파일 절대/상대 경로
        :param feature_columns_path: feature_columns_universal.json 경로
        :param target_classes_path:  target_classes_universal.json 경로
        :param model_config_path:    model_config_universal.json 경로 (없으면 기본값 사용)
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

        # 아키텍처 설정 로드 — 저장된 hidden_dims와 MLP가 반드시 일치해야 state_dict 로드 가능
        hidden_dims   = [128, 64]  # fallback 기본값
        dropout_rate  = 0.2
        if model_config_path and os.path.exists(model_config_path):
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            hidden_dims  = model_config.get("hidden_dims",  hidden_dims)
            dropout_rate = model_config.get("dropout_rate", dropout_rate)
            print(f"[UniversalModel] 아키텍처 로드: hidden_dims={hidden_dims}, dropout={dropout_rate}")
        else:
            print(f"[UniversalModel] model_config 없음 — 기본값 사용: hidden_dims={hidden_dims}")

        instance.model = MLP(input_dim, output_dim, hidden_dims, dropout_rate).to(instance.device)
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

        # ── [hit_by_pitch 제거] ───────────────────────────────────────────────────
        # 전체의 0.3%이며 투수의 의도적 선택이 아닌 비의도적 결과.
        # 극소수 클래스로 인한 학습 불균형(F1=0.000)을 방지하기 위해 제거.
        # X_num/X_cat 구성 전에 필터링해야 인덱스 불일치 없음.
        _hbp_mask = self.df['description'] != 'hit_by_pitch'
        _removed  = (~_hbp_mask).sum()
        if _removed > 0:
            self.df = self.df[_hbp_mask].reset_index(drop=True)
            print(f"  hit_by_pitch 제거: {_removed:,}건 → 남은 데이터: {len(self.df):,}건")

        # ── [수치 피처: 볼카운트/아웃/주자] ─────────────────────────────────────
        # count_state one-hot(288차원) 대신 6개 정수로 직접 표현 (B안 입력 재설계)
        # 카운트 간 일반화 가능: "2-2_2_000"과 "1-2_2_000"의 공통 패턴을 학습
        # on_1b/2b/3b는 "0"/"1" 문자열(data_loader 출력) 또는 0/1 정수 모두 허용
        X_num = pd.DataFrame({
            'balls':   self.df['balls'].astype(int),
            'strikes': self.df['strikes'].astype(int),
            'outs':    self.df['outs_when_up'].astype(int),
            'on_1b':   self.df['on_1b'].astype(int),
            'on_2b':   self.df['on_2b'].astype(int),
            'on_3b':   self.df['on_3b'].astype(int),
        })

        # 실험적 파생 피처: 해당 컬럼이 df에 있을 때만 추가 (Exp3 feature engineering용)
        _optional_feats = ['is_two_strike', 'is_first_pitch', 'balls_minus_strikes', 'zone_row', 'zone_col']
        for _feat in _optional_feats:
            if _feat in self.df.columns:
                X_num[_feat] = self.df[_feat].astype(float)

        # ── [One-Hot Encoding: 구종/존/타자군집/투수군집] ────────────────────────
        # 최종 입력 차원: 수치(6) + pitch_name(9) + zone(13) + batter(8) + pitcher(4) ≈ 40차원
        X_cat = self.df[['mapped_pitch_name', 'zone', 'batter_cluster', 'pitcher_cluster']]
        X_cat_encoded = pd.get_dummies(X_cat, columns=['mapped_pitch_name', 'zone', 'batter_cluster', 'pitcher_cluster'])

        y_raw = self.df['description']

        # ── [수치 + 카테고리 결합] ──────────────────────────────────────────────
        X_encoded = pd.concat([X_num.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1).astype(float)
        self.feature_columns = X_encoded.columns.tolist()  # MDP/RL에서 동일 컬럼 순서로 입력 구성 시 사용

        # ── [Label Encoding] ─────────────────────────────────────────────────────
        # 투구 결과(description)를 정수 레이블로 변환
        # 예: "ball"→0, "foul"→1, "hit_into_play"→2, "strike"→3 (4클래스)
        y_encoded = self.label_encoder.fit_transform(y_raw)
        self.target_classes = self.label_encoder.classes_.tolist()

        # ── [클래스 분포 출력] ────────────────────────────────────────────────────
        total_samples = len(y_encoded)
        print(f"\n[클래스 분포 — 전체 {total_samples:,}건]")
        for label_idx, cls_name in enumerate(self.target_classes):
            cnt = int((y_encoded == label_idx).sum())
            print(f"  {cls_name:<20}: {cnt:>7}건  ({cnt / total_samples:.1%})")
        print()

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

    def train_model(self, epochs: int = 50, hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.2,
                    patience: int = 5, use_lr_scheduler: bool = False, use_class_weights: bool = False):
        """
        PyTorch MLP 모델을 학습하고 검증 성능을 W&B에 로깅.
        val_loss 기준 EarlyStopping 적용 (patience=5 기본값).
        use_lr_scheduler=True  : ReduceLROnPlateau 스케줄러 적용.
        use_class_weights=True : 클래스 빈도 역수로 CrossEntropyLoss 가중치 설정.
                                 소수 클래스(foul, hit_into_play) recall 개선 목적.
        """
        train_loader, val_loader, input_dim, output_dim = self._prepare_data()

        self.model = MLP(input_dim, output_dim, hidden_dims, dropout_rate).to(self.device)

        if use_class_weights:
            # 훈련 데이터 클래스 빈도 계산 → 역수 가중치 (평균=1 정규화)
            y_train_np = train_loader.dataset.y.numpy()
            counts = np.bincount(y_train_np, minlength=output_dim).astype(float)
            weights = 1.0 / np.where(counts > 0, counts, 1.0)
            weights = weights / weights.mean()
            weight_tensor = torch.FloatTensor(weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print("클래스 가중치 적용:")
            for i, cls in enumerate(self.target_classes):
                print(f"  {cls:<20}: count={int(counts[i]):>7}  weight={weights[i]:.4f}")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        scheduler = None
        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5
            )
            print(f"LR 스케줄러 활성화: ReduceLROnPlateau (factor=0.5, patience=3)")

        print(f"모델 학습 시작 (최대 {epochs} Epochs, EarlyStopping patience={patience})...")
        best_val_loss = float('inf')
        best_val_acc  = 0.0
        patience_counter = 0

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

            # --- LR 스케줄러 업데이트 ---
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step(val_loss)

            # --- W&B 로깅 ---
            if wandb.run:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr,
                })

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

            # --- EarlyStopping + 최고 모델 저장 ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc  = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_transition_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: epoch {epoch}에서 중단 (patience={patience}, best_val_loss={best_val_loss:.4f})")
                    if wandb.run:
                        wandb.log({"early_stop_epoch": epoch})
                    break

        print(f"학습 완료! 최고 검증 손실(Best Val Loss): {best_val_loss:.4f}  |  Best Val Acc: {best_val_acc:.4f}")

        # 가장 좋았던 모델 로드
        self.model.load_state_dict(torch.load("best_transition_model.pth", weights_only=True))

        # --- Confusion Matrix + Per-class Metrics + Top-K Accuracy (val set 기준) ---
        self.model.eval()
        all_preds, all_labels = [], []
        all_outputs = []  # Top-K accuracy 계산용 logit 수집
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = self.model(X_batch.to(self.device))
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.numpy())
                all_outputs.append(outputs.cpu())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        # --- Top-K Accuracy ---
        # Top-1: 1등 예측만 정답이면 맞음 (= 기존 val_accuracy)
        # Top-2: 상위 2개 중 정답이 있으면 맞음
        # Top-3: 상위 3개 중 정답이 있으면 맞음
        # MDP/DQN은 확률 분포 전체를 사용하므로 Top-K가 높을수록 정책 품질이 좋아짐
        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        all_labels_tensor = torch.LongTensor(all_labels)
        num_classes = all_outputs_tensor.shape[1]

        print(f"\n[Top-K Accuracy (Val set, {num_classes}클래스)]")
        for k in range(1, min(4, num_classes + 1)):
            top_k_preds = torch.topk(all_outputs_tensor, k=k, dim=1).indices
            top_k_acc = (top_k_preds == all_labels_tensor.unsqueeze(1)).any(dim=1).float().mean().item()
            print(f"  Top-{k} Accuracy: {top_k_acc:.4f} ({top_k_acc:.1%})")
            if wandb.run:
                wandb.run.summary[f"top_{k}_accuracy"] = top_k_acc

        cm     = confusion_matrix(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=self.target_classes,
            digits=3,
            zero_division=0,
        )
        print("\n[Confusion Matrix (Val set)]")
        header = "          " + "  ".join(f"{c[:8]:>8}" for c in self.target_classes)
        print(header)
        for i, row in enumerate(cm):
            row_str = f"{self.target_classes[i][:8]:<10}" + "  ".join(f"{v:>8}" for v in row)
            print(row_str)
        print("\n[Per-class Precision / Recall / F1 (Val set)]")
        print(report)

        # --- W&B summary 명시적 세팅 ---
        if wandb.run:
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_val_acc"]  = best_val_acc
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels.tolist(),
                    preds=all_preds.tolist(),
                    class_names=self.target_classes,
                )
            })

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

    def run_modeling_pipeline(self, epochs: int = 50, hidden_dims: List[int] = None,
                              upload_artifact: bool = True, use_lr_scheduler: bool = False,
                              use_class_weights: bool = False):
        """
        데이터 준비부터 모델 학습, W&B 로깅 및 아티팩트 업로드까지 일괄 수행
        hidden_dims가 None이면 train_model() 기본값([128, 64]) 사용
        """
        kwargs = {
            "epochs": epochs,
            "use_lr_scheduler": use_lr_scheduler,
            "use_class_weights": use_class_weights,
        }
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
