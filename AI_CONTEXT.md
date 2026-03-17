# AI_CONTEXT.md
## 프로젝트 개요
**SmartPitch: MLB 투구 전략(볼배합) 최적화 파이프라인**
- 마르코프 결정 과정(MDP)과 강화학습(DQN)을 활용해, 투수에게 최적의 구종과 투구 코스(Zone)를 추천하는 시스템입니다.
- 특정 볼카운트, 아웃카운트, 주자 상황에서 **RE24(기대 실점)**를 최소화하는 것을 목표로 합니다.

## 아키텍처 및 환경
- 패키지 관리: `uv` 기반의 초고속 파이썬 환경 (Python 3.12+)
- 딥러닝/강화학습: `PyTorch`, `Stable-Baselines3` (CUDA 가속 적용)
- 데이터 소스: `pybaseball` (MLB Statcast Pitch-by-Pitch Data)
- MLOps: `WandB(Weights & Biases)`를 통한 실험 및 Artifact 버저닝

## 현재 작업 진행도
1. 투수 구종 군집화 (완료): 투수의 과거 투구 데이터를 UMAP + K-Means로 줄여 동적으로 구종을 식별 (`src/clustering.py`).
2. 타자 타격 어프로치 군집화 (완료): 
   - 좌/우타 타석 분리 후 8가지 타격 특성 지표(Whiff%, Zone Contact, O-Swing 등) 추출 (파울 제외 타구 속도/발사각 적용).
   - 2023시즌(500구 이상 타자) UMAP + K-Means (K=8) 적용.
   - 타자 군집화(K=8) CSV 추출 완료 및 `pitch_env.py` 상태 공간 확장 완료(State 2304개).
3. 전이 확률 모델 (진행 완료): `batter_cluster` 변수를 입력 특성에 추가 및 학습/MDP 호환성 전체 파이프라인 개조.

## 다음 목표 (Next Step)
- 타자 군집 정보가 포함된 데이터로 전이 확률 모델(MLP) 재학습 및 강화학습(DQN) 에이전트 본격 학습 준비.
