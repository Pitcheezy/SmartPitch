# ⚾ SmartPitch: MLB 투구 전략 최적화 파이프라인

딥러닝(PyTorch) 기반 전이 확률 예측 모델과 마르코프 결정 과정(MDP)을 활용하여, 특정 볼카운트 및 주자 상황에서 기대 실점(RE24)을 최소화하는 최적의 투구 전략을 도출하는 파이프라인입니다.

## 파이프라인 아키텍처
1. **Data Loader:** `pybaseball` 연동 및 스탯캐스트 데이터 전처리
2. **Clustering:** UMAP & K-Means 기반 동적 구종 식별 체계
3. **Transition Model:** PyTorch 다층 퍼셉트론(MLP) 기반 타격 결과 예측
4. **MDP Solver:** 벨만 방정식 역순 가치 반복(Value Iteration)을 통한 볼배합 최적화
5. **MLOps:** W&B(Weights & Biases)를 통한 데이터/모델 Artifact 관리 및 대시보드 로깅

## 팀원 실행 가이드 (Quick Start)

본 프로젝트는 초고속 파이썬 패키지 관리자 `uv`를 사용합니다.

```bash
# 1. 레포지토리 클론
git clone [레포지토리 주소]
cd SmartPitch

# 2. 완벽하게 동일한 환경(Dependencies) 1초 만에 세팅
uv sync

# 3. 전체 파이프라인 실행
uv run main.py