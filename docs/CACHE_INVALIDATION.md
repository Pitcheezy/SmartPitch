# 캐시 무효화 가이드

SmartPitch는 여러 단계에서 캐시를 활용한다. 파라미터나 데이터가 변경되면 해당 캐시를 삭제해야 올바른 결과를 얻을 수 있다.

## 1. RE24 JSON 캐시 (`re24_loader.py`)

| 항목 | 설명 |
|------|------|
| 캐시 방식 | `lru_cache(maxsize=8)` — 프로세스 수명 동안 유지 |
| 무효화 시점 | `data/re24_{YYYY}.json` 파일 내용 수정 후 |
| 방법 | 프로세스 재시작, 또는 코드 내 `load.cache_clear()` 호출 |
| 영향 범위 | PitchEnv, MDPOptimizer의 보상 함수 |

## 2. MDP 정책 캐시 (`data/mdp_optimal_policy.pkl`)

| 항목 | 설명 |
|------|------|
| 캐시 방식 | pickle 파일 (디스크 영구 저장) |
| 무효화 시점 | 아래 중 하나라도 변경 시 |
| 방법 | `rm data/mdp_optimal_policy.pkl` 후 `evaluate_baselines.py` 재실행 |

**변경 시 삭제 필수**:
- RE24 매트릭스 (시즌 변경 또는 JSON 값 수정)
- `mdp_solver.py` 로직 (VI 반복 횟수, γ, 보상식, 전이 규칙)
- 범용 MLP 모델 가중치 (`best_transition_model_universal.pth`)
- `physical_feature_lookup.csv` (물리 피처 → MLP 입력 변경)
- Action space 변경 (유효 구종 필터, zone 목록)

```bash
# 정책 캐시 삭제 + 재계산
rm -f data/mdp_optimal_policy.pkl
uv run src/evaluate_baselines.py
```

## 3. pybaseball 캐시 (`.pybaseball/`)

| 항목 | 설명 |
|------|------|
| 캐시 방식 | 디스크 파일 (HTTP 응답 캐시) |
| 무효화 시점 | 시즌 진행 중 최신 데이터 필요 시 |
| 방법 | `rm -rf .pybaseball/` |

## 4. 변경 시나리오별 체크리스트

### RE24 시즌 변경 (예: 2024 → 2025)

1. `data/re24_2025.json` 생성 (기존 JSON 포맷 준수)
2. 호출부에서 `season=2025` 전달 (또는 `re24_loader.DEFAULT_SEASON` 변경)
3. `rm data/mdp_optimal_policy.pkl`
4. MDP 정책 재계산
5. DQN 재학습 또는 기존 모델로 평가 재실행

### MLP 모델 재학습 후

1. `rm data/mdp_optimal_policy.pkl`
2. MDP 정책 재계산
3. DQN 재학습 (MLP이 바뀌면 환경 전이가 달라지므로)

### 구종 필터 변경 후

1. `rm data/mdp_optimal_policy.pkl`
2. MDP 정책 재계산 (action space 변경됨)
3. 해당 군집 DQN 재학습
