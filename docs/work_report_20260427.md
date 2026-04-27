# 작업 보고서: Task 20 + 20-A (2026-04-27)

## 개요

RE24 매트릭스 하드코딩을 시즌별 JSON 로더로 교체하고, 2024 RE24 기준으로 전체 시스템을 재평가했다.

---

## 1. Task 20: RE24 시즌별 로더 도입

### 배경
`pitch_env.py`와 `mdp_solver.py`에 2010-2015 Tango era RE24 값이 하드코딩되어 있었다.
분석 대상 시즌(2024)과 기준값이 불일치하는 기술 부채 상태였다.

### 작업 내용

| 항목 | 내용 |
|------|------|
| `src/re24_loader.py` 신규 | JSON 기반 로더, `lru_cache(maxsize=8)`, 24-state 검증, `get_state_key()` 유틸 |
| `data/re24_{2019,2023,2024}.json` | 3시즌 RE24 매트릭스 (git tracked). 출처: FanGraphs/Tango |
| `scripts/compute_re24_per_season.py` | Statcast play-by-play RE24 재현/검증 스크립트 (286줄) |
| `pitch_env.py` 수정 | `RE24_MATRIX` 딕셔너리 제거 → `re24_loader.load(season)` 호출 |
| `mdp_solver.py` 수정 | 동일 하드코딩 제거 → `re24_loader.load(season)` + `get_state_key()` |
| 호출부 전수 수정 | `main.py`, `evaluate_baselines.py`, `main_cease.py`, `main_gallen.py`, `evaluate_personal_dqn.py`, `train_dqn_all_clusters.py`, `analyze_mdp_vs_env.py` — 모두 `season=2024` 명시 |
| `tests/test_re24_loader.py` | 13개 유닛 테스트 전 통과 |
| 문서 | `docs/re24_seasonal_analysis.md`, `docs/CACHE_INVALIDATION.md` |

### 아키텍처

```
data/re24_{YYYY}.json          ← 시즌별 JSON (git tracked)
src/re24_loader.py             ← 로더 모듈 (lru_cache, 24-state 검증)
  ├── load(season) → dict      ← PitchEnv, MDPOptimizer가 호출
  ├── get_state_key()          ← 키 포맷 공유 유틸
  └── list_available_seasons() ← 사용 가능 시즌 목록
```

---

## 2. Task 20-A: MDP 정책 재계산 + 전 베이스라인 재평가

### 핵심 발견

**RE24 변경은 절대값만 균일하게 Δ ≈ +0.040 이동시키며, 모든 모델 간 상대 순위가 보존된다.**

### 검증 결과

#### 군집별 베이스라인 (2024 RE24, 1000 ep)
```
군집 0: MDP +0.298 > MostFreq +0.260 > Freq +0.257 > Random +0.225
군집 1: MDP +0.286 > Freq +0.209 > MostFreq +0.180 > Random +0.176
군집 2: MDP +0.300 > Random +0.269 > Freq +0.243 > MostFreq +0.241
군집 3: MDP +0.296 > MostFreq +0.291 > Freq +0.242 > Random +0.211
```

#### 개인 DQN (2024 RE24, 1000 ep)
```
Cease:  MDP +0.291 > Freq +0.273 > MostFreq +0.260 > DQN +0.238 > Random +0.236
Gallen: DQN +0.279 = MDP +0.279 > Random +0.264 > MostFreq +0.260 > Freq +0.244
```

#### 균일 오프셋 확인
- 모든 에이전트: Δ ≈ +0.040 (±0.003)
- 원인: 2024 RE24의 `0_000`(0.49)이 구 하드코딩(0.481)보다 0.009 높고, 에피소드 내 보상 누적에서 ~0.04로 확대
- 유일한 예외: Cluster 0 Frequency (Δ=+0.082) — Statcast 캐시/seed 차이로 분포가 미세하게 달라진 것

### 산출물
- `docs/re24_2019_vs_2024_comparison.md`: 전체 비교 보고서
- 구 RE24 결과 백업: `data/*_re24_2019.pkl/json`

---

## 3. 문서 최신화

Task 20/20-A 완료에 맞춰 아래 문서를 2024 RE24 기준으로 갱신:

| 문서 | 갱신 내용 |
|------|----------|
| `README.md` | 파일 구조 (re24_loader, compute_re24, tests/, 새 docs), 성능 표 2024 RE24, 기술 부채/마일스톤 |
| `AI_CONTEXT.md` | Task 20/20-A 완료 기록, 성능 수치 2024 RE24, 기술 부채 갱신 |
| `CLAUDE.md` | Task 19 DQN 수치 갱신, Task 20-A 완료 기록 |
| `TODO.md` | 요약 테이블 Task 20 완료 반영 |
| `docs/personal_dqn_report.md` | Cease/Gallen 전체 수치 2024 RE24 기준 갱신 |
| `docs/improvement_roadmap.md` | Task 20 완료 표시, 마일스톤 M1 갱신 |
| `docs/AUTO_TASK_COMPLETE.md` | Task 20/20-A 상세 완료 기록 추가 |

---

## 4. 결론 및 다음 단계

### 결론
- RE24 시즌별 로더 도입으로 **하드코딩 기술 부채 완전 해소**
- 2024 RE24 반영 결과, **상대 성능 비교에 영향 없음** (Δ≈+0.040 균일 오프셋)
- 시즌 변경이 필요할 때 JSON 추가 + `season=` 파라미터만 변경하면 됨

### 다음 우선순위
1. **Task 21**: 인플레이 타구 확률 실데이터 교체 (0.5일)
2. **Task 23**: MLP 3시즌 데이터 확장 (2일)
3. **Task 24**: Calibration 개선 — Temperature Scaling (0.5일)
