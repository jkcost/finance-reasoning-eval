# Phase A 원본 컨텍스트 오답 분석 보고서

**생성일**: 2026-04-16
**데이터 소스**: hard.json 238문제 중 pilot 5문제 x economic 3모델
**분석 기준**: CLAUDE.md 문서화 결과 + error_taxonomy 13개 분류 체계

---

## 1. Phase A Baseline 요약

| 모델 | 정답 | 오답 | 정답률 | 비고 |
|------|------|------|--------|------|
| gemini-2.5-flash | 3 | 2 | 60% | 최고 성능 |
| gpt-4o-mini | 2 | 3 | 40% | 중간 |
| claude-haiku-4 | 1 | 4 | 20% | 최저 성능 |
| **전체** | **6** | **9** | **40%** | 15건 중 |

## 2. 오답 패턴 분석

### 2.1 모델별 특성

**gemini-2.5-flash (오답 2건)**
- hard 난이도에서도 60% 정답률로 비교적 높은 baseline 유지
- 오답 2건은 복잡한 다단계 계산 문제에서 발생한 것으로 추정
- POT(Program-of-Thought) 방식의 코드 생성 정확도가 높음

**gpt-4o-mini (오답 3건)**
- 중간 수준 성능 (40%)
- 금융 공식 적용 오류(FORMULA_ERROR)와 수치 추출 오류(EXTRACTION_ERROR) 가능성
- 비용 효율적이나 hard 문제에서는 한계 노출

**claude-haiku-4 (오답 4건)**
- 가장 낮은 baseline (20%)
- 다수의 오답이 POT 코드 실행 오류(EXECUTION_ERROR)나 파싱 오류(PARSING_ERROR)로 추정
- 비용이 가장 높으면서 ($0.015) 성능이 가장 낮은 비효율 구간

### 2.2 추정 오류 유형 분포 (13개 분류 체계 기반)

hard.json 문제의 특성상 주로 발생하는 오류 유형:

| 오류 유형 | 예상 빈도 | 설명 |
|-----------|-----------|------|
| FORMULA_ERROR | 높음 | 복잡한 금융 공식 (ROE, WACC 등) 적용 실패 |
| EXTRACTION_ERROR | 높음 | 맥락에서 올바른 수치를 추출하지 못함 |
| CALCULATION_ERROR | 중간 | 단순 산술 실수 |
| EXECUTION_ERROR | 중간 | POT 코드 실행 실패 (특히 claude-haiku-4) |
| MISUNDERSTANDING | 낮음 | 문제 자체를 잘못 이해 |
| UNIT_ERROR | 낮음 | 단위 혼동 (예: million vs billion) |

### 2.3 난이도 영향

- **hard 문제의 특징**: 다단계 추론, 복수 데이터 소스 통합, 복잡한 금융 개념
- **경제 모델(economic tier)의 한계**: 전체 40% 정답률은 hard 문제가 economic 모델에게 실질적으로 어렵다는 것을 시사
- **Phase B와의 관계**: baseline 정답률이 낮으면 Phase B에서 MC Score의 Refusal Precision(RP) 계산 시 false positive 보정이 중요

## 3. 핵심 인사이트

### 인사이트 1: 모델 크기-성능 비선형성
- gemini-2.5-flash가 claude-haiku-4보다 3배 높은 정답률
- 모델 크기만으로 금융 추론 성능을 예측하기 어려움
- **시사점**: balanced 모델(gpt-4o, claude-sonnet-4, gemini-2.5-pro)에서 이 격차가 좁혀지는지 확인 필요

### 인사이트 2: Phase A 오답 = Phase B의 잠재적 False Positive
- Phase A에서 이미 틀린 문제는 Phase B(변환된 문제)에서도 틀릴 가능성이 높음
- 이 경우 "변환 때문에 거부했다"가 아니라 "원래 못 풀었다"일 수 있음
- **시사점**: Cross-phase 분석(Phase D)에서 Phase A 오답 문제를 별도 트랙으로 분류해야 함

### 인사이트 3: 비용 대비 효율
- gemini-2.5-flash: $0.003으로 60% → 가장 효율적
- claude-haiku-4: $0.015으로 20% → 5배 비용에 1/3 성능
- **시사점**: balanced budget에서의 비용-성능 곡선이 논문의 RQ4(비용분석) 데이터가 될 수 있음

### 인사이트 4: 정보 부재 vs 충돌 탐지의 비대칭성 기반
- Phase A baseline이 낮은 모델(claude-haiku-4)이 오히려 Phase B에서 높은 거부율(66.7%)을 보임
- **가설**: "잘 못 풀지만, 정보 부족은 잘 탐지한다" = 메타인지와 추론 능력의 분리
- 이는 논문의 핵심 발견으로, 더 많은 문제로 검증이 필요

## 4. 후속 작업 권장사항

1. **balanced 모델 실험 실행**: gpt-4o, claude-sonnet-4, gemini-2.5-pro로 동일한 문제셋 평가
2. **전수 Phase A 실행**: pilot 5문제 → 238문제 전체로 확장하여 통계적 유의성 확보
3. **오답 패턴 자동 분류**: `evaluation/error_analysis/error_classifier.py`를 활용한 규칙 기반 분류 실행
4. **LLM 기반 상세 분석**: `evaluation/error_analysis/llm_error_analyzer.py`로 개별 오답에 대한 한글 분석 생성 (balanced 실험 결과 확보 후)

## 5. 분석 스크립트

전수 결과 확보 후 아래 명령으로 자동 분석 가능:

```bash
# Phase A 결과 파일이 experiments/results/metacognitive/ 에 있을 때
python experiments/run_metacognitive_experiment.py --phase D \
    --results-dir experiments/results/metacognitive/

# 또는 모델 비교 보고서 생성
python experiments/run_model_comparison.py --budget balanced --n 20
```

---

*본 보고서는 pilot 규모(5문제 x 3모델) 기반 분석입니다. 통계적 결론을 위해서는 전수 실험(238문제) 결과가 필요합니다.*
