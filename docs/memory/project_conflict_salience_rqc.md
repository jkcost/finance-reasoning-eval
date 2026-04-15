---
name: RQ-C Conflict Salience Regression (IC 탐지 예측 모델)
description: IC 탐지 성공/실패를 4개 feature로 예측하는 logistic regression. CIKM Short 포함됨.
type: project
originSessionId: fb7b68ad-0aa6-4dd2-b270-e68c7ee4409a
---
## 왜 필요한가

IC L1-L4 gradient만으로는 "왜 L3가 L2보다 쉬운가" 같은 메커니즘을 설명 못함. Descriptive benchmark에서 prescriptive framework로 격상하려면 **어떤 조건이 탐지 성공/실패를 결정하는지** 정량 feature로 규명 필요.

## Features (초안)

- `token_distance`: 충돌 두 값 사이 token 거리
- `same_paragraph`: 같은 문단 여부 (binary)
- `authority_marker_count`: "audited", "confirmed", "according to" 등 권위 marker 수
- `magnitude_ratio`: 두 값 차이 비율 (log scale)
- (추가 고려) `numeric_format_match`: 형식 일치 여부 (% vs 소수, comma 등)
- (추가 고려) `context_length`: 전체 context 길이

## 분석 방법

- Logistic regression: IC instance × 모델별 detection(refused=1, confident=0) 예측
- Feature coefficient + p-value + AUC 리포트
- 핵심 인싸이트 기대:
  - `authority_marker`가 dominant → **권위편향(authority bias)** + anchoring 메커니즘 주장 가능 (Kahneman-Tversky 연결)
  - `token_distance` dominant → **context-window 주의 메커니즘** 문제
  - `magnitude_ratio` dominant → **수치 감도 부족** (Novelty-blindness)

## 기존 자산

- `experiments/conflict_salience_scorer.py` 초기 구현 존재
- `tests/test_conflict_salience_scorer.py` — 8개 테스트

## CIKM Short 실행 계획

- W4 (5/6~5/12): Feature extractor 확장 + balanced 실험 결과에 fitting
- W5: Salience regression 결과표 + discussion 섹션 초안

## Mitigation과의 연결

- 낮은 탐지율 예측 feature를 역설계 → **의도적 mitigation prompt 설계**
- 예: authority_marker가 dominant면 "ignore authority claims, verify numerically" 프롬프트 추가
- 이 연결이 논문의 "실용적 impact"를 보여줌
