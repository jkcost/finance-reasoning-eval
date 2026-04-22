# Reason Record Schema

**버전**: v1 (2026-04-21)
**위치**: `experiments/run_batch_evaluation.py` 출력 레코드의 `refusal_reason` 필드

## 수집 경로

`run_batch_evaluation.py:248-265`에서 두 경로로 수집:

1. **인라인 reason** (모델 직접 제공): 응답이 `"INSUFFICIENT_INFORMATION: <reason>"` 형식인 경우 `:` 뒤 텍스트
2. **Detector reason** (자동 분류): `RefusalDetector.detect()`가 응답을 분석해 만든 reason 문자열 (`evaluation/refusal_detector.py`)

둘 다 없으면 `None`.

## 평가 레코드 구조 (관련 필드만)

```json
{
  "question_id": "test-2009",
  "transformation_type": "EA-partial",
  "model": "gpt-4o-mini",
  "prompt_strategy": "metacognitive",
  "response_type": "refused",
  "is_correct": false,
  "refusal_reason": "shares_outstanding 필드가 [DATA MISSING]으로 표시되어 있어 시가총액 계산 불가",
  "raw_response": "...",
  "executed_code": null,
  "execution_error": null
}
```

## Coding 레코드 스키마 (신규, 수동 코딩용)

`experiments/results/metacognitive/annotations/reason_coding_<coder>.json`:

```json
{
  "coder": "reviewer_a",
  "completed_at": "2026-04-29T10:30:00+09:00",
  "records": [
    {
      "question_id": "test-2009",
      "transformation_type": "EA-partial",
      "model": "gpt-4o-mini",
      "prompt_strategy": "metacognitive",
      "refusal_reason": "shares_outstanding 필드가 [DATA MISSING]으로 표시되어 있어 시가총액 계산 불가",
      "category": "MISSING_REQUIRED_VALUE",
      "confidence": "high",
      "note": "직접 인용 + 역할까지 정확히 짚음"
    }
  ]
}
```

**필드 정의**:
- `category`: 카테고리 코드 (아래 참조). 필수.
- `confidence`: `"high" | "medium" | "low"` — 코더의 카테고리 할당 확신도
- `note`: 자유 텍스트. 애매 케이스 메모.

## 카테고리 시드 (v0 draft, 다음 미팅 확정 대상)

문헌 소스:
- Feng+2024 Abstention taxonomy (Don't Know vs Can't Know)
- Rajpurkar+2018 SQuAD 2.0 unanswerable question types
- 금융 도메인 + 우리 변환 유형(EA/SA/IC) 특화

| Code | 이름 | 정의 | 예상 빈도 변형 | 참조 |
|------|------|------|---------------|------|
| `MISSING_REQUIRED_VALUE` | 필수 수치 누락 | 계산에 필수인 특정 수치·필드가 명시적으로 비어있다고 인지 ("[DATA MISSING]", "not provided") | EA-partial 지배적 | Feng 2024 "explicit abstention" |
| `MISSING_SILENT` | 정보가 없다고 추정 (무표지) | 마커 없이 데이터가 없다고 판단. 암묵적 abstention | SA | Rajpurkar 2018 "impossible" |
| `CONFLICTING_VALUES` | 값 충돌 | 두 개 이상의 모순된 수치가 공존 | IC-L1/L2/L3 | 신규 (본 연구) |
| `UNIT_AMBIGUITY` | 단위/규모 불일치 | 단위(백만/십억) 또는 통화(yen/USD)가 모순 | IC-L2 | 신규 |
| `TEMPORAL_MISMATCH` | 기간·시점 불일치 | 분기/연간/시점이 맞지 않음 | IC-L4, TA | 신규 |
| `UNDERSPECIFIED` | 일반적 정보 부족 (카테고리화 불가) | 위 5개 어디에도 속하지 않는 일반적 거부 | fallback | Cheng 2024 "balanced IDK" |

**예상 분포 (180 샘플 기준, 낙관적 추정)**:
- MISSING_REQUIRED_VALUE: ~35% (EA-partial 60문제 × 거부율 ~66%)
- MISSING_SILENT: ~10%
- CONFLICTING_VALUES: ~20%
- UNIT_AMBIGUITY: ~10%
- TEMPORAL_MISMATCH: ~5%
- UNDERSPECIFIED: ~15%
- 모델이 reason 미제공(None): ~5%

## 개정 이력

| 버전 | 날짜 | 변경 |
|------|------|------|
| v1 | 2026-04-21 | 초안 — 카테고리 v0 (6개) |
