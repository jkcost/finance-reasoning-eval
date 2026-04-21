---
name: 판정 이유(refusal_reason) 카테고리 v0 draft — 다음 회의 확정 대상
description: 문헌(Feng 2024, Rajpurkar 2018, Cheng 2024) + 금융 변환 유형에 기반한 reason 카테고리 6개 초안. 수동 코딩용 시드.
type: project
source: docs/meetings/2026-04-15-team-sync.md
date: '2026-04-21'
---

## 목적

4/15 회의 결정 [A1] 판정 이유 카테고리화 파이프라인의 시드 카테고리. 다음 주 회의에서 수집된 ~180 reason 샘플을 보고 확정.

## 카테고리 v0 (6개)

| Code | 정의 | 커버 변형 | 문헌 근거 |
|------|------|-----------|-----------|
| `MISSING_REQUIRED_VALUE` | 계산 필수 수치가 **명시적으로** 비었음을 인지 | EA-partial | Feng 2024 explicit abstention |
| `MISSING_SILENT` | 데이터 부재를 **암묵적으로** 추론 (마커 없이) | SA | Rajpurkar 2018 impossible QA |
| `CONFLICTING_VALUES` | 두 개 이상의 모순 수치 공존 | IC-L1/L2/L3 | 신규 — 본 연구 |
| `UNIT_AMBIGUITY` | 단위/통화/규모 불일치 | IC-L2 | 신규 |
| `TEMPORAL_MISMATCH` | 기간·시점 불일치 (분기 합 ≠ 연간) | IC-L4, TA | 신규 |
| `UNDERSPECIFIED` | 일반적 정보 부족 (fallback) | 모든 변형 | Cheng 2024 balanced IDK |

## 설계 원칙

1. **Orthogonal**: 각 reason은 정확히 하나의 primary category에 속해야 함. 중첩 시 가장 구체적인 것.
2. **Asymmetry 보존**: MISSING_*는 "부재 탐지", CONFLICTING_*/UNIT_*/TEMPORAL_*는 "충돌 탐지". RQ-A의 핵심 비대칭성이 카테고리 수준에서도 드러나야 함.
3. **Fallback 존재**: `UNDERSPECIFIED`로 애매 케이스 흡수 → 뒤에 generative 분석.

## 수동 코딩 시 판정 팁

- **인용 여부**: reason이 구체적 필드/값을 직접 인용하면 `MISSING_REQUIRED_VALUE` 가능성 높음
- **"not provided" 류 표현**: `MISSING_SILENT` 의심 (모델이 스스로 추론)
- **"however"/"but"/"contradicts"**: `CONFLICTING_VALUES` 신호
- **"million vs billion"/"yen"/"dollar"**: `UNIT_AMBIGUITY`
- **"Q1"/"annual"/"2019"**: `TEMPORAL_MISMATCH`
- 위 모두 약하면: `UNDERSPECIFIED`

## 확정 절차 (다음 회의)

1. 수집된 180 reason 전수를 2인이 v0 카테고리로 코딩 → Cohen's Kappa 계산
2. **Kappa ≥0.6**: v0 확정, 남은 샘플 single coder로 진행
3. **Kappa 0.4~0.6**: 애매 케이스 재검토 → 카테고리 정의 refine → 재코딩
4. **Kappa < 0.4**: 카테고리 재설계 (통합 or 분할)

## 코딩 프로토콜 (v1)

### 샘플링 전략

| 옵션 | 크기 | 장단점 | 채택 |
|------|------|--------|------|
| A. 전수 | ~180 (10문제×6변형×3모델) | 완전 커버리지. 시간 부담 (2인×180 ≈ 6~8h) | **채택** (pilot 단계라 규모 작음) |
| B. Stratified 80 | 변형 유형별 균등 (13/유형) | 시간 절감. 저빈도 유형 샘플 부족 위험 | 180으로 Kappa 낮으면 fallback |

**근거**: 10문제 pilot 단계라 전수해도 코더당 3~4시간. 확장 시(예: 50문제 × 6변형 × 4전략 × 6모델 = 7,200) stratified로 전환.

### 코더 배정

- **Primary**: 진규(Speaker 2) + 우익(Speaker 3) — 2인 독립 코딩
- **독립성 보장**: 서로의 코딩 결과 공유 금지 (완료 후 merge에서 최초 대조)
- **URL 분배**: `?coder=jkcost&start=0&end=180` / `?coder=wooik&start=0&end=180`
  (현 pilot은 전수라 start/end 동일, 향후 확장 시 분할)

### Kappa 기준

- **≥0.80**: 거의 완벽 (Landis & Koch 1977)
- **0.60~0.79**: 실질적 일치 → **v0 확정 기준선**
- **0.40~0.59**: 보통 → 정의 refine + 재코딩 1회
- **<0.40**: 부적절 → 회의에서 카테고리 재설계

### 불일치 해결

1. 불일치 reason만 별도 목록화 (자동 diff)
2. 두 코더 합의 시도 (15분 내 결론). 합의 시: 채택 + note에 근거 명시
3. 미합의: 제3자(교수) 중재 또는 `UNCATEGORIZABLE` 승격 → 다음 회의 안건

### 파일명 컨벤션

```
experiments/results/metacognitive/annotations/
├── reason_coding_jkcost_0_180.json
├── reason_coding_wooik_0_180.json
└── reason_coding_merged_0_180.json   # merge 후 자동 생성 (Kappa 포함)
```

### 다음 단계 — Merge/Kappa 스크립트

`experiments/merge_reason_codings.py` (주말 prep에서 생성 대상):
- 두 JSON 로드 → per-record 비교 → Cohen's Kappa 계산
- 불일치 목록 출력 (HTML report)
- merged JSON 생성 (합의 규칙: 확신도 높은 쪽 우선, 같으면 primary)

## 제외/비채택 고려 카테고리 (논의 이력)

- ~~`INSUFFICIENT_CONTEXT_LENGTH`~~: 짧은 문제엔 비발생. 일반 UNDERSPECIFIED로 흡수.
- ~~`AUTHORITY_MISMATCH`~~: IC-L3 전용 같지만, 실제로는 CONFLICTING_VALUES의 서브타입. 분리 시 sparsity 우려. **Kappa 낮으면 분리 재검토**.
- ~~`REASONING_CHAIN_BROKEN`~~: 애매하고 측정 어려움 → 제외.

## C2 Sanity Check 결과 (2026-04-21)

**데이터**: `eval_metacognitive.json` 기존 결과에서 R4 solvable 필터 적용 후 287 reason 추출 (gpt-4o-mini × metacognitive × 7 변형).

| 카테고리 | 검증 결과 | 휴리스틱 샘플 수 | 비고 |
|---------|---------|-----------------|------|
| MISSING_REQUIRED_VALUE | ✅ 매우 적합 | 183 (63.8%) | EA-partial/full, SA 주류 패턴 |
| MISSING_SILENT | ⚠️ reason 텍스트로 구분 불가 | — | EA-partial과 reason 텍스트 거의 동일. v1 통합 검토 |
| CONFLICTING_VALUES | ❌ 샘플 부족 | 1 | IC 거부율 자체가 낮음 (회의 finding 재확인) |
| UNIT_AMBIGUITY | ❌ 샘플 부족 | 1 | IC-L2 reason 필요 |
| TEMPORAL_MISMATCH | ❌ 샘플 부족 | IC-L4 1건만 유효 | 휴리스틱 false positive 70건(EA의 "quarterly"/"annual" 단어 혼동) |
| UNDERSPECIFIED | ✅ fallback 적합 | 31 (10.8%) | "The question does not provide..." variant |

**주요 결론**:
1. EA/SA 축(reason 292개 중 80%)은 v0로 즉시 본 코딩 진행 가능
2. IC 축은 샘플 절대 부족 → `phase_B_contradiction_aware_*.json` 추가 추출 또는 IC 전용 A1 재실행 필요
3. **v1 수정 제안**: `MISSING_SILENT`와 `MISSING_REQUIRED_VALUE`를 `MISSING_DATA`로 통합, `transformation_type`을 metadata로 병용하여 분석 시 구분. **5-카테고리 체계**로 단순화 후보.
4. 구 프롬프트(4/13) reason은 EA/SA 구분이 불가능 → **신규 프롬프트(12902ed) 재실행은 MISSING_* 서브타입을 구분하려는 경우에만 의미 있음**. 현재 목적이 "비대칭성 측정"이면 통합 카테고리도 무방.

## 개정 이력

| 버전 | 날짜 | 변경 |
|------|------|------|
| v0 | 2026-04-21 | 초안 — 카테고리 6개 |
| v0.1 | 2026-04-21 | C2 sanity check — EA/SA v0 적합, IC 샘플 부족 발견, MISSING 통합 제안 |

## 연결

- 스키마: [docs/schemas/reason_record.md](../schemas/reason_record.md)
- 수집 프롬프트: `evaluation/refusal_detector.py` + `run_batch_evaluation.py`
- 워크북: `experiments/reason_coding_workbook.html` (prep 중)
