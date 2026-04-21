---
name: RQ-B POT Faithfulness 측정 도구 (Value Provenance Classifier + Memorization Score)
description: POT 응답이 변환 context 값을 썼는지 원본 회상인지 자동 분류. CIKM Short 포함됨.
type: project
originSessionId: fb7b68ad-0aa6-4dd2-b270-e68c7ee4409a
---
## 왜 필요한가

POT(Program-of-Thought) 형식 LLM이 **context에 없는 원본 숫자로 답을 생성하는 현상** 발견 (test-2001 등). 이건 metacognitive evaluation의 타당성을 위협함 — "IC 탐지 실패"가 실제 능력 gap인지, 단순 원본 기억에 의존한 건지 구분 불가.

## 4/15 회의 재현 증거 — 모델별 편차

test-2009(EA-partial) 원금 1M 제거 → 모델이 스스로 1million 회상해 정답 도출. 단위(yen → dollar) 치환 ablation 결과:

| 모델 | 단위 바꿔도 1million 고수 | 해석 |
|------|---------------------------|------|
| Gemini-2.5-flash | **강함** — 1million 그대로 | 학습 데이터 편향 강함 |
| GPT-4o-mini | 약함 — 10K/100K 등으로 따라감 | 편향 상대적으로 약함 |
| Claude-haiku-4 | 혼재 — 상황 따라 | 중간 |

**해석 (회의 합의)**: 실험 설계 문제가 아니라 **학습 데이터에서 원금=1million 표현 빈도가 압도적**. POT가 중간 추론을 숨기므로 "회상인지 추론인지" 겉보기로 구분 불가 → Value Provenance Classifier 필요성 재확인.

## 구성 요소

**Value Provenance Classifier** — 응답의 각 숫자를 4가지로 자동 분류:
- `from_context`: 변환된 context에서 그대로 왔음
- `from_removed_data`: 제거/변환된 원본값을 회상 (memorization 신호)
- `fabricated`: 환각 (아무 출처 없음)
- `common_constant`: 보편상수 (risk-free rate, tax rate 등)
- `derived`: 위 값들의 계산 결과

**Memorization Score** — 모델별 응답에서 `from_removed_data` 비율. 높을수록 benchmark 측정이 오염됨.

## 기존 자산

- `evaluation/reasoning_trace_analyzer.py`에 NumberExtractor, ValueProvenanceClassifier 초안 존재
- `evaluation/reverse_calc_detector.py`에 EA-full 역산 탐지 로직
- Case 분류 (Case 1 거부=유효 / Case 2 오답=유효 / Case 3 정답=암기/추론 분석) 파이프라인 존재

## CIKM Short 실행 계획

- W3 (4/29~5/5): Value Provenance Classifier fine-tuning + balanced 모델 결과에 적용
- W4 중: Memorization Score 표 계산 + Analysis 섹션 writing

## 연결 레퍼런스

- **"Memorization in LLM-Based Program Repair"** FSE 2025 자매논문 *"Demystifying Memorization..."* — GPT-3.5 78.83%/CodeLlama 87.42% bug 원본 일치. POT 스타일 code-generation memorization 직접 증거.
- **Lanham+2023 CoT Faithfulness** (arXiv:2307.13702, Anthropic) — CoT faithfulness 원조 개념
