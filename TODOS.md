# TODOS

## P0: 판정 이유 카테고리화 파이프라인 (4/15 회의 A1)
**What:** 10문제 × 6변형 × 3모델 = ~180 reason 샘플 수집 → 4~6 카테고리 수동 도출 → 모델이 카테고리 맞추는지 정량 평가. 삐져나온 건 generative NLI 매칭.
**Why:** 최동희 교수 피드백 — binary classification만으로 재미 부족. 메타인지 Part 2("어떻게 모르는지")가 논문 엣지.
**Context:** `INSUFFICIENT_INFORMATION: <reason>` 프롬프트 이미 커밋됨(12902ed). 다음 주간 회의에서 카테고리 확정.
**Effort:** M (1주)
**Priority:** P0
**Depends on:** 판정 이유 프롬프트 실험 1회 실행 (진규)

## P0: Hard Example 정답 재계산 (4/15 회의 A2)
**What:** IC-L1·L2 변형 문제 중 정답 재계산 가능한 유형 → Python solution 값 치환 → 사람 1명 검수 → verified hard example → POT/CoT로 accuracy 측정.
**Why:** HLE 스타일 앵글. A1과 독립적으로 정량 table 제공. 전체 규모의 ~1/10이면 충분.
**Effort:** M (1~2주)
**Priority:** P0
**Depends on:** A1 카테고리 확정, Python solution validator

## P0: IC용 CoT/reason-first 프롬프트 전략 추가
**What:** `evaluation/` 프롬프트 전략에 `ic_cot_trace` 추가 — "필요한 값 식별 → 식 → 값 대입 → 답" 형식. IC-L2/L3/L4만 분기.
**Why:** POT로는 "어느 값을 왜 선택했는지" 추적 불가 → RQ-C의 feature 인과 주장 약화.
**Effort:** S (2~3일)
**Priority:** P0
**Depends on:** 없음

## P1: Balanced 모델 실험 실제 실행
**What:** `run_balanced_experiment.sh --execute --strategy all` — gpt-4o, claude-sonnet-4, gemini-2.5-pro × 4~6 전략 × 238문제.
**Why:** CIKM W2(4/22~4/28) 계획. RQ-A 비대칭성 + L1-L4 gradient 통계적 유의성 확보.
**Effort:** M (실행 시간 + 비용 ~$5~8)
**Priority:** P1
**Depends on:** Phase 0 변환 완료 확인, API 키 3종

## DEFERRED: 로컬 모델(Llama 등) 결과 통합
**What:** `run_local_model_eval.py`로 Llama 3.1/qwen2.5 실험 → HTML에 API 모델과 나란히 표시.
**Status:** CIKM Short 본 실험 안정화 후 재개. GPU 인프라 의존성으로 별도 트랙 분리.
**See:** [docs/backlog/local_model_experiment.md](docs/backlog/local_model_experiment.md)

## P2: Cross-Domain IC Validation (Legal/Medical)
**What:** IC 탐지 실패가 금융만의 문제인지 LLM 근본 한계인지 확인. SQuAD 2.0, 의료 QA 등에 IC 변환 적용하여 재현성 테스트.
**Why:** 논문의 일반화 주장 강화. 리뷰어 "금융에만 해당되는 거 아닌가?" 질문 대비.
**Pros:** 후속 논문의 핵심 contribution. ACL main 수준 기여.
**Cons:** 도메인 전문성 필요. 데이터셋 확보 + 변환 품질 검증 effort 큼.
**Context:** CEO review에서 DEFERRED. 본 IC Finding Paper 출판 후 진행. IC L1-L5 taxonomy를 그대로 적용 가능한지가 핵심 검증 포인트.
**Effort:** XL (human) → L (CC+gstack)
**Priority:** P2
**Depends on:** IC Finding Paper 출판, IC L1-L5 taxonomy 확정

## P2: Attention Pattern Analysis (IC Mechanism Explanation)
**What:** Open model(Llama 70B)의 attention weight를 추출해서 IC 문제에서 충돌하는 두 값 사이의 attention 분포를 분석. "모델이 권위 있는 값에 attention이 쏠린다"는 메커니즘 설명.
**Why:** IC 실패의 원인을 black-box에서 꺼내면 논문의 설명력이 크게 올라감. "관찰 + 설명 + 해결" 3박자 완성.
**Pros:** 논문 수준을 findings paper에서 analysis paper로 격상. Oral presentation 가능성.
**Cons:** A100+ GPU 필요 (FP16: 140GB, 4-bit: 35GB). Attention 해석 가능성 논란 있음 (Jain & Wallace 2019).
**Context:** CEO review에서 ACCEPTED, eng review에서 GPU 제약으로 DEFERRED. 4-bit quantized → full precision 2단계 접근 권장.
**Effort:** L (human: 2-3주) → M (CC: GPU 인프라 확보 시)
**Priority:** P2
**Depends on:** GPU 인프라 (A100+), IC balanced model 실험 완료
