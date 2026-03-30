# TODOS

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
