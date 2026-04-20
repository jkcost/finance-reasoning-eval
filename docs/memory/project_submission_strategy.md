---
name: CIKM 2026 Short 제출 전략
description: Primary target CIKM Short (6/6) + Follow-up AAAI 2027 (8/1). RQ-A/B/C를 CIKM에 포함, RQ-D/E는 AAAI 확장.
type: project
originSessionId: fb7b68ad-0aa6-4dd2-b270-e68c7ee4409a
---
## 결정 (2026-04-15)

**Primary**: CIKM 2026 Short Research (4p, 마감 2026-06-06 AoE, Rome 11/7~11/11)
**Follow-up**: AAAI 2027 (7p, 마감 2026-08-01 / abstract 7/25, Montréal 2027-02)

**Why:** Full Research (5/23, 38일)은 single-dataset 약점 + 10p 글쓰기 부담. Resource Papers (6/6, 52일)는 3개 dataset 이식 비현실. Short은 현재 work 그대로 4p 압축 가능 + 52일 여유. 이후 AAAI로 cross-dataset 확장.

**How to apply:**
- 6/6까지 Short 제출에 집중. 5/22 체크포인트에서 진행률 평가.
- CIKM Short 제출 직후 AAAI W1 시작 (6/9~).
- CIKM 확정 트랙 없음 (Findings 트랙 없음 공식 확인).

## CIKM Short RQ 범위 (확장판 — 2026-04-15 결정)

- **RQ-A**: FinanceReasoning에서 absence vs conflict 비대칭 + IC L1-L4 gradient (핵심 finding)
- **RQ-B**: POT Faithfulness 정량 — Value Provenance Classifier + Memorization Score
- **RQ-C**: Conflict Salience Regression — token_distance / authority_marker / magnitude_ratio / same_paragraph로 detection 예측

**Why:** 기존에는 RQ-A만 CIKM, B/C는 AAAI 확장이었으나 사용자 결정으로 B/C까지 CIKM에 포함. POT faithfulness 발견이 measurement validity 위협이라 CIKM 단계에서 제시하는 게 논문 credibility에 핵심.

## AAAI 2027 추가 축 (CIKM 제출 후 8주)

- **RQ-D**: Cross-dataset 일반화 (FinQA 최소 1개 추가)
- **RQ-E**: Human Baseline (금융 전문가 5~10명 IC-L3)
- Error Pattern Taxonomy 세분화 (무시/암묵채택/인지후진행)
- Reasoning 모델 (o1/o3, DeepSeek R1) 커버리지

## 경쟁 논문 포지셔닝 (2026-04-15 검증됨)

- **AbstentionBench** (arXiv:2506.09038, Meta FAIR) — 일반 도메인 abstention, **conflict 미커버**로 차별화
- **ConflictBank** (NeurIPS 2024, arXiv:2408.12076) — RAG 검색 충돌, **intra-context 수치 충돌 약함**으로 차별화
- **MAGIC** (Findings EMNLP 2025, 2025.findings-emnlp.466) — multi-hop gradient, **수치 ladder·financial 도메인 아님**으로 차별화
