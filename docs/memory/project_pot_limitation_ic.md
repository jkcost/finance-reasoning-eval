---
name: POT 프롬프트의 IC 판정 한계 — CoT/reasoning trace로 전환 (4/15 결정)
description: IC-L2/L3/L4는 "어느 값을 왜 선택했는지" 추적 필수. POT는 코드만 내 추적 불가 → 후속 실험은 명시적 reasoning trace 프롬프트 사용. POT 한계 자체를 CIKM Short finding 중 하나로 포함.
type: project
source: docs/meetings/2026-04-15-team-sync.md
date: '2026-04-15'
---

## 핵심 발견 (4/15 회의)

POT(Program-of-Thought)는 EA(부재) 계열에는 적합하지만, **IC(정보 충돌) 계열에는 측정 도구로 부적합**:

- **IC-L2 (단위 불일치)**: 모델은 역산해서 맞추거나 틀리거나 — POT 코드만으로는 "왜 이 단위를 택했는가" 판단 불가
- **IC-L3 (권위 충돌)**: 원본값과 권위 출처 값이 공존 → 기대 응답은 "두 값이 충돌해서 판단 불가". 하지만 POT는 그냥 한 값 찍어 계산. 권위 marker에 attention 쏠리는지 측정 못함
- **IC-L4 (기간 합산 불일치)**: 분기합 ≠ 연간총계 인식하려면 중간 추론이 필요한데 POT는 숨김

**결정적 관찰 (Speaker 3, 우익)**: "원본 수치와 1.5배 수치가 공존하면 어느 게 ground truth인지 모델이 판단 불가. 그럼에도 거의 다 한 값을 찍음. 어느 것을 왜 선택했는지 설명이 나와야 판단 가능하다."

## 방향 전환

**Why:** 현재 실험에서 IC 변형의 판정이 대부분 "거부 없이 한 값 선택 → 오답"으로 수렴. 이는 모델의 실제 능력 공백이 아니라 **POT 프롬프트가 측정 대상을 가릴 수 있음**을 시사. 또한 RQ-C(Conflict Salience Regression)의 feature(authority_marker, token_distance 등)가 실제로 선택에 영향 줬는지도 reasoning trace 없이는 인과 주장 불가.

**How to apply:**
- **신규 실험**: IC 변형은 CoT 또는 "reason-first" 프롬프트 — "먼저 필요한 값 식별 → 식 쓰기 → 값 대입 → 답" 형식. 모델의 값 선택 이유를 명시적으로 뽑아냄.
- **기존 실험 재활용**: EA 계열은 POT 유지(변경 이득 낮음). IC 계열만 프롬프트 분기.
- **논문 framing**: "POT limitations on IC detection"을 CIKM Short의 한 finding으로 포함. Negative result지만 post-POT 연구(Chain-of-Verification, Tool-use 등)와 자연스럽게 연결.
- **RQ-C 연결**: Salience feature가 "값 선택 이유" 로그에 실제로 언급되는지 cross-reference하면 인과 주장이 강해짐.

## 관련 코드

- `evaluation/model_runner.py` 프롬프트 전략 (POT 기반)
- `experiments/run_batch_evaluation.py` — 전략 스위치 존재 (standard/metacognitive/self_verification/contradiction_aware)
- 신규 전략 필요: `ic_cot_trace` (reason-first CoT) — 다음 스프린트 추가
