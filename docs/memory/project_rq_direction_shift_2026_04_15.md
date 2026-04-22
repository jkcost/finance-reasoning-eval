---
name: 연구 앵글 확장 — 판정 이유 카테고리화 + Hard Example 생성 (4/15 결정)
description: Binary classification만으로는 재미 부족. 판정 이유 클러스터링(메타인지 Part 2) + IC-L1/L2 정답 재계산 hard example 두 앵글 추가.
type: project
source: docs/meetings/2026-04-15-team-sync.md
date: '2026-04-15'
---

## 결정

CIKM Short의 핵심 finding을 **3축**으로 확장:
1. (기존) **Binary classification**: solvable/unsolvable 탐지 — 기존 RQ-A
2. (신규) **A1. 판정 이유 카테고리화**: 모델이 거부 시 남긴 reason을 4~6개 카테고리로 클러스터링 → "모델이 *왜* 못 푸는지 맞추는지" 평가 (메타인지 Part 2). 단순 모름 여부(Part 1)에서 한 단계 깊어짐.
3. (신규) **A2. Hard example accuracy**: IC-L1(10x)·L2(단위·1.5x) 등 변형 후에도 정답 재계산 가능한 유형은 Python solution 값만 치환 → 사람 검수 → 새 ground truth로 "어렵게 만든 문제의 정답 맞추기" 실험. HLE(Humanity's Last Exam) 스타일. 전체 규모의 ~1/10 수준이면 충분.

**Why:** 최동희 교수 피드백 — "지금 그 perturbation만 가지고는 재미가 없어 보인다. 결국 binary classification dataset에 그친다. 휴리스틱으로 나눈 6개 중 하나 맞추는 게 의미 있지 않다. 한 발자국만 더 나가면 훨씬 이쁘게 될 것 같다." 또한 모건 교수 — 금융 literacy 측정은 "왜 못 푸는지 설명하는 능력"에 있음.

**How to apply:**
- **A1 파이프라인**: 기존 `INSUFFICIENT_INFORMATION: <reason>` 프롬프트(커밋 12902ed)로 수집된 reason 텍스트를 대상으로 (a) 수동 코딩 10문제 × 6변형 × 3~N모델 = ~180 reason 샘플 → 4~6 카테고리 도출 → (b) 모델이 카테고리 맞추는지 정량 평가 + 삐져나온 것은 generative NLI 매칭. 다음 주간 회의에서 카테고리 확정.
- **A2 파이프라인**: IC-L1·L2만 대상(값 하나로 결정 가능). (a) 원본 Python solution에서 변형된 값 치환 → 새 answer 계산, (b) 사람 1명 검수 → verified hard example, (c) POT/CoT로 각 모델에 풀게 하고 accuracy 측정. 전체 규모의 ~1/10이면 논문용으론 충분.
- **결과 배치 계획**: CIKM Short Results 섹션 = {RQ-A 비대칭성, A1 reason cluster, A2 hard example} 3표 + L1-L4 gradient figure.
- **8월 AAAI 확장 방향**: A1 카테고리 중 "실패 유형"을 multi-agent / tool-calling으로 mitigate하는 연구로 자연스럽게 이어짐.

## 기각/보류된 대안

- **NLI-only generative 매칭**: Speaker 1이 "generative로 하면 스코어링 어려움" → 카테고리 classification 먼저. 삐져나온 것만 generative.
- **카테고리 수 세분화(6변형 맞추기)**: "휴리스틱으로 바꾼 거니까 1/2/3/4번 맞추는 것 자체는 의미 없음" (Speaker 1). 대신 reason 카테고리로 가자.
- **모두 어노테이션 후 인사이트 도출**: Speaker 2 원안이었으나 기각 — 10문제 집중해서 먼저 아이디어 픽스 후 확장.
