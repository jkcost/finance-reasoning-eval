---
name: 원본 오답 문제는 분석에서 제외 가능 (4/15 결정)
description: Phase A 원본 컨텍스트에서 이미 틀린 문제는 Phase B 분석에서 제외해도 무방. "원본 맞춤 → 변환 후 어려워짐" 경로에 집중.
type: feedback
source: docs/meetings/2026-04-15-team-sync.md
date: '2026-04-15'
---

## 규칙

Phase A(원본 컨텍스트)에서 **오답을 낸 문제는 메인 분석 경로에서 제외**해도 된다. 보고는 하되, 핵심 결론은 "원본 맞춤 → 변환 후 거부/오답" 집합에 집중.

**Why:** 원본 컨텍스트에서도 풀지 못하는 문제는 metacognitive 측정의 혼란 요소(noise):
- 변환 후 거부했어도 "정보 때문에" vs "원래 못 풀어서"가 섞임
- Refusal Precision/Recall 계산 시 false positive 유발
- 최동희 교수 명시적 발언(회의 64:17): "원본 컨텍스트에서 오답 나온 애들은 좀 날리고 해도 일단 전 될 것 같아요. 원본 맞췄는데 바꿔서 어려워진 애들 위주로 하는 것도 우리 선택."

**How to apply:**
- 결과 표에 두 트랙 명시: `solvable_original ∩ refused_transformed` (메인) vs `unsolvable_original` (noise track, 부록)
- MC Score 계산 시 solvable 모집단은 **원본에서 맞춘 문제**로 한정
- Phase A 원본 오답률이 높은 모델(claude-haiku-4 20%)은 해석 시 이 제약 명시
- 예외: **1million 암기 사례**(test-2001 계열)는 원본 정답이어도 RQ-B에서 별도 하이라이트 (회상 증거)
