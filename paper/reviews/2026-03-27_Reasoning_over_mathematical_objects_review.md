# Peer Review: Reasoning over Mathematical Objects: On-Policy Reward Modeling and Test Time Aggregation

**논문 정보:**
- 저자: Pranjal Aggarwal, Marjan Ghazvininejad, Seungone Kim 외 (FAIR at Meta, UCLA, CMU)
- 페이지: 70페이지 (본문 ~45 + 부록 ~25)
- 리뷰 일자: 2026-03-27

---

## 1. 요약 (Summary Statement)

본 논문은 LLM의 수학적 객체(mathematical objects) 도출 능력을 평가하고 개선하기 위한 세 가지 주요 기여를 제시한다. **(1) Principia Suite**: 수학적 객체(방정식, 부등식, 구간, 집합, 행렬, 조각함수)를 답으로 요구하는 벤치마크(PrincipiaBench, 2,558문제)와 대규모 학습 데이터셋(Principia Collection, 248,748문제)을 구축했다. **(2) RLLM**: LLM 자체를 보상 모델로 활용하는 통합 포스트 트레이닝 프레임워크로, on-policy 학습된 LM-as-RM이 RLHF(스칼라 RM)와 RLVR(규칙 기반 검증기) 모두를 능가함을 보였다. **(3) ParaGator**: pass@k 최적화와 self-aggregation을 결합한 테스트 타임 컴퓨트 스케일링 방법으로, 병렬 생성의 다양성과 집약 품질을 동시에 개선한다.

### 전반적 평가: **Minor Revisions (소폭 수정 후 수락 권장)**

### 핵심 강점
- 수학적 객체 도출이라는 중요하고 간과된 평가 차원을 체계적으로 제시하며, MCQ 제거 시 10-20% 성능 하락이라는 강력한 동기 부여를 제공
- Principia Suite(벤치마크 + 학습 데이터 + 검증 벤치마크)의 3중 구성이 연구 생태계 기여 측면에서 매우 포괄적
- RLLM의 on-policy LM-as-RM이 easy/hard-to-verify 및 non-verifiable 작업에 걸쳐 일관된 개선을 보여주는 통합 프레임워크 설계

### 핵심 약점
- 세 가지 독립적 기여(Principia, RLLM, ParaGator)의 결합이 논문의 초점을 분산시키며, 각각이 개별 논문 수준의 분량
- RLLM의 핵심 실험이 Qwen3-1.7B에 집중되어 있어 규모 확장성 검증이 부족
- Principia Collection의 합성 데이터 품질에 대한 체계적 분석이 불충분

---

## 2. Major Comments

### M1. 논문 범위의 과도한 확장 — 세 논문이 한 논문에 (구조)

본 논문은 사실상 세 개의 독립적 연구를 하나로 합쳐놓았다:
- **Part 1 (Section 1)**: Principia Suite — 벤치마크 + 데이터셋 + 검증 벤치마크
- **Part 2 (Section 2)**: RLLM — LM-as-RM 기반 통합 포스트 트레이닝
- **Part 3 (Section 3)**: ParaGator — 테스트 타임 aggregation

각 파트가 독립적인 실험 설계, 관련 연구, 분석을 포함하며, 파트 간 상호작용이 제한적이다. Part 2(RLLM)는 Part 1(Principia)과 독립적으로도 작동하며, Part 3(ParaGator)도 마찬가지다. 70페이지 분량은 리뷰어의 심층 평가를 어렵게 만들고, 각 기여의 핵심 메시지가 희석될 수 있다.

**권고**: 논문의 통합적 내러티브("수학적 객체 추론의 데이터 → 학습 → 추론 파이프라인")를 서론에서 더 명확히 제시하고, 세 기여 간의 시너지를 정량적으로 보여주는 실험(예: RLLM + ParaGator 조합 결과)을 추가하라.

### M2. RLLM 실험의 모델 규모 제한 (실험 설계)

RLLM의 핵심 실험(Table 6-9)은 **Qwen3-1.7B 정책 모델**에 집중되어 있다. 1.7B 규모에서의 결과가 더 큰 모델(7B, 14B, 70B)에서도 유지되는지 검증이 부족하다. Appendix에 OctoThinker-8B 및 Qwen3-8B 결과가 있으나, 이들은 보조적 실험에 그친다.

특히 우려되는 점:
- Generator-verifier gap이 "essential"하다고 주장하면서(Section 2.4.2), 1.7B policy + 32B RM 조합만 집중 테스트
- 더 큰 정책 모델에서는 이 gap이 줄어들어 RLLM의 이점이 감소할 가능성
- RLHF와 RLVR 대비 RLLM의 우위가 모델 규모에 따라 어떻게 변하는지 불명확

**권고**: 최소한 7B-14B 규모의 정책 모델에서 RLLM vs RLHF vs RLVR 비교를 본문에 포함하거나, 규모 확장성에 대한 명시적 한계 논의를 추가하라.

### M3. Principia Collection의 합성 데이터 품질 검증 부족 (방법론)

248,748개 문제-답 쌍이 GPT-OSS-120B로 생성되었으나, 품질에 대한 체계적 검증이 부족하다:

- **정답률 검증**: 다수결 투표(majority voting)와 union-find 알고리즘으로 필터링했으나, 134,172개(전체의 35%)를 제거한 후에도 남은 데이터의 정답률에 대한 인간 평가가 없다
- **난이도 분포**: "graduate-level STEM"이라고 주장하지만, 실제 대학원 수준 전문가에 의한 난이도 검증이 부재
- **다양성**: 9,573 subject entities에서 생성했으나, 실제 생성된 문제의 주제 편향(over-representation) 분석이 없다

Figure 4에서 토큰 길이 분포만 비교하고 있으나, 이것이 품질이나 난이도의 대리 지표가 될 수 없다.

**권고**: (1) 무작위 샘플 200-500개에 대한 전문가 정답률 평가, (2) 주제별/유형별 분포 분석, (3) 생성된 문제의 난이도가 실제 대학원 과정과 비교 가능한지에 대한 정성적 분석을 추가하라.

### M4. 모델 기반 검증기(o3)에 대한 순환 의존성 (방법론)

PrincipiaBench의 평가에 o3를 judge로 사용하고, RLLM의 학습에도 모델 기반 검증기를 핵심으로 사용한다. Table 3에서 o3의 인간 동의율이 94.05%로 보고되지만:

- 168개 인스턴스는 **math-verify와 o3가 불일치하는 사례만** 선별한 것이므로, 전체 모집단에서의 o3 정확도를 반영하지 않는다
- o3가 체계적으로 실패하는 유형(예: 특정 수학적 객체 유형)이 있다면, 이 편향이 벤치마크 전체에 전파된다
- PrincipiaVerifyBench 자체가 168개로 소규모이며, 8명 주석자의 수학적 전문성 수준이 명시되지 않았다

**권고**: (1) PrincipiaVerifyBench의 규모를 확대하고, (2) 불일치 사례뿐 아니라 일치 사례에서도 샘플링하여 o3의 전체적 정확도를 추정하며, (3) 주석자의 전문성 수준을 명시하라.

### M5. ParaGator의 self-aggregation 상한선 문제 (이론/실험)

Figure 21과 논문의 주요 발견 중 하나는 "self-aggregation 성능이 초기 pass@k에 의해 상한이 결정된다"는 것이다. 이는 중요한 이론적 관찰이지만, 동시에 ParaGator의 근본적 한계를 드러낸다:

- 모델이 k개 샘플 중 어떤 것도 정답을 포함하지 않으면, aggregation으로 정답을 "합성"할 수 없다
- 이 상한선 관찰에 대한 이론적 분석(왜 이런 제한이 존재하는지)이 부족
- 상한선을 돌파할 수 있는 조건(예: 부분적으로 맞는 해들의 조합)에 대한 탐구가 없다

**권고**: (1) pass@k 상한선의 이론적 근거를 분석하고, (2) 부분 정답들의 조합으로 완전 정답을 생성할 수 있는 사례가 있는지 정성적으로 탐구하라.

---

## 3. Minor Comments

### m1. 비공개 모델 의존성 (재현성)

GPT-OSS-20B, GPT-OSS-120B가 핵심적으로 사용되지만 비공개 내부 모델이다. 데이터 생성(Principia Collection), 평가(PrincipiaBench judge), RLLM의 교사 모델로 사용되어, 외부 연구자가 이 결과를 재현하거나 확장하기 어렵다. 공개 모델(Qwen3-235B, o3)과의 비교 결과를 더 강조하는 것이 바람직하다.

### m2. 비용 분석 부재 (실용성)

RLLM은 32B LM-as-RM을 추론에 반복적으로 사용하고, ParaGator는 k개 샘플 + 다수의 aggregation 라운드를 필요로 한다. 각 방법의 학습 및 추론 비용(GPU 시간, API 비용)에 대한 비교 분석이 없어, 실용적 적용 가능성을 판단하기 어렵다.

### m3. 교차 형식 일반화의 메커니즘 설명 부족 (분석)

Table 2와 Figure 1(c)에서 Principia Collection 학습이 AIME(numerical)과 GPQA-Diamond(MCQA)에서도 개선을 가져온다는 핵심 주장이 있으나, **왜** 수학적 객체 학습이 수치 답변과 MCQ 성능을 향상시키는지에 대한 메커니즘 분석이 부족하다. 단순히 "cross-format reasoning gains"라고 기술하는 것을 넘어, 학습 과정에서 어떤 능력이 향상되는지(예: 중간 추론 단계의 질, 표현 다양성)에 대한 정성적 분석이 필요하다.

### m4. Table 2 가독성 (표현)

Table 2는 4개 모델 그룹 × 9개 벤치마크 + 17개 참조 모델로 매우 밀집되어 있다. 핵심 비교(동일 base LM에서의 Principia vs baseline 비교)가 주변 정보에 묻힌다. 핵심 결과를 강조하는 별도의 요약 테이블이나 시각화를 추가하면 좋겠다.

### m5. PrincipiaBench의 문제 유형별 분석 부족 (분석)

6가지 수학적 객체 유형(equation, inequality, interval, set, matrix, piecewise function)에 따른 모델 성능 차이가 보고되지 않았다. 특정 유형에서 모델이 특히 어려워하는지, Principia Collection 학습이 어떤 유형에서 가장 큰 개선을 가져오는지에 대한 유형별 분석이 유용할 것이다.

### m6. On-policy vs Off-policy 비교의 제한된 변수 (RLLM)

Table 11에서 on-policy가 off-policy를 능가함을 보여주지만, Qwen3-32B 단일 RM에서만 비교한다. 다른 규모의 RM에서도 이 효과가 유지되는지, on-policy 학습의 추가 비용 대비 이점이 어떤 조건에서 정당화되는지에 대한 분석이 필요하다.

### m7. 수식 표기의 일관성 (표현)

PDF 추출 과정에서의 문제일 수 있으나, 일부 수식 표기가 불일치한다 (예: 제목의 "RLLM"이 본문에서는 "R_LLM"과 혼용). 수식 표기의 일관성을 점검할 것을 권고한다.

---

## 4. 저자에게 묻는 질문

1. **Generator-verifier gap 의존성**: RLLM이 효과적이려면 RM이 정책 모델보다 "충분히 강해야" 한다고 주장하는데, 이 gap의 최소 크기에 대한 정량적 기준이 있는가? 예를 들어 7B policy + 14B RM으로는 충분한가?

2. **Principia Collection의 커버리지**: MSC2020과 PhySH에서 9,573개 subject entities를 추출했는데, 이 중 실제로 유효한 문제가 생성된 entities의 비율은 얼마인가? 특정 하위 분야에 편중되어 있는가?

3. **ParaGator의 aggregation 라운드 수**: 실험에서 3 라운드까지 보여주었는데, 수렴까지의 최적 라운드 수를 결정하는 기준이 있는가? 추가 라운드의 한계수익(marginal return)은 어떤가?

4. **교차 형식 일반화의 방향성**: Principia Collection(수학적 객체) → AIME(수치)/GPQA(MCQ)로의 전이를 보였는데, 반대 방향(수치 데이터 → 수학적 객체)의 전이는 왜 효과가 제한적인가? 이 비대칭성에 대한 가설이 있는가?

5. **Non-verifiable 작업에서의 RLLM**: Table 9에서 AlpacaEval/ArenaHard에서의 RLLM 성능이 보고되었는데, 이 설정에서 pairwise LM-as-RM을 사용한 이유와, pointwise 대비 pairwise의 비용-성능 트레이드오프는 어떠한가?

---

## 5. 재현성 및 투명성 평가

| 항목 | 평가 |
|------|------|
| 데이터 공개 | **양호** — PrincipiaBench, Principia Collection 모두 HuggingFace에 공개 |
| 코드 공개 | **미확인** — 학습 코드(GRPO, RLLM)의 공개 여부가 명시되지 않음 |
| 모델 공개 | **부분적** — 학습된 모델의 공개 계획이 언급되지 않음 |
| 핵심 의존성 | **제한적** — GPT-OSS-20B/120B(비공개)에 대한 의존도가 높아 완전한 재현 불가 |
| 하이퍼파라미터 | **양호** — Section 1.4.1, 2.4.1에 상세히 기술 |
| 평가 프로토콜 | **양호** — PrincipiaVerifyBench를 통한 검증기 신뢰성 검증 포함 |

---

## 6. 관련성 평가 (finance_LLM 프로젝트)

본 논문은 finance_LLM 프로젝트와 다음과 같은 연결점이 있다:

- **벤치마크 설계 방법론**: Principia의 "MCQ 옵션 제거 → 성능 하락 측정" 접근은 FinanceReasoning에서의 메타인지 평가와 유사한 "shortcut 의존성" 문제를 다룬다
- **모델 기반 검증기**: RLLM의 LM-as-RM 개념은 LLM의 자체 평가 능력(메타인지)과 관련되며, IC(Information Conflict) 탐지 연구에서의 검증기 설계에 참고 가능
- **합성 데이터 생성 파이프라인**: Principia Collection의 entity → strategy → problem → answer → verification 파이프라인은 FinanceReasoning의 변환(transformation) 파이프라인과 방법론적으로 비교 가능

**관련성 수준: 중간** — 직접적인 금융 도메인 논문은 아니지만, LLM 평가 방법론과 학습 레시피 측면에서 참고 가치가 있다.
