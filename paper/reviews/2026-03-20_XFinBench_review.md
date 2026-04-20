# Peer Review: XFINBENCH — Benchmarking LLMs in Complex Financial Problem Solving and Reasoning

**논문**: XFINBENCH: Benchmarking LLMs in Complex Financial Problem Solving and Reasoning
**저자**: Zhihan Zhang, Yixin Cao, Lizi Liao (Singapore Management University, Fudan University)
**학회**: ACL 2025 Findings
**페이지**: 44 pages (본문 11 + Appendix 33)
**리뷰 날짜**: 2026-03-20

---

## 1. 이 논문이 풀려는 문제 (Problem Statement)

### 1.1 핵심 문제 정의

기존 금융 도메인 LLM 벤치마크들은 두 가지 근본적 한계를 가진다:

1. **단순한 과제에 집중**: TAT-QA, FinQA, ConvFinQA 등 기존 데이터셋은 기업 재무제표에서 숫자를 추출하고 사칙연산을 수행하는 "Quantity Extraction" 수준에 머무른다. 예: "2021년 매출액은 얼마인가?" → 테이블에서 숫자를 찾아 보고하는 수준.

2. **고급 추론 역량 미평가**: 실제 금융 전문가가 수행하는 복잡한 문제 해결 — 시간에 따른 재무 데이터 추론(temporal reasoning), 미래 경제 추세 예측(future forecasting), 다양한 시나리오 하의 의사결정(scenario planning), 수학적 모델 구축(numerical modelling) — 을 평가할 수 없다.

3. **멀티모달 부재**: 금융 실무에서는 차트, 그래프, 테이블이 핵심인데, 기존 벤치마크는 텍스트 위주이거나 시각적 데이터가 있더라도 "이 차트에서 최대값은?" 같은 단순 인식에 그친다.

### 1.2 저자들의 접근: "5가지 금융 추론 역량"으로 재정의

저자들은 금융 문제 해결에 필요한 역량을 5개로 분류하고, 이를 체계적으로 평가하는 벤치마크를 구축한다:

| 역량 | 정의 | 예시 |
|------|------|------|
| **Terminology Understanding** (56.1%) | 금융 개념, 용어, 공식의 정확한 이해 | "strip을 보유한 투자자가 큰 주가 상승 시 더 큰 이익을 얻는다" — True/False 판단 |
| **Temporal Reasoning** (21.7%) | 시계열 데이터의 시간적 관계 추론 | 배당금이 시간에 따라 변하는 주식의 현재 가치 계산 (다기간 할인) |
| **Future Forecasting** (5.0%) | 경제 이론에 기반한 미래 추세 예측 | "금리 상승 시 채권 수요·공급 다이어그램에서 균형점이 어디로 이동하나?" |
| **Scenario Planning** (7.6%) | 다양한 미래 시나리오 분석 → 최적 의사결정 | "American call option의 binomial tree 3단계 가격 책정" |
| **Numerical Modelling** (17.2%) | 재무제표 기반 구조화된 수리 모델 구축 | "2007년 손익계산서로부터 영업현금흐름 계산" |

---

## 2. 접근 방법 상세 분석 (Methodology Deep Dive)

### 2.1 데이터셋 구축 파이프라인

전체 파이프라인은 **4단계**로 구성된다:

#### Stage 1: 초기 데이터 수집 (교과서 → Raw QA)
- **소스**: 대학원 수준 금융 교과서 3권
  - *Fundamentals of Corporate Finance* (Stephen A. Ross) — 기업재무 22장
  - *Options, Futures and Other Derivatives* (John C. Hull) — 파생상품 32장
  - *The Economics of Money, Banking and Financial Markets* (Frederic S. Mishkin) — 화폐/금융 25장
- **추출 방법**: pdfplumber로 OCR → 3명의 어노테이터가 각 챕터 끝 "after-class questions" 수집
- **결과**: 2,018개 after-class 질문 (343개는 시각/테이블 맥락 포함)
- **과제 분류**: 수집된 질문을 3가지 과제로 분류
  - Statement Judging (813개): 금융 개념의 참/거짓 판단
  - Multi-choice QA (624개): 전략적 의사결정과 예측
  - Financial Calculation (858개): 수학적 추론과 계산

#### Stage 2: GPT-4o Enhanced Annotation (Generate-then-Verify)

교과서 문제의 핵심 난제: 대부분이 **개방형(open-ended)**이라 LLM 평가가 어려움.
예: "옵션과 선물계약의 장단점을 논하라" → 정답 기준이 모호.

**해결책 — 2단계 프레임워크:**

**(a) Generation Stage**: GPT-4o에게 few-shot 프롬프트로 변환 지시
- **Statement Judging**: 개방형 질문에서 True/False 문장 추출. True/False 균형을 위해 2개의 다른 프롬프트 템플릿 사용
- **Multi-choice QA**: STARC 규칙에 따라 정답 1개 + 그럴듯한 오답 2개 생성. 오답은 정답과 유사한 길이/어휘를 사용하되 상호 배타적이어야 함
- **Financial Calculation**: 연속된 서브 질문을 독립적인 단일 질문으로 분해

**(b) Verification Stage**: 품질 필터링 (4가지 기준)
1. 질문이 정답 도출에 필요한 모든 정보를 포함하는가? (Completeness)
2. 정답이 원본 교과서 답안 기준으로 정확한가? (Correctness)
3. Statement Judging: 같은 after-class 질문 내 True 문장이 False 문장의 근거를 제공하지 않는가?
4. Multi-choice QA: 오답이 정답과 상호 배타적인가?

→ **35.2%의 질문이 폐기**되어 최종 4,235개

#### Stage 3: Knowledge Bank 구축
- 교과서 뒷부분 Subject Index에서 금융 용어 추출
- 3명의 어노테이터가 해당 페이지에서 정의 수동 추출
- 결과: **3,032개 용어, 1,766개 고유 정의** (34.3%가 수학 공식 포함)
- 각 질문에 1~3개의 관련 용어를 인간이 태깅 (평균 1.3개/질문)

#### Stage 4: 인간 품질 검증
- 3명의 검증자가 각 예제를 5점 척도로 평가
  - Question Fluency: 92.9% ≥5점
  - Question Completeness: 95.2% ≥5점
  - Answer Correctness: 96.3% ≥5점
  - Knowledge Helpfulness: 94.1% ≥5점

### 2.2 실험 설계

#### 평가 대상 모델 (18개)
- **Multimodal LLMs** (9개): gpt-4o, gpt-4o-mini, claude-3.5-sonnet, claude-3-opus, claude-3-haiku, gemini-1.5-flash, gemini-1.5-pro, Llama-3.2-90B-Vision, Llama-3.2-11B-Vision
- **Text-only LLMs** (9개): o1, o1-mini, Llama-3.1-405B, deepseek-chat, Llama-3.1-70B, Llama-3-70B, Llama-3.1-8B, Llama-3-8B, Mixtral-8×7B

#### 추론 방법
- **Chain-of-Thought (CoT)**: 모든 과제에 적용. 단계별 사고 과정을 유도
- **Program-of-Thought (PoT)**: Financial Calculation에 추가 적용. Python 코드를 생성하여 계산 수행

#### 평가 지표
- **Accuracy**: Statement Judging, Multi-choice QA
- **Acc_ERR@5**: Financial Calculation 전용 — 정답의 ±0.5% 오차 허용
- **Exact Match**: Financial Calculation의 엄격한 평가

#### Knowledge Augmentation 실험 (3가지 설정)
1. **Oracle**: 인간이 태깅한 정답 용어를 그대로 제공 (상한선)
2. **BM25**: 스파스 검색으로 top-3 관련 용어 검색
3. **Ada Embed**: OpenAI text-embedding-ada-002로 밀집 검색하여 top-3 제공

### 2.3 핵심 실험 결과

#### Main Results (Table 3)
| 모델 | 전체 정확도 | 특이사항 |
|------|:---:|------|
| o1 (text-only) | **67.3%** | 최고 성능. 텍스트 전용이지만 추론력이 압도적 |
| claude-3.5-sonnet | 64.1% | 멀티모달 최고. visual-context에서 65.3%로 1위 |
| gpt-4o | 63.6% | 전반적으로 균형 잡힌 성능 |
| Human experts | **79.8%** | o1 대비 +12.5% |

#### 핵심 발견사항
1. **인간-LLM 격차의 양극화**: Terminology Understanding에서는 LLM이 인간에 근접(87.6% vs 90.9%), 하지만 Temporal Reasoning(34.2% vs 81.1%)과 Scenario Planning에서 큰 격차
2. **PoT가 대부분 모델에서 역효과**: LLM이 생성한 Python 코드의 실행 실패가 주원인. 단, o1은 실행률이 낮음에도 정확도가 높아 추론 능력으로 보상
3. **Knowledge Augmentation의 제한적 효과**: Oracle 설정에서도 큰 개선이 없음 → 문제가 "지식 부족"이 아니라 "추론 능력 부족"에 기인
4. **소형 오픈소스 모델만 지식 증강 효과**: Llama-3.1-8B는 Oracle 설정에서 눈에 띄는 개선. 대형 모델은 이미 지식을 내재화하고 있어 외부 지식이 덜 유용

#### Error Analysis 결과
- **Financial Calculation** (o1, 400개 샘플): 55.2%는 정확한 추론. 오류 중 Rounding Error(중간 계산 반올림)와 Knowledge Misuse(잘못된 공식 적용)가 주요 원인
- **Visual-context** (GPT-4o, 100개 샘플): **Blindness**(곡선의 교차점/위치 인식 실패)가 35%, Knowledge Misuse 3%
- **Knowledge Augmentation 실패** (GPT-4o, 100개): Reasoning Error(지식과 무관한 추론 오류), OverThinking(지식이 직접 답을 주는데 불필요한 추가 추론), Over Reliance(지식에 과도하게 의존하여 간단한 접근법 무시)

---

## 3. 요약 (Summary Statement)

본 논문은 LLM의 복잡한 금융 문제 해결 및 추론 능력을 평가하기 위한 벤치마크 XFINBENCH를 제안한다. 3권의 대학원 수준 금융 교과서에서 추출한 4,235개의 예제로 구성되며, 5가지 핵심 능력을 평가한다. 18개 주요 LLM에 대한 실험에서 최고 성능 모델 o1(67.3%)도 인간 전문가(79.8%) 대비 12.5% 뒤처지며, 특히 temporal reasoning과 scenario planning에서 큰 격차를 보인다.

### 총평: Minor Revisions 권고

### 핵심 강점
- **체계적인 벤치마크 설계**: 5가지 금융 역량을 명확히 정의하고, 3가지 과제(statement judging, multi-choice QA, financial calculation)로 구조화
- **높은 데이터 품질**: Generate-then-Verify 프레임워크 + 3인 인간 검증(정답 정확도 98.0%, 질문 완전성 96.8%)
- **포괄적 실험**: 18개 모델 평가, CoT/PoT 비교, knowledge augmentation 3가지 설정, 체계적 error analysis
- **멀티모달 지원**: 시각적 맥락(차트, 그래프)을 포함한 질문으로 기존 텍스트 전용 금융 벤치마크의 한계 극복
- **재현성**: 코드, 데이터셋 공개 (GitHub 링크 제공)

### 핵심 약점
- **데이터 소스의 제한성**: 교과서 3권에만 의존하여 실제 금융 시장의 복잡성 반영이 부족
- **평가 모델의 시대성**: 2024년 기준 모델만 평가하여 급변하는 LLM 생태계에서의 유효기간이 짧음
- **과제 유형의 불균형**: Future Forecasting(5.0%)과 Scenario Planning(7.6%)의 비율이 매우 낮아 이 역량에 대한 평가 신뢰성이 제한적

---

## 4. Major Comments

### Major 1: 교과서 기반 데이터의 한계 — 실세계 금융 문제와의 괴리

XFINBENCH의 모든 데이터가 3권의 교과서 "after-class questions"에서 파생된다. 이는 데이터 품질과 정답의 명확성을 보장하지만, 근본적으로 **교과서적 문제(textbook problems)**와 **실제 금융 문제(real-world financial problems)** 사이의 괴리를 야기한다.

**구체적 문제점:**
- 교과서 문제는 정형화된 조건과 깔끔한 정답이 존재하지만, 실제 금융에서는 불완전한 정보, 노이즈, 모호한 조건이 일반적
- 논문 제목의 "Complex Financial Problem Solving"이 암시하는 것과 달리, 대부분의 문제가 공식 적용이나 개념 이해 수준
- 3권의 교과서가 Corporate Finance, Derivatives, Banking/Money를 다루지만, Asset Management, Quantitative Finance, Behavioral Finance 등 중요 분야가 누락

**제안:**
- "Complex"의 정의를 더 명확히 하고, 교과서 기반 벤치마크의 한계를 Discussion에서 더 솔직하게 인정할 것
- 실제 금융 보고서, 뉴스 기반 문제를 포함하는 확장 가능성을 논의할 것

### Major 2: 능력(Capability) 분류의 불균형과 평가 신뢰성

5가지 핵심 역량의 분포가 극도로 불균형하다:

| 역량 | 비율 | 테스트셋 수 |
|------|------|-----------|
| Terminology Understanding | 56.1% | ~1,814 |
| Temporal Reasoning | 21.7% | ~703 |
| Numerical Modelling | 17.2% | ~557 |
| Scenario Planning | 7.6% | ~246 |
| Future Forecasting | 5.0% | ~162 |

Future Forecasting은 162개에 불과하여 모델 간 성능 차이의 통계적 유의성을 확보하기 어렵다. 특히 논문에서 "LLMs significantly lag behind human experts in temporal reasoning and scenario planning"이라고 주장하지만, Scenario Planning의 테스트셋은 246개에 불과하다.

**제안:**
- 소수 역량(FF, SP)에 대한 신뢰 구간(confidence interval) 보고
- 향후 데이터 확장 시 역량 간 균형 확보 계획 제시

### Major 3: GPT-4o를 활용한 데이터 생성의 순환 편향 (Circular Bias) 우려

데이터셋 구축에서 GPT-4o가 핵심 역할을 한다:
1. 개방형 질문을 객관식/계산 문제로 변환
2. Statement judging 과제의 true/false 문장 생성
3. Multi-choice QA의 오답 보기(distractors) 생성

그런데 평가 대상에 GPT-4o가 포함되어 있다. GPT-4o가 생성한 distractors는 GPT-4o 자신의 "blind spots"를 반영하지 않을 가능성이 높다. 즉, GPT-4o가 구별하기 어려운 오답을 GPT-4o가 만들기는 힘들다.

- 논문에서 human verification(35.2% 폐기)으로 이를 완화했지만, 남은 64.8%의 질문에서 이 편향이 완전히 제거되었다는 보장이 없음
- 특히 multi-choice QA에서 distractors의 질이 모델 성능에 직접적 영향을 미침

**제안:**
- GPT-4o 생성 질문 vs 순수 인간 작성 질문에서의 모델 성능 차이를 분석하는 ablation 추가
- 또는 distractors가 실제로 모델을 혼란시키는지(distractor analysis) 검증

### Major 4: 통계적 유의성 검증 부재

Table 3의 메인 결과에서 모든 성능 비교가 단일 실행(single run) 결과로 보고된다. LLM은 temperature, sampling 등에 따라 출력이 달라질 수 있으므로:
- 모델 간 성능 차이의 통계적 유의성이 불분명
- 예: claude-3.5-sonnet(64.1%) vs gpt-4o(63.6%)의 0.5% 차이가 유의미한지 판단 불가
- Human performance baseline도 3명의 전문가에 기반하여 분산이 클 수 있음

**제안:**
- 최소한 주요 모델에 대해 복수 실행(3-5회)의 평균 ± 표준편차 보고
- 또는 bootstrap confidence interval 제공

---

## 5. Minor Comments

### Minor 1: PoT(Program-of-Thought) 분석의 깊이 부족
Figure 4(a)에서 PoT의 실행률과 정확도의 관계를 보여주지만, **왜** 특정 모델(Llama-3.1-405B)이 실행 가능한 Python 코드를 생성하지 못하는지에 대한 분석이 부족하다. 코드 생성 실패의 구체적 패턴(문법 오류, 라이브러리 의존성, 잘못된 로직 등)을 분류하면 더 유용한 인사이트를 제공할 수 있다.

### Minor 2: 멀티모달 평가의 불완전성
시각적 맥락 질문(146개 이미지, 330개 테이블)이 전체의 약 11%에 불과하다. "multi-modal context"를 논문의 핵심 차별점으로 제시하면서도 멀티모달 질문의 비중이 낮다. 특히 이미지 질문이 146개뿐이어서 멀티모달 능력에 대한 일반화가 제한적이다.

### Minor 3: Knowledge Bank의 활용 방식이 단순
3,032개 금융 용어의 지식 은행을 구축했지만, 활용 방식이 단순한 top-n retrieval(n=3)에 한정된다. RAG(Retrieval-Augmented Generation)의 다양한 전략(chunking, reranking, multi-hop retrieval 등)이나 fine-tuning 기반 지식 주입은 시도되지 않았다.

### Minor 4: Acc_ERR@5 지표의 관대함
Financial calculation에서 정답의 ±0.5% 오차를 허용하는 Acc_ERR@5 지표를 사용한다. 금융에서 0.5%의 오차는 맥락에 따라 매우 클 수 있다 (예: 이자율 계산, 옵션 가격). Exact match와 다양한 오차 범위(0.1%, 0.5%, 1%, 5%)에서의 성능을 함께 보고하면 더 정보가 풍부할 것이다.

### Minor 5: 인간 성능 베이스라인의 제한
인간 전문가 3명으로 1,000개 문제 서브셋에서만 평가했다. 3명은 표본이 매우 작으며, 전문가 간 편차(inter-annotator agreement)가 보고되지 않았다. 특히 금융 세부 전공에 따라 성능 차이가 클 수 있다 (예: 파생상품 전공자 vs 거시경제 전공자).

### Minor 6: 교과서 판본 명시 부족
Table 5에서 교과서 3권을 명시하지만, **판본(edition)**이 기재되지 않았다. 금융 교과서는 판본마다 문제가 다를 수 있으므로 재현성을 위해 정확한 판본을 명시해야 한다.

### Minor 7: Error Analysis의 표본 크기
Error analysis가 o1에서 400개, GPT-4o에서 100개의 랜덤 샘플에 기반한다. 특히 GPT-4o의 visual-context error analysis(100개)는 표본이 작다. 오류 유형의 비율(e.g., Blindness 35%, Knowledge Misuse 3%)이 이 작은 표본에서 안정적인지 의문이다.

### Minor 8: 최신 모델 미포함
평가 대상이 2024년 중반 기준 모델이다. o1-preview, GPT-4-turbo, Claude 3.5 Sonnet은 포함되었지만, GPT-4o (2024-11), Claude 3.5 Haiku, Gemini 2.0 등 이후 모델이 누락되었다. ACL 2025 발표 시점을 고려하면 camera-ready에서 업데이트가 권장된다.

### Minor 9: 교과서 저작권 관련 표현의 모호함
"sourced from publicly available platforms on the Internet, with strict adherence to copyright and licensing regulations"이라고 하지만, 교과서 문제의 직접 사용이 저작권법상 어떻게 정당화되는지 더 명확히 서술할 필요가 있다 (fair use 조항 등).

### Minor 10: Related Work에서 금융 벤치마크 비교의 세분화 필요
Table 1에서 기존 데이터셋과의 비교가 이루어지지만, "Complex-Problem" 열의 기준이 명확하지 않다. XFINBENCH만 체크 표시가 있는데, 이 기준의 정의와 다른 데이터셋이 해당하지 않는 구체적 이유를 보충해야 한다.

---

## 6. 방법론 및 통계적 엄밀성 평가

### 데이터 구축 방법론
- **강점**: Generate-then-Verify 프레임워크는 자동 생성의 효율성과 인간 검증의 정확성을 잘 결합
- **강점**: 35.2%의 질문 폐기는 품질 기준이 엄격함을 시사
- **약점**: GPT-4o에 대한 의존도가 높아 모델 편향 가능성 존재

### 평가 방법론
- **강점**: CoT + PoT 이중 평가, exact match + Acc_ERR@5 이중 지표
- **강점**: Oracle / BM25 / Ada Embed 3가지 knowledge augmentation 설정으로 체계적 비교
- **약점**: 단일 실행 결과, 신뢰 구간 미보고

### 재현성
- **강점**: GitHub에 코드와 데이터셋 공개, 모델 버전 명시 (Table 9)
- **약점**: 교과서 판본 미명시, 프롬프트 템플릿이 Appendix G에 있으나 검증에는 한계

---

## 7. Figure 및 Data Presentation 평가

- **Figure 1** (레이더 차트): 5가지 역량에 대한 모델/인간 비교를 효과적으로 시각화. 다만 o1과 Llama-3.1-405B가 visual-context를 커버하지 못한다는 점이 범례에서만 설명되어 혼동 가능
- **Figure 2** (예시): 3가지 과제의 구체적 예시를 잘 보여주며 벤치마크의 성격을 직관적으로 전달
- **Figure 3** (토픽 분포): 28개 금융 토픽의 분포를 보여주나 2.5% 미만 토픽이 생략되어 불완전
- **Figure 4**: 3개 서브플롯이 각각 PoT 실행률-정확도, 지식 증강 개선, 역량별 개선을 보여줌. 정보량이 풍부하나 서브플롯이 작아 가독성 개선 필요
- **Figure 5-6** (에러 분석): 구체적 케이스 스터디가 포함되어 에러 유형 이해에 도움. 다만 Figure 6의 수식 렌더링이 PDF에서 깨질 수 있음
- **Table 3** (메인 결과): 가장 핵심적인 테이블. 구조가 명확하고 최고/차선 성능 하이라이팅이 유용. 다만 신뢰 구간 미포함

---

## 8. 윤리적 고려사항

- 공개 금융 데이터만 사용, 개인정보 미포함 — 적절
- 어노테이터 보상이 기관 정책에 따름 — 언급은 있으나 구체적 금액/시간당 비용 미공개
- 교과서 저작권 관련 "strict adherence"를 주장하나 구체적 라이선스 정보 부재
- LLM 기반 금융 문제 해결의 잠재적 오용 가능성(투자 조언으로의 오해)에 대한 논의 부재

---

## 9. 글쓰기 품질 평가

- **구조**: EMNLP/ACL 표준 형식을 잘 따르며 논리적 흐름이 명확
- **명료성**: 전반적으로 잘 쓰여졌으며 기술적 설명이 정확
- **간결성**: 본문 11페이지로 적절하나 Appendix 33페이지는 다소 과도 (전체 44페이지)
- **접근성**: 금융 비전문가도 주요 발견을 이해할 수 있도록 잘 설명됨

---

## 10. 저자에게 묻는 질문

1. **GPT-4o 생성 편향**: Multi-choice QA에서 GPT-4o가 생성한 distractors에 대해 GPT-4o 자신의 정답률이 다른 모델 대비 유의미하게 높거나 낮지는 않은지? Distractor 품질에 대한 정량적 분석이 있는가?

2. **역량 라벨링의 일관성**: 3명의 어노테이터 간 capability labeling의 inter-annotator agreement(e.g., Cohen's kappa)는 얼마인가?

3. **데이터 오염(Data Contamination)**: 교과서 after-class 문제가 인터넷에 공개된 솔루션 매뉴얼에서 추출되었다면, 평가 대상 LLM의 학습 데이터에 이미 포함되어 있을 가능성은 어떻게 통제했는가? 특히 o1이나 GPT-4o의 학습 데이터에 해당 교과서 솔루션이 포함되어 있다면 성능이 과대평가될 수 있다.

4. **문제 난이도 분석**: 문제의 난이도(e.g., 인간 정답률 기준)에 따른 모델 성능 분석이 있는가? 쉬운 문제에서의 성능과 어려운 문제에서의 성능을 분리하면 모델의 실질적 추론 능력을 더 잘 평가할 수 있을 것이다.

5. **실험 재현성**: 모델 API 호출 시 temperature와 sampling 설정은 어떻게 했는가? Table 9에 모델 소스는 명시되어 있으나 decoding 파라미터가 불명확하다 (Section D.1에서 "default values"라고만 명시).

---

## 최종 평가

| 평가 항목 | 점수 (1-5) |
|----------|:---:|
| 참신성 (Novelty) | 3.5/5 |
| 기술적 건전성 (Soundness) | 4.0/5 |
| 실험의 철저함 | 4.0/5 |
| 재현성 | 4.0/5 |
| 글쓰기 품질 | 4.0/5 |
| 학문적 기여도 | 3.5/5 |

**총평**: ACL 2025 Findings에 적합한 수준의 벤치마크 논문. 금융 도메인에서 LLM의 복잡한 추론 능력을 평가하기 위한 체계적이고 잘 설계된 벤치마크를 제안한다. 데이터 품질이 높고 실험이 포괄적이지만, 교과서 기반 데이터의 한계, GPT-4o 생성 편향 가능성, 역량 간 데이터 불균형, 통계적 유의성 검증 부재가 주요 개선점이다. Minor revisions로 이러한 점들을 보완한다면 학술 커뮤니티에 유용한 기여가 될 것이다.

**판정: Minor Revisions (ACL 2025 Findings 수준에 적합)**
