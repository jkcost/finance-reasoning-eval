# Peer Review: MAPLE: Multi-Agent Portfolio Learning Ensemble with Hierarchical Architecture for Heterogeneous Asset Management

**논문 정보:**
- 제출 학회: KDD'26 (32nd ACM SIGKDD Conference)
- 저자: Anonymous (더블 블라인드)
- 페이지: 12페이지 (본문 8 + 부록 4)
- 리뷰 일자: 2026-03-10

---

## 1. 요약 (Summary Statement)

본 논문은 이종(heterogeneous) 자산 클래스(주식, 채권, 원자재, REITs, 암호화폐)에 걸친 포트폴리오 관리를 위해 **MAPLE**(Multi-Agent Portfolio Learning Ensemble)이라는 계층적 멀티 에이전트 LLM 프레임워크를 제안한다. (1) Dynamic Asset Selection으로 리밸런싱마다 투자 유니버스를 재구성하고, (2) Cross-Asset Risk Context를 모든 에이전트에 공유하며, (3) 3-tier 계층적 에이전트 구조(Asset Expert → Coordination → Risk Controller)로 의사결정을 분리하고, (4) Co-MARL을 통해 에이전트별 LoRA 어댑터를 학습시킨다. 170개 자산, 9개월 테스트 기간(10시드)에서 59.59% 수익률, 2.06 Sharpe ratio를 달성하여 최고 RL baseline 대비 +115%, LLM-agent baseline 대비 +349%의 성능을 보고한다.

### 전반적 평가: **Major Revisions**

### 핵심 강점
- LLM 기반 멀티 에이전트 + RL 학습(Co-MARL)의 결합이 참신하며, 5개 자산 클래스에 걸친 대규모 실험 설계가 인상적
- 10시드 평균±표준편차 보고, 모델 스케일링 분석, closed-source 모델 비교 등 실험의 재현성이 양호
- 3개의 상세한 case study(crypto 폭락, 관세 충격, 노동시장 악화)가 계층적 의사결정의 해석 가능성을 효과적으로 보여줌

### 핵심 약점
- 9개월 단일 테스트 기간으로 일반화 주장에 한계가 크며, 2025년 상반기 강세장 편향 가능성
- Baseline 비교가 불공정 — RL baseline은 수치 데이터만, MAPLE은 뉴스+비정형 데이터까지 사용
- Hedging/Rebalancing 잔차(36.29%)의 수익 기여가 불투명하며, 거래비용이 학습에만 반영되고 보고 수치에는 미포함(gross return)

---

## 2. Major Comments

### M1. 단일 테스트 기간(9개월)의 일반화 한계

2025-01~2025-09의 단일 9개월 기간만으로 시스템을 평가한다. 이 기간은 crypto 급등(BTC $74K→$100K+), AI 주식 강세 등 특정 시장 체제를 반영하며, bear market이나 장기 침체 환경에서의 성능은 알 수 없다. 저자도 Conclusion에서 "single 9-month test period"의 한계를 인정하지만, walk-forward validation은 미실시이다.

**권고:** (1) 최소 2-3개의 비겹치는 테스트 기간(예: 2022 bear market 포함)으로 확장하거나, (2) rolling window 기반 walk-forward validation을 실시하라. 9개월 단일 기간 결과만으로 "robust" 성능을 주장하기 어렵다.

### M2. Baseline 비교의 공정성 문제

MAPLE은 뉴스 텍스트, 대안 데이터, LLM 추론 능력을 활용하는 반면, RL baseline(FinRL, DeepTrader, SARL)은 순수 수치 특성만 사용한다. 이는 **방법론의 우위가 아닌 정보의 우위**일 수 있다. "algorithm choice within this paradigm has limited impact"(Section 5.2)라는 저자의 서술이 이를 암시한다.

**권고:** (1) RL baseline에도 동일한 뉴스 감성 점수를 feature로 제공하여 정보 동등 조건에서 비교하거나, (2) MAPLE에서 뉴스/비정형 데이터를 제거한 ablation을 추가하라. 현재 비교로는 LLM 아키텍처의 기여 vs. 추가 데이터 소스의 기여를 분리할 수 없다.

### M3. Hedging/Rebalancing 잔차(36.29%)의 불투명성

Table 5에서 개별 자산 클래스의 정적 수익 기여 합계는 23.3%(12.8+1.8+2.5+1.3+4.9)이며, 나머지 36.29%가 "Hedging/Rebalancing"으로 분류된다. 이는 전체 수익의 **61%**에 해당하며, "cross-class rotation timing and correlation-aware hedging"이라고 설명하지만 구체적 분해가 없다.

**권고:** (1) 이 잔차를 timing alpha(리밸런싱 효과), hedging alpha, transaction cost 영향 등으로 세분화하거나, (2) 리밸런싱 빈도(현재 5일)를 변경한 민감도 분석을 추가하라. 수익의 60% 이상이 블랙박스인 것은 해석 가능성 주장과 모순된다.

### M4. 거래비용의 이중 기준

학습 목적함수(Eq. 2)에는 거래비용(κ=0.01)이 포함되지만, 평가 지표는 "gross of transaction costs"로 보고된다(Section 5.1.4). 170개 자산을 5일마다 리밸런싱하면 turnover가 상당할 수 있으며, 실제 net return은 크게 달라질 수 있다.

**권고:** (1) Net return(거래비용 차감 후)을 반드시 병행 보고하고, (2) turnover 통계(평균 일방 회전율)를 보고하라. "prior work [37,38]을 따른다"는 것은 정당화가 아닌 관행의 답습이다.

### M5. LLM 추론의 재현성 문제

LLM의 autoregressive decoding은 본질적으로 stochastic하며(temperature, sampling), 동일 입력에도 다른 JSON 출력을 생성할 수 있다. 10시드 실험이 이를 부분적으로 다루지만, seed가 LoRA 초기화만 제어하는지, LLM decoding randomness도 제어하는지 불명확하다.

**권고:** Decoding 설정(temperature, top-p 등)을 명시하고, seed가 제어하는 난수원(LoRA init, replay buffer, decoding 등)을 명확히 기술하라.

---

## 3. Minor Comments

### m1. Seed 100의 이상치 성능

Table 6에서 seed 100은 Sharpe 2.579, Sortino 3.414로 다른 시드(Sharpe 1.98-2.02, Sortino 2.55-2.63)와 현저히 다르다. Figure 3-4의 case study도 seed 100 기반이다. Seed 100의 이상치를 main figure에 사용하는 것은 cherry-picking 우려가 있다.

**권고:** Case study에 median seed를 사용하거나, seed 100이 대표적인 이유를 설명하라.

### m2. Dynamic Asset Selection의 필터 임계값 민감도

Φ^liq(market cap >$500M), Φ^score(top quintile), Φ^risk(volatility >60%, drawdown >50%, correlation >0.85) 등의 임계값이 고정되어 있으나, 이 값들의 선택 근거와 민감도 분석이 없다.

**권고:** 주요 필터 임계값(특히 correlation 0.85, volatility 60%)의 민감도 분석을 추가하라.

### m3. Co-MARL 보상 가중치의 선택 근거

ω=(0.50, 0.30, 0.20)의 선택 근거가 불분명하다. Ablation에서 이 가중치 변화의 영향을 보여주지 않는다.

**권고:** 보상 가중치 민감도 분석 또는 선택 근거를 추가하라.

### m4. Crypto 자산의 과대 수익 기여

Table 5에서 crypto는 5.5% 배분으로 89.2% 클래스 수익률을 달성했는데, 이 기간(2025년)의 crypto 강세장 효과가 클 수 있다. 다른 시장 환경에서도 동일한 패턴이 유지되는지 불명확하다.

### m5. 9개 에이전트의 컴퓨팅 비용

Table 10에서 단일 결정에 3.4초가 소요된다. 5일 리밸런싱이므로 실시간 문제는 아니지만, 학습 시간(18시간, A100)은 상당하다. Baseline RL 기법 대비 학습 비용 비교가 없다.

### m6. Position Sizing Agent의 η 범위

Eq. 9의 η_min, η_max 구체적 수치가 본문에 없다 (Section 5.1.1 참조라 했으나 구체적 수치 미기재).

### m7. "Seqential" 오타 없음 확인

(RiskBound와 달리 오타 없음 — 글쓰기 품질 양호)

### m8. Synergy analysis의 해석

"Sum of Tier-2/3 deltas (−74.43%) exceeds joint removal (A5: −42.21%) by 32.22pp"에서 이를 "mutual reinforcement"로 해석하지만, 이는 단순히 개별 제거 시 나머지 컴포넌트들이 부분적으로 보상하기 때문일 수 있다. "Super-additive"보다는 "sub-additive degradation"이 더 정확한 표현이다.

---

## 4. 방법론 및 통계적 엄밀성 평가

### 실험 설계
- **장점:** 10시드 평균±표준편차 보고, 3개 시드 학습 재현성 확인(Figure 5), 모델 스케일링 분석(4개 모델 크기)
- **장점:** 포괄적 ablation(9개 변형, Table 4)으로 각 컴포넌트 기여도 분석
- **약점:** 9개월 단일 테스트 기간, gross return 보고, 정보 비동등 baseline 비교

### 통계적 측면
- 10시드 실험은 RiskBound(seed=0 하나)보다 크게 개선됨
- 그러나 seed 100의 이상치 미설명, baseline에 대한 통계 검정(p-value) 미수행

### 재현성
- LoRA 설정, PPO 하이퍼파라미터, 프롬프트 템플릿(Appendix F) 등이 매우 상세히 기술됨
- 코드/모델 가중치는 "acceptance 후 공개" — 리뷰 시점에 검증 불가

---

## 5. 재현성 및 투명성 평가

| 항목 | 평가 |
|------|------|
| 코드 공개 | Acceptance 후 공개 예정 |
| 데이터 출처 | FMP API, CoinGecko 명시 |
| 하이퍼파라미터 | 매우 상세 (Table 2, Appendix F) |
| 학습 절차 | Algorithm 1 + Appendix F.5 |
| 랜덤 시드 | 10시드 평균±표준편차 |
| 프롬프트 | 전체 템플릿 공개 (Appendix F.1) |
| 하드웨어 | A100 40GB 명시 |

---

## 6. 글쓰기 품질 평가

- 논문 구조가 체계적: 2개 도전과제 → 4개 메커니즘 → RQ1-RQ4
- Figure 2의 전체 프레임워크 다이어그램이 복잡한 시스템을 효과적으로 시각화
- Case study(Section 5.5)가 실제 시장 이벤트(crypto 폭락, 관세 충격, 고용 악화)와 연결되어 설득력 있음
- Table 1의 feature 비교표가 MAPLE의 차별성을 명확히 보여줌
- 프롬프트 전문 공개(Appendix F.1)는 LLM 연구의 모범 사례

---

## 7. 저자에게 드리는 질문

1. **RL baseline에 뉴스 감성 점수를 추가 feature로 제공하면 성능 차이가 얼마나 줄어드는가?** MAPLE의 우위가 LLM 아키텍처에서 오는지, 추가 데이터 소스에서 오는지 분리할 수 있는가?

2. **Net return(거래비용 차감 후)은 얼마인가?** 5일 리밸런싱에서 평균 turnover는 어떤 수준인가?

3. **Seed 100이 왜 다른 시드보다 Sharpe가 29% 높은가?** (2.579 vs 평균 1.99) 이 이상치는 어떤 요인에 의한 것인가?

4. **2022년 같은 bear market 기간에서도 MAPLE이 우수한 성능을 보이는가?** Training 기간(2021-07~2024-06)에 이 기간이 포함되어 있으므로, 별도 out-of-sample 테스트가 가능한가?

5. **LLM decoding의 temperature/top-p 설정은 무엇인가?** Structured JSON 출력의 일관성은 어떻게 보장하는가?

6. **Hedging/Rebalancing 잔차 36.29%를 timing alpha와 hedging alpha로 분해할 수 있는가?**

---

## 8. 최종 평가 요약

| 평가 항목 | 점수 (1-5) |
|-----------|:---------:|
| 참신성 (Novelty) | 4.5 |
| 기술적 건전성 (Soundness) | 3.0 |
| 실험 설계 (Experimental Design) | 3.0 |
| 명확성 (Clarity) | 4.0 |
| 재현성 (Reproducibility) | 4.0 |
| 의의 (Significance) | 3.5 |
| **종합** | **3.5 / 5** |

**최종 권고: Major Revisions**

MAPLE의 아이디어(LLM 멀티 에이전트 + Co-MARL + 동적 유니버스)는 매우 참신하고 시의적절하다. 그러나 (1) 단일 9개월 테스트 기간의 일반화 한계, (2) RL baseline과의 정보 비동등 비교, (3) 수익의 61%를 차지하는 Hedging/Rebalancing 잔차의 불투명성, (4) gross return 보고가 핵심 약점이다. 이 중 (2)와 (4)는 비교적 쉽게 해결 가능하며, (1)은 추가 테스트 기간 확보가 필요하다. 이 문제들이 해결되면 KDD 수준의 탁월한 기여로 인정될 수 있다.
