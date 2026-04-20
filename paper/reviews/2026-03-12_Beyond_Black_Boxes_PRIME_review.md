# Peer Review: Beyond Black Boxes: An Energy-Based Unified Framework for Interpretable Stock Selection

**논문 정보:** KDD'26 제출 (Anonymous), 27페이지
**리뷰 날짜:** 2026-03-12
**리뷰어:** AI-Assisted Review

---

## 요약 (Summary Statement)

본 논문은 PRIME (Potential Robust Integrated Macro Energy)을 제안하며, 에너지 기반 모델링(EBM)과 게임 이론을 결합하여 해석 가능한 주식 선택 프레임워크를 구축한다. 주식 가치를 bullish momentum, bearish resistance, frictional dissipation으로 분해하고, 거시경제 조절 메커니즘과 Nash 균형 기반 에너지 집계, 위험 감시 모듈을 통합한다. S&P 500과 CSI 500에서 기존 SOTA 대비 약 10% 성능 향상을 보고한다.

### 전체 평가: **Major Revisions (대폭 수정 후 재검토)**

### 핵심 강점
- 에너지 기반 모델을 주식 선택에 적용한 창의적인 프레임워크 설계로, 물리학적 직관과 금융 해석 가능성을 잘 연결함
- 방대한 이론적 분석 (partition-free learning, ranking consistency, convergence guarantees 등)을 포함하여 학술적 깊이가 뛰어남
- 포괄적인 실험 설계: ablation study, robustness analysis, regime-dependent analysis, computational efficiency 비교 등

### 핵심 약점
- 테스트 기간이 2024년 단 1년으로, 결과의 통계적 신뢰성과 일반화 가능성이 심각하게 제한됨
- "약 10% 개선"이라는 주장이 부정확하며, 실제 Table 1 결과는 메트릭마다 큰 차이를 보임
- 해석 가능성(interpretability)이 논문의 핵심 주장이지만, 실제 해석 가능성에 대한 체계적 평가나 사용자 연구가 전무함

---

## Major Comments (주요 문제)

### M1. 테스트 기간의 불충분성 (실험 설계)

테스트 기간이 2024년 단 1년(out-of-sample)으로 설정되어 있다. 금융 시장의 비정상성(non-stationarity)을 핵심 문제로 제기하면서, 정작 테스트는 단일 연도의 단일 시장 조건에서만 수행되었다. 2024년은 미국 시장 기준 강세장이었으므로, 이 결과가 약세장이나 횡보장에서도 유지되는지 알 수 없다.

**제안:** Rolling window backtesting (예: 2018-2024, 매년 재훈련)을 도입하여 다양한 시장 조건에서의 성능을 보고해야 한다. 최소 3-5년의 테스트 기간이 필요하다.

### M2. 성능 주장의 부정확성

Abstract에서 "approximately 10% improvement over state-of-the-art baselines"라고 주장하나, Table 1을 보면:
- S&P 500 ARR: PRIME 45.84% vs StockFormer 40.22% (약 14% 개선)
- CSI 500 ARR: PRIME 62.86% vs AlphaGAT 60.43% (약 4% 개선)
- S&P 500 SR: PRIME 2.304 vs GPT4TS 2.049 (약 12% 개선)
- CSI 500 SR: PRIME 2.539 vs StockFormer 3.211 (**23% 열위**)

특히 **CSI 500에서 Sharpe Ratio가 StockFormer(3.211) 대비 크게 낮다(2.539)**는 점은 논의되지 않았다. SR은 위험 조정 수익률의 핵심 지표로, 이 결과를 무시하는 것은 선택적 보고(selective reporting)에 해당한다.

**제안:** 성능 주장을 정확하게 수정하고, 각 메트릭별 상대 성능을 정직하게 논의해야 한다. 특히 CSI 500에서 StockFormer 대비 SR 열위를 설명해야 한다.

### M3. 해석 가능성 평가의 부재

논문 제목이 "Beyond Black Boxes"이고 해석 가능성이 핵심 기여 중 하나로 주장되나, 해석 가능성에 대한 체계적 평가가 없다:
- 학습된 direction parameters가 실제로 도메인 전문가에게 의미 있는 해석을 제공하는지 검증 없음
- 기존 XAI 방법(SHAP, LIME 등)과의 비교 없음
- 실제 금융 실무자(quant analyst)에 의한 사용자 연구(user study) 없음
- "inherently interpretable"이라는 주장에 대한 정량적 근거 없음

에너지 분해가 bull/bear/friction으로 이루어진다는 것 자체는 구조적 해석 가능성을 제공하지만, 이것이 실제로 올바른 시장 동학을 포착하는지는 별개의 문제이다.

**제안:** (1) 학습된 feature direction parameters를 금융 도메인 지식과 비교 분석하는 사례 연구, (2) 에너지 분해 결과가 실제 시장 이벤트와 어떻게 대응하는지 시계열 분석, (3) 가능하면 사용자 연구 추가.

### M4. 통계적 유의성 검증 부족

Table 1에서 5회 독립 실행의 평균 결과를 보고하지만:
- 표준편차가 Table 1에는 보고되지 않음 (Table 2에만 존재)
- baseline 모델들과의 통계적 유의성 검정(t-test, Wilcoxon signed-rank test 등)이 없음
- Table 3 ablation에서 일부 variant의 표준편차가 매우 큼 (w/o bear: 20.98%, w/o mod: 18.65%)

금융 시계열에서 5회 실행의 평균만으로 우월성을 주장하는 것은 통계적으로 불충분하다.

**제안:** (1) Table 1에 표준편차 추가, (2) 주요 baseline과의 paired statistical test 수행, (3) 특히 StockFormer와 AlphaGAT 등 강력한 baseline과의 차이가 통계적으로 유의한지 검증.

### M5. 물리학 비유의 과도한 사용과 정당성 문제

논문은 금융 시장을 물리적 에너지 시스템으로 비유하며 열역학 제2법칙을 언급하지만, 이 비유의 한계에 대한 논의가 부족하다:
- 금융 시장은 참가자들의 전략적 행동에 의해 구동되며, 물리적 입자와 근본적으로 다름
- "stocks spontaneously tend toward low-energy states following the Second Law of Thermodynamics" (line 350-351) — 주식이 열역학 법칙을 따른다는 것은 비유일 뿐 과학적 주장이 아님
- Appendix B.2에서 "this line of argument does not rely on any analogy to physical systems"라고 밝히면서도, 본문에서는 물리적 비유를 핵심 설명 도구로 사용하는 모순

**제안:** 본문에서 물리학적 비유의 한계를 명시적으로 언급하고, 프레임워크의 정당성이 물리적 비유가 아닌 정보 이론(maximum entropy)에 근거함을 더 명확히 해야 한다.

### M6. Risk Guardian Module의 설계 문제

Risk Guardian Module이 LightGBM 기반의 별도 모델로 구현되어 있는데:
- 에너지 모델과 독립적으로 학습되므로 end-to-end 최적화가 아님
- Hard example mining의 threshold (𝜌=80%)와 weighting coefficient (𝜅=3)의 선택에 대한 sensitivity analysis 없음
- crash label의 정의가 불명확 — 논문에서 "crash"의 정확한 기준 (예: -X% 수익률)이 명시되지 않음
- Table 3에서 w/o guardian의 ARR(42.88%)이 여전히 높아, guardian의 기여가 주로 MDD 감소에 있는지, ARR 향상에 있는지 불분명

**제안:** (1) crash label의 구체적 정의 명시, (2) 𝜌와 𝜅의 sensitivity analysis 추가, (3) guardian 유무에 따른 MDD, SR 비교를 별도로 상세 보고.

### M7. 거래 비용과 실제 적용 가능성

- 거래 비용을 0.3% (commission + slippage + stamp tax)로 설정했으나, 이는 현실적인가? 특히 S&P 500에서의 거래 비용과 CSI 500에서의 거래 비용은 크게 다를 수 있음
- Top-K 포트폴리오의 rebalancing 주기가 명시되지 않음 — 일별? 주별?
- 포트폴리오 내 종목 가중치 방법(equal-weight? score-proportional?)이 불명확
- 실제 시장에서의 유동성 제약(liquidity constraints)이 고려되지 않음

**제안:** (1) rebalancing 주기 명시, (2) 포트폴리오 가중치 방법 설명, (3) 거래 비용 sensitivity analysis (0.1%~0.5% 범위), (4) 턴오버율(turnover ratio) 보고.

---

## Minor Comments (부수 문제)

### m1. 표기법 일관성
- Friction energy가 본문에서는 𝐸_fric으로 시작하지만, 수식에서는 𝐸_heat로 사용됨 (Eq. 13 등). "heat"이라는 명칭이 friction과 어떻게 관련되는지 설명이 부족하다. 열(heat)과 마찰(friction)이 물리학에서 관련이 있지만, 금융 맥락에서는 혼란스러울 수 있다.

### m2. Figure 1의 가독성
- Figure 1의 에너지 랜드스케이프가 논문의 핵심 직관을 전달하는 중요한 그림이지만, 3D 표현이 2D 페이지에서 정보 손실이 크다. 등고선(contour plot)이 더 효과적일 수 있다.

### m3. Related Works 구성
- Section 2.1과 2.2가 다소 장황하다. KDD 논문의 지면 제약을 고려하면, 핵심 차별화에 더 집중하는 것이 좋다.

### m4. Curriculum Learning
- 학습 과정에서 3단계 curriculum (10, 30, 20 epochs)이 사용되었으나, 각 단계의 구체적 내용과 설계 근거가 본문에서 설명되지 않음.

### m5. Feature Group 분류 기준
- Bull, bear, friction features의 구체적 분류 기준과 실제 사용된 feature 목록이 명시되지 않음. 예를 들어, "momentum and capital inflow"가 bull features라고 했으나, 구체적으로 어떤 기술적 지표(technical indicators)가 포함되는지 불명확.

### m6. Macro Features의 시차 문제
- CPI, PMI 등 거시경제 지표는 실시간이 아닌 월별/분기별로 발표되며, 발표 시차(publication lag)가 있다. 이를 어떻게 처리했는지 설명이 필요하다. Look-ahead bias가 없는지 확인해야 한다.

### m7. Table 2의 Noise 3% 결과 이상
- Noise perturbation 실험에서 Noise 3%의 결과(62.86%, SR 2.539)가 Noise 0%(59.67%, SR 1.971)보다 **오히려 더 좋다**. 이는 매우 비직관적이며, 노이즈가 일종의 regularization 역할을 하는 것인지, 단순한 통계적 변동인지 논의가 필요하다.

### m8. AlphaGAT 및 StockFormer 결과 불일치
- CSI 500에서 StockFormer의 SR이 3.211로 PRIME(2.539)보다 월등히 높지만, ARR은 54.27%로 PRIME(62.86%)보다 낮다. 이는 StockFormer가 낮은 volatility(0.169)를 달성하기 때문인데, 이러한 trade-off에 대한 논의가 있어야 한다.

### m9. 코드/데이터 공개
- 재현성을 위한 코드, 데이터, 또는 모델 체크포인트 공개 계획이 언급되지 않았다.

### m10. 수식 표기
- Eq. (8)에서 batch normalization 기반 에너지 정규화의 𝜖은 정의되었으나 구체적 값이 본문에 없음 (Appendix에서 확인 가능할 수 있으나 본문에도 명시 필요).

---

## 저자에 대한 질문 (Questions for Authors)

1. **테스트 기간 확장:** 2024년 이전 기간(예: 2020-2023 rolling test)에서의 성능은 어떠한가? 특히 2020년 COVID 시장 폭락, 2022년 금리 인상기에서의 성능이 궁금하다.

2. **Crash label 정의:** Risk Guardian Module에서 사용하는 crash label(𝑦_crash)의 구체적 정의는 무엇인가? 어떤 기준(임계값)으로 crash를 정의했는가?

3. **Noise 3% 결과:** Table 2에서 Noise 3%가 Noise 0%보다 더 좋은 성능을 보이는 이유는 무엇인가?

4. **Feature 분류:** Bull, bear, friction feature groups에 구체적으로 어떤 features가 포함되는가? 이 분류는 사전에 수동으로 정했는가?

5. **Macro data lag:** 거시경제 데이터의 발표 시차를 어떻게 처리했는가? T+1 이상의 시차를 적용했는가?

6. **CSI 500 SR:** StockFormer 대비 CSI 500에서 SR이 열위인 이유에 대한 분석은?

7. **Rebalancing:** 포트폴리오 rebalancing 주기와 가중치 결정 방법은?

---

## 보고 기준 체크리스트 (Reporting Standards)

| 항목 | 상태 | 비고 |
|------|------|------|
| 데이터 기간 및 출처 | O | 2014-2024, 다만 출처 미명시 |
| 훈련/검증/테스트 분할 | O | 2014-2022/2023/2024 |
| 거래 비용 포함 | O | 0.3% friction |
| Look-ahead bias 방지 | △ | 시계열 분할은 적용, 매크로 데이터 시차 불분명 |
| 복수 실행 평균 | O | 5회 실행 |
| 표준편차 보고 | △ | Table 2에만, Table 1에는 없음 |
| 통계적 유의성 검정 | X | 없음 |
| 코드 공개 | X | 언급 없음 |
| 턴오버/거래 빈도 | X | 보고 없음 |
| Feature 목록 | △ | 그룹만 기술, 구체적 목록 없음 |

---

## 최종 평가

PRIME은 에너지 기반 모델을 금융 주식 선택에 창의적으로 적용한 의미 있는 연구이다. 물리학적 직관과 정보 이론적 기반을 결합한 프레임워크 설계는 학술적으로 흥미롭고, 이론적 분석의 깊이도 인상적이다. 특히 partition-free learning과 Nash equilibrium 기반 에너지 집계는 기술적으로 견고한 기여이다.

그러나 실험적 검증의 엄밀성에서 상당한 개선이 필요하다. 1년 테스트 기간, 통계적 유의성 미검증, 해석 가능성에 대한 정량적 평가 부재, 그리고 일부 결과의 선택적 보고는 논문의 신뢰성을 약화시킨다. 또한 금융 실무 적용 관점에서 turnover, rebalancing, 유동성 제약 등 핵심 정보가 누락되어 있다.

이러한 문제들이 해결되면 KDD에 충분히 기여할 수 있는 논문이 될 것으로 판단된다.

---

*이 리뷰는 AI 보조 도구를 활용하여 작성되었으며, 최종 판단은 전문 리뷰어의 검토가 필요합니다.*
