# Peer Review: Mosaic: Multi-agent Opinion Synthesis with Adaptive Investor Calibration for Portfolio Construction

**논문 정보:** KDD'26 제출 (Anonymous), 17페이지
**리뷰 날짜:** 2026-03-13
**리뷰어:** AI-Assisted Review

---

## 요약 (Summary Statement)

본 논문은 Mosaic을 제안하며, LLM 기반 다중 에이전트 시뮬레이션을 통해 직접적인 포트폴리오 구성(end-to-end portfolio construction)을 수행하는 프레임워크이다. 핵심 메커니즘은 backward optimization으로, 과거 시장 데이터를 기반으로 이질적(heterogeneous) 에이전트 유형의 최적 분포를 동적으로 학습한다. 중국 A-share(SSE 50, CSI 300, ChiNext 100)와 미국 시장(Nasdaq 100, S&P 500)에서 9개 baseline 대비 일관된 우수 성능을 보고한다.

### 전체 평가: **Minor Revisions (소폭 수정 후 수락 가능)**

### 핵심 강점
- 개별 주식 예측이 아닌 직접 포트폴리오 구성이라는 문제 정의의 전환이 실용적이고 설득력 있음
- Backward optimization을 통한 에이전트 분포의 동적 조정은 시장 레짐 변화 대응에 효과적이며, ablation에서 핵심 기여 확인됨
- Data leakage 우려에 대한 선제적 검증(Q1 2025 미래 데이터), 코드/데이터 공개, 다양한 LLM 백본 검증 등 실험의 엄밀성이 높음

### 핵심 약점
- 에이전트 수 증가에 따른 성능 향상 메커니즘의 이론적 분석 부족
- Market Disagreement Hypothesis(MDH)에 대한 비판적 검토 없이 전적으로 의존
- LLM API 비용과 실시간 운영의 현실적 제약에 대한 논의 불충분

---

## Major Comments (주요 문제)

### M1. 에이전트 수 확장(scaling)의 이론적 근거 부족

논문의 핵심 기여 중 하나가 "에이전트 수가 증가하면 성능이 향상된다"는 것인데(Figure 3, Table 3), 이에 대한 이론적 분석이 부재하다.

- Figure 3에서 RIC가 에이전트 수에 대해 "approximately linear growth"를 보인다고 주장하나, 실제로는 64→512에서 약 0.04→0.08로 증가하며, 이것이 통계적으로 유의한 선형 관계인지 검증이 없음
- Table 3에서 1024와 1536 사이 성능이 plateau하는데, 왜 특정 지점에서 수렴하는지 설명이 없음
- "each additional agent observes a distinct subset of the market, thereby expanding the system's collective information coverage"는 직관적 설명일 뿐, 정보 이론적 또는 통계적 근거가 제시되지 않음

**제안:** (1) 에이전트 수와 정보 커버리지 간의 관계를 정량화 (예: 에이전트별 관찰 주식 겹침 비율 분석), (2) ensemble learning의 다양성-정확도 트레이드오프 프레임워크로 성능 향상 메커니즘 분석, (3) plateau 지점의 이론적 설명 (예: 주식 풀 크기 대비 포화).

### M2. Market Disagreement Hypothesis(MDH) 의존의 한계

Score aggregation (Eq. 4)이 MDH에 전적으로 의존하고 있으나:

- MDH의 원래 맥락[10, 30]은 **개인 투자자들의 실제 의견 불일치**를 전제로 한다. 그러나 Mosaic의 에이전트들은 동일한 LLM에서 생성된 것으로, 진정한 의견 불일치(genuine disagreement)가 아닌 **프롬프트 차이에 의한 인위적 다양성**이다. 이 구분이 중요한 이유는, MDH가 예측하는 수익률 패턴이 "정보 비대칭에 의한 과대평가" 메커니즘에 기반하기 때문이다.

- w/o MDH ablation (Table 4)에서 ChiNext 100의 RIC가 -3.12로 **음수**가 되는 것은, 단순 consensus만 사용하면 오히려 역효과가 남을 의미한다. 이는 MDH의 disagreement penalty가 특정 시장에서는 필수적이지만, 왜 그런지에 대한 분석이 부족하다.

- α 하이퍼파라미터가 시장별로 다르게 설정됨(SSE 50/CSI 300: 0.5, ChiNext 100: 0.2). 이는 MDH의 적용이 시장 특성에 민감하다는 것을 시사하지만, α를 동적으로 학습하지 않는 이유가 불명확하다.

**제안:** (1) LLM 에이전트의 "disagreement"가 실제 시장 disagreement와 어떤 관계인지 분석, (2) α를 backward optimization에 포함하여 동적 학습 가능성 검토, (3) MDH 이외의 대안적 aggregation 전략과 비교.

### M3. LLM 비용 및 실시간 운영 현실성

Table 7에서 일일 비용을 보고하고 있으나:

| 주식풀 | 시간 | 비용 |
|--------|------|------|
| SSE 50 | 125s | $0.679 |
| CSI 300 | 378s | $2.265 |
| ChiNext 100 | 227s | $1.192 |

- 이는 512 에이전트 기준이며, 최적 성능의 1024 에이전트에서는 비용이 약 2배로 증가할 것으로 예상
- 연간 약 250 거래일 기준, CSI 300에서 512 에이전트만으로도 **연간 약 $566** (1024 에이전트면 ~$1,132)
- 비용 자체는 높지 않으나, LLM API의 응답 안정성(rate limiting, timeout), 모델 버전 변경에 따른 성능 변화 등 운영 리스크가 논의되지 않음
- Qwen 2.5 72B Instruct 모델의 호스팅 비용(self-hosted인 경우)이 별도로 발생

**제안:** (1) LLM 버전 업데이트 시 성능 안정성 검증, (2) API 장애 시 fallback 전략, (3) self-hosted vs API 비용 비교.

### M4. 테스트 기간과 시장 환경의 제한성

- 메인 실험이 2023년 단일 연도로 제한됨
- Q1 2025 추가 검증은 data leakage를 다루기 위한 것이지, 다양한 시장 환경 검증이 아님
- 2023년 중국 A-share는 전반적 하락장이었는데, **강세장에서의 성능**은 검증되지 않음
- Table 8에서 대부분 baseline의 AR이 **음수**인 것으로 보아, 2023년이 매우 어려운 시장이었음. Mosaic의 양수 AR(+2.16%, +4.95%)은 인상적이나, 이것이 특정 시장 조건에서만 유효한 것인지 알 수 없음

**제안:** (1) 2021-2022 등 다른 시장 환경에서의 rolling test, (2) 특히 2020년 급등장, 2022년 급락장 등 극단적 시장에서의 검증.

---

## Minor Comments (부수 문제)

### m1. 에이전트 유형 설계의 임의성
- 16개 에이전트 유형(Agent A~P)의 구체적 투자 스타일이 Appendix A.3.2에 설명되어 있으나, 이 16가지 유형이 어떻게 선정되었는지 불명확. LLM이 자동 생성한 것인지, 수동 설계인지? 유형 수(𝑛_type=16)의 선택 근거는?

### m2. Simulated Annealing 최적화의 선택 근거
- Backward optimization에 simulated annealing을 사용한 이유가 불충분. simplex 위의 최적화 문제에 gradient-based method나 Bayesian optimization 등 대안과의 비교가 없음.

### m3. 포트폴리오 구성 상세
- Top-k 전략으로 상위 20% 주식을 equal weight로 보유한다고 했으나 (A.6), 이것이 메인 실험(Table 1의 IC/RIC 결과)에도 동일하게 적용된 것인지 명확하지 않음. IC/RIC는 signal과 return의 상관이므로, 실제 포트폴리오 구성과는 다른 평가일 수 있음.

### m4. 거래 비용 설정
- Backtesting에서 round-trip 거래 비용 0.1%를 사용했으나, 중국 A-share의 인지세(stamp tax, 매도 시 0.05%) + 수수료 + slippage를 고려하면 0.1%는 다소 낮을 수 있음. 특히 ChiNext의 소형주는 유동성이 낮아 slippage가 클 수 있음.

### m5. LLM의 확률적 출력(stochasticity)
- Section A.3.5에서 두 번의 독립 추론 결과가 유사하다고 보여주지만, temperature 설정이 명시되지 않음. Temperature가 0이면 결정적 출력이므로 당연히 일관적이고, 높으면 변동이 커질 것. 이 설정이 실험 전체에 영향을 미침.

### m6. Candidate Stock Pool의 고정 할당
- 각 에이전트에 정적으로 할당된 주식 풀(Pool(i,k))이 |Pool| = 𝑛_sel = 20 또는 30으로 설정됨. Mosaic(DU) 실험에서 동적 업데이트와 차이가 없다고 했지만, 이는 풀 크기가 충분히 작아서일 수 있음. 더 큰 주식 유니버스(예: 전체 A-share ~5000종목)에서도 유효한지 확인 필요.

### m7. 논문 제목의 "Mosaic" 약어
- MOSAIC = "Multi-agent Opinion Synthesis with Adaptive Investor Calibration"으로 제시되었으나, 약어가 자연스럽지 않음 (O, I, C가 각각 Opinion, Investor, Calibration에서 왔지만 S와 A의 매핑이 불명확). 사소한 문제이지만 정리가 필요.

### m8. 재현성 관련
- 코드와 데이터 공개를 약속한 점은 매우 긍정적(anonymous 링크 제공). 다만 LLM 기반 시스템이므로, 동일한 모델 버전과 API 조건에서만 재현 가능. 모델 가중치 snapshot을 제공한다고 했는데, 이는 Qwen의 경우에 해당하며 GPT-OSS-120B는 API 종속적.

### m9. 표기 일관성
- Table 1에서 "RICIR"과 "ICIR"이 사용되었으나, 본문에서는 "Rank Information Coefficient Information Ratio"로 명시. 약어가 직관적이지 않으므로, "RIC-IR"처럼 하이픈 사용을 권장.

### m10. Limitations 절의 위치
- Limitations가 Appendix A.1에 배치됨. KDD 페이지 제한 때문으로 보이나, 본문 Conclusion에 핵심 한계를 간략히 언급하는 것이 학술적 관행에 부합.

---

## 저자에 대한 질문 (Questions for Authors)

1. **에이전트 유형 설계:** 16개 에이전트 유형은 어떻게 결정되었는가? 유형 수를 8개 또는 32개로 변경하면 성능은 어떻게 변하는가?

2. **α의 동적 학습:** Score aggregation의 α를 시장별로 수동 설정하는 대신, backward optimization에 포함시켜 동적으로 학습하는 것을 시도했는가? 시도했다면 결과는?

3. **LLM temperature:** 에이전트의 LLM 추론 시 temperature 설정은? 이 값이 에이전트 간 다양성과 성능에 미치는 영향은?

4. **강세장 성능:** 2023년은 전반적 하락장이었는데, 2019-2020년 같은 강세장에서 Mosaic의 backward optimization이 동일하게 효과적인가?

5. **MDH vs. 단순 다수결:** Disagreement penalty 없이 단순 가중 다수결(majority voting)만 사용한 경우와의 비교 결과는?

6. **Macro data 업데이트 빈도:** 주 1회 macro data 업데이트가 일일 업데이트 대비 성능 차이는?

7. **GPT-OSS-120B 모델:** 이 모델은 2026년 기준으로 비교적 최신 모델인데, 더 작은 모델(예: 7B, 14B급)에서의 성능은?

---

## 보고 기준 체크리스트 (Reporting Standards)

| 항목 | 상태 | 비고 |
|------|------|------|
| 데이터 기간 및 출처 | O | 2023 (main), Q1 2025 (leakage test) |
| 훈련/검증/테스트 분할 | O | Online learning 방식, look-back window 활용 |
| 거래 비용 포함 | O | 0.1% round-trip |
| Look-ahead bias 방지 | O | 시계열 순서 준수, 미래 데이터 검증 별도 수행 |
| 복수 실행 평균 | △ | Case study에서 2회 inference 일관성 확인, 메인 결과의 복수 실행 보고 없음 |
| 표준편차 보고 | X | 없음 |
| 통계적 유의성 검정 | X | 없음 |
| 코드 공개 | O | Anonymous GitHub 링크 제공 |
| 턴오버/거래 빈도 | △ | 주간 리밸런싱 언급, 턴오버율 미보고 |
| Feature 목록 | O | Appendix A.2에 상세 기술 |
| API 비용 | O | Table 7에 보고 |
| LLM 프롬프트 | O | Appendix A.3에 전체 프롬프트 공개 |

---

## 최종 평가

Mosaic은 LLM 기반 다중 에이전트 시스템을 포트폴리오 구성에 적용한 실용적이고 잘 설계된 연구이다. 특히 다음 세 가지 점에서 높이 평가한다:

1. **실용적 문제 정의:** 개별 주식 예측이 아닌 직접 포트폴리오 구성으로의 전환은 금융 실무와의 간극을 줄임
2. **철저한 실험 설계:** Data leakage 검증, 다중 LLM 백본 검증, 다시장 검증, ablation study, case study 등 포괄적
3. **재현성 노력:** 코드, 데이터, 학습 스냅샷 공개 약속

주요 개선 사항은 에이전트 확장의 이론적 근거, MDH 의존의 정당화, 그리고 다양한 시장 환경에서의 추가 검증이다. 이러한 사항들은 논문의 근본적 기여를 훼손하지 않으며, 수정을 통해 충분히 보완 가능하다.

전반적으로 KDD 수준의 기여를 제공하는 논문으로 판단된다.

---

*이 리뷰는 AI 보조 도구를 활용하여 작성되었으며, 최종 판단은 전문 리뷰어의 검토가 필요합니다.*
