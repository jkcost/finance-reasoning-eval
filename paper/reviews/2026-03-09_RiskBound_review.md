# Peer Review: RiskBound: Risk-Aware Boundary-Guided Portfolio Optimization via Action Space Reshaping

**논문 정보:**
- 제출 학회: SIGKDD'26 (32nd ACM SIGKDD Conference)
- 저자: Anonymous (더블 블라인드)
- 페이지: 11페이지 (본문 8 + 부록 3)
- 리뷰 일자: 2026-03-09

---

## 1. 요약 (Summary Statement)

본 논문은 강화학습(RL) 기반 포트폴리오 최적화에서 시변(time-varying) 자산별 리스크를 명시적으로 행동 공간 제약으로 인코딩하는 프레임워크 **RiskBound**를 제안한다. (1) Risk-Aware Boundary Generator (RABG)로 자산별 하방 리스크 점수를 추정하여 동적 상한 바운드를 생성하고, (2) Fast Projection with Surrogate Gradients (FPSG)로 box-simplex 제약을 효율적으로 적용하며, (3) 아키텍처 비의존적 어댑터로 설계하여 다양한 정책 백본에 통합 가능하도록 했다. 미국, 중국, 영국 3개 주식 시장에서 ARR 및 ASR 기준 최고 성능을 달성했다.

### 전반적 평가: **Minor Revisions**

### 핵심 강점
- 리스크 제어를 RL 목적함수에서 분리하여 행동 공간 제약으로 인코딩한 설계가 개념적으로 깔끔하고 실용적
- Surrogate gradient를 통한 end-to-end 학습과 fast projection의 결합이 기술적으로 견고
- 3개 시장 데이터셋, 다양한 baseline, ablation study 등 실험 설계가 체계적

### 핵심 약점
- 단일 정책 백본(ALSTM+DDPG)에서만 검증하여 아키텍처 비의존성 주장의 실증 근거 부족
- 통계적 유의성 검증(신뢰구간, 다중 시드 실험) 부재
- 비교 baseline이 제한적이며, 최신 RL 포트폴리오 기법 누락

---

## 2. Major Comments

### M1. 아키텍처 비의존성(Architecture-agnostic) 주장의 실증 부족

논문의 3대 핵심 기여 중 하나인 "architecture-agnostic adapter design" (I3)이 ALSTM+DDPG 단일 조합에서만 실험되었다. 이는 주장의 핵심을 뒷받침하기에 불충분하다.

**권고:** 최소 2-3개의 다른 정책 백본(예: Transformer 기반, CNN 기반) 및/또는 다른 RL 알고리즘(예: TD3, SAC의 deterministic variant)과의 조합 실험을 추가하여 아키텍처 비의존성을 실증적으로 보여야 한다.

### M2. 통계적 유의성 검증 부재

Table 2의 모든 성능 수치가 단일 시드(seed=0)에 기반하며, 표준편차, 신뢰구간, 또는 통계 검정이 전혀 보고되지 않는다. 금융 데이터의 높은 변동성을 고려하면, 단일 시드 결과로 "superior performance"를 주장하는 것은 과도하다.

**권고:** 최소 5개 이상의 랜덤 시드로 실험을 반복하고, 평균 및 표준편차(또는 신뢰구간)를 보고해야 한다. 주요 baseline 대비 paired t-test 또는 Wilcoxon signed-rank test 결과를 포함하라.

### M3. Baseline 비교의 제한성

RL baseline이 DeepTrader(2021)와 MetaTrader(2022)로 한정되어 있어, 2023-2025년 최신 연구와의 비교가 부족하다. 특히 Constrained RL 기반 포트폴리오 기법(예: Winkel et al. [28, 29])은 관련 연구에서 언급하면서도 baseline으로 비교하지 않았다.

**권고:** (1) [28]의 autoregressive policy optimization을 직접 비교 baseline에 포함하거나, (2) 비교하지 않은 이유를 명확히 설명하라(예: 해당 방법이 long-only 포트폴리오를 지원하지 않는 등).

### M4. RABG 사전학습의 데이터 누수(data leakage) 우려

Section 3.3.2에서 RABG의 학습 타겟이 "subsequent holding period"의 실현 하방 변동성(σ_t,i)이다. Appendix F에서 "same data splits"를 사용한다고 언급하지만, RABG가 미래 데이터(holding period 수익률)를 사용하여 학습되는 구조는 잠재적 look-ahead bias를 야기할 수 있다.

**권고:** (1) RABG 학습 시 사용되는 타겟의 시간적 범위를 더 명확히 기술하고, (2) RABG가 테스트 기간의 미래 정보에 접근하지 않음을 명시적으로 확인하라. Walk-forward validation 방식의 적용 여부를 설명하라.

---

## 3. Minor Comments

### m1. RiskBound-NO의 높은 ARR에 대한 설명 부족 (Table 6)

Ablation에서 RiskBound-NO(RABG+FPSG 모두 제거)가 ARR 0.255로 full RiskBound(0.139)보다 높다. 이는 리스크 제어가 절대 수익률을 희생한다는 의미인데, Discussion에서 "less favorable return-risk balance"로 간략히 언급만 하고 있다. 이 현상에 대한 더 깊은 분석이 필요하다.

**권고:** RiskBound-NO의 높은 ARR이 어떤 리스크 프로파일(MDD, 최대 손실 등)을 동반하는지 구체적으로 보여주고, 왜 risk-adjusted 관점에서 RiskBound가 더 나은 선택인지 설득력 있게 논의하라.

### m2. Long-only 제약의 한계 논의 부재

행동 공간이 simplex(Δ^{N-1})로 정의되어 long-only 전략만 다룬다. Short selling이 가능한 시장 환경에서의 확장 가능성을 Discussion이나 Conclusion에서 언급해야 한다.

### m3. 하이퍼파라미터 민감도 분석 부족

Table 8에서 c, α, β의 탐색 범위와 최적값만 보고하고, 이 값들의 변화에 따른 성능 민감도를 보여주지 않는다.

**권고:** 주요 하이퍼파라미터(특히 global scaling factor c)에 대한 민감도 분석 그래프를 추가하라.

### m4. 거래비용 모델의 단순성

0.1% 고정 거래비용(turnover-based)은 실제 시장의 슬리피지, 시장 충격(market impact), bid-ask spread 등을 반영하지 못한다. 128-239개 자산의 대규모 포트폴리오에서는 이 점이 더욱 중요하다.

**권고:** Limitation에서 거래비용 모델의 단순성을 명시적으로 언급하라.

### m5. Figure 3의 해석 보완 필요

Figure 3(b)는 COVID-19 시기 BA의 upper bound가 감소함을 보여주지만, 이 bound 변화가 실제 포트폴리오 배분과 수익률에 미친 영향을 정량적으로 보여주지 않는다.

**권고:** 해당 시기의 RiskBound vs. baseline의 실제 배분 비중 및 누적 수익률 비교를 추가하라.

### m6. Reward function 표기 오류 가능성

Eq. 3.2의 V_t = (μ_t · G_t) · V_{t-1}에서 V_0 = 1 정의가 같은 줄에 이어져 있어 가독성이 떨어진다.

### m7. Downside volatility vs. CVaR/ES

리스크 측정에 downside volatility만 사용하는데, CVaR(Conditional Value at Risk)나 Expected Shortfall 같은 보다 표준적인 tail risk 측정치와의 비교 또는 대안적 타겟으로서의 논의가 있으면 좋겠다.

### m8. 코드/데이터 가용성

GitHub 링크가 제공되어 있으나, 리뷰 시점에 접근 가능 여부를 확인할 수 없다. 데이터셋의 경우 출처(TradeMaster, SNU DataLab)는 명시되어 있으나, 전처리 파이프라인의 재현 가능성이 불명확하다.

---

## 4. 방법론 및 통계적 엄밀성 평가

### 실험 설계
- **장점:** Train/Val/Test 분리 명확, 단일 시장(US) validation으로 하이퍼파라미터 선정 후 전 시장 적용 (시장 간 과적합 방지)
- **장점:** 동일 데이터 분할, 특성 세트, 거래비용 조건 하 공정 비교
- **약점:** 단일 시드, 단일 백본 (위 M1, M2 참조)

### 수학적 엄밀성
- FPSG의 shift-and-clip projection에 대한 KKT 기반 증명(Lemma 1, Appendix D)은 수학적으로 타당
- Surrogate Jacobian의 simplex-consistent 조건 유도(Eq. 7-10)가 명확하고 정확
- Box-simplex feasibility 보장에 대한 논증이 충분

### 계산 효율성
- Table 5의 projection latency 비교(4-5ms vs. 143-246ms)는 인상적이나, 전체 학습 시간 비교도 있으면 유용

---

## 5. 재현성 및 투명성 평가

| 항목 | 평가 |
|------|------|
| 코드 공개 | 링크 제공 (접근 확인 불가) |
| 데이터 출처 | 명시됨 (TradeMaster, SNU DataLab) |
| 하이퍼파라미터 | 완전히 보고됨 (Table 8, 9) |
| 학습 절차 | Algorithm 1, 2로 상세 기술 |
| 랜덤 시드 | 단일 시드(seed=0)만 사용 |
| 하드웨어 환경 | Xeon Silver 4214 + 4x RTX 3090 명시 |

---

## 6. 글쓰기 품질 평가

- 전반적으로 명확하고 논리적인 구조
- 수식 표기가 일관적이며 Appendix A의 기호 정리표가 유용
- Figure 1의 개념적 일러스트레이션이 핵심 아이디어를 효과적으로 전달
- Figure 2의 아키텍처 다이어그램이 전체 파이프라인을 잘 보여줌
- Related Works가 체계적으로 분류(architecture-level, reward-level, regime-aware)되어 있음

---

## 7. 저자에게 드리는 질문

1. **RABG가 미래 정보를 사용하지 않음을 어떻게 보장하는가?** RABG 사전학습 시 holding period의 실현 수익률을 타겟으로 사용하는데, 테스트 시에는 이 정보 없이 추론이 가능한가? (즉, RABG는 inference 시 과거 데이터만으로 리스크 점수를 예측하는 것인가?)

2. **RiskBound-NO가 ARR 0.255를 달성하면서 ASR은 0.615에 불과한 이유는?** 이 variant의 MDD(Maximum Drawdown)는 얼마인가? 극단적 집중 투자가 발생하는가?

3. **RABG의 risk score가 시장 급변 시(예: flash crash) 얼마나 빠르게 반응하는가?** Lookback window L=40은 약 2개월인데, 급격한 리스크 변화에 충분히 빠른 반응이 가능한가?

4. **c=1.5의 의미는 무엇인가?** 즉, 모든 자산의 upper bound 합이 1.5라는 것은 상위 바운드가 포트폴리오 전체 예산의 150%라는 의미인데, 이것이 실질적으로 얼마나 binding한 제약인가?

5. **다른 RL 알고리즘(TD3, SAC 등)과의 통합 시 추가적인 수정이 필요한가?** 특히 stochastic policy gradient 기반 알고리즘과의 호환성은?

---

## 8. 최종 평가 요약

| 평가 항목 | 점수 (1-5) |
|-----------|:---------:|
| 참신성 (Novelty) | 4 |
| 기술적 건전성 (Soundness) | 3.5 |
| 실험 설계 (Experimental Design) | 3 |
| 명확성 (Clarity) | 4.5 |
| 재현성 (Reproducibility) | 3.5 |
| 의의 (Significance) | 3.5 |
| **종합** | **3.5 / 5** |

**최종 권고: Minor Revisions (조건부 수락)**

핵심 아이디어(리스크를 행동 공간 제약으로 인코딩)는 참신하고 기술적으로 잘 실행되었다. 그러나 (1) 다중 시드 실험을 통한 통계적 유의성 확보, (2) 최소 1개 추가 정책 백본에서의 아키텍처 비의존성 검증, (3) RiskBound-NO의 높은 ARR에 대한 심층 분석이 반드시 보완되어야 한다. 이 세 가지가 해결되면 SIGKDD 수준의 기여를 충족한다고 판단한다.
