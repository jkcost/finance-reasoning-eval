# Paper Outline: IC Finding Paper

**Title:** "LLMs Can Detect Missing Data But Not Contradictory Data: A Metacognitive Evaluation in Financial Reasoning"

**Target Venue:** ACL/EMNLP Main or Findings, FinNLP Workshop (backup)

---

## 1. Introduction (1.5 pages)

- LLMs are increasingly deployed in financial analysis, yet evaluation focuses on accuracy — not whether models *know when they cannot answer reliably*. Frame metacognition as a safety-critical capability.
- Distinguish two metacognitive dimensions: **absence detection** (recognizing missing data) vs **conflict detection** (recognizing contradictory data). Argue these are cognitively distinct tasks, analogous to human omission detection vs inconsistency detection.
- Preview the core finding: LLMs succeed at absence detection but fail systematically at conflict detection, with performance modulated by conflict difficulty level (L1-L4) and prompt strategy. This is not a "models are bad" story — it is a "detection is conditional" story.
  <!-- 핵심: "IC 전부 실패"가 아니라 "IC 탐지는 조건부" — 어떤 조건에서 성공/실패하는가가 contribution -->

## 2. Related Work (1.5 pages)

- **Unanswerable question detection:** SQuAD 2.0 (Rajpurkar+2018), AbstentionBench (Meta), Feng+2024 Abstention F1. Gap: these test *absence* only, never *conflict*.
- **LLM calibration and honesty:** Yang+2024 Honesty Score, Cheng+2024 Balanced IDK, "Know Your Limits" survey. Gap: calibration measures confidence, not conflict-awareness.
- **Financial NLP benchmarks:** FinanceReasoning (ACL 2025), XFinBench (ACL 2025). Gap: FinanceReasoning identifies "unsolvable" as future work; we operationalize it with controlled information manipulation.
  <!-- 핵심 포지셔닝: absence 탐지는 연구됨, conflict 탐지는 공백. 우리가 최초 -->

## 3. Methodology (2.5 pages)

- **3.1 Transformation Taxonomy:** Absence types (EA-partial, EA-full, SA) as experimental control; IC Conflict Difficulty Ladder (L1-L4) as treatment.
  - L1: Obvious Typo (10x magnitude shift, rule-based)
  - L2: Unit Mismatch (unit swap, LLM-generated)
  - L3: Authority Conflict (authoritative source contradicts table, LLM-generated)
  - L4: Cross-Period Conflict (temporal aggregation inconsistency, LLM-generated)
  <!-- L5 (Implicit Ratio) excluded: generation quality uncertain, noted as future work -->
- **3.2 MC Score Metric:** MC Score = Refusal_F1 x (1 - Hallucination_Rate). Define Refusal Recall, Refusal Precision, F1. Cite Feng+2024, Yang+2024, Rajpurkar+2018 for components.
- **3.3 Transformation Pipeline:** LLM-based transformation generation + Human Review validation (4 annotators, majority vote, inter-annotator agreement reported). Describe quality control process.
- **3.4 Prompt Strategies:** 6 strategies tested — standard (control), metacognitive, contradiction_aware, self_verification, ic_fewshot (mitigation), ic_crosscheck (mitigation).

## 4. Experimental Setup (1 page)

- **Dataset:** FinanceReasoning hard set (120 problems after Human Review filtering). Justify hard-only: highest information density, most transformation targets.
- **Models:** Economic tier (GPT-4o-mini, Claude Haiku, Gemini Flash) as pilot; Balanced tier (GPT-4o, Claude Sonnet, Gemini Pro) as primary. 6 models total.
- **Phases:** Phase A (baseline accuracy on originals) -> Phase B (absence + conflict detection on transformed problems).
- **Human Baseline:** 5-10 finance professionals solve L3 IC-transformed problems (10-15 problems). Same refusal detection criteria as LLMs.
  <!-- 리뷰어가 반드시 물어볼 질문: "사람은 할 수 있는가?" — 선제적으로 답변 -->

## 5. Results (3 pages)

- **5.1 Absence vs Conflict Asymmetry (RQ1):** Absence detection rates (EA/SA) vs IC detection rates across all models. Core finding figure.
- **5.2 Model Size Effect (RQ2):** Economic vs balanced tier IC detection comparison. Does scaling help conflict detection?
- **5.3 Prompt Strategy Effect (RQ3):** 6 strategies compared on IC detection. Focus on whether ic_fewshot and ic_crosscheck mitigate the failure.
- **5.4 Conflict Difficulty Gradient (RQ4):** L1-L4 detection rates per model. Is L1 (obvious typo) detectable? Where does detection break down?
- **5.5 Human Baseline Comparison (RQ5):** Human vs LLM IC detection gap. Quantify the human-LLM performance delta.

## 6. Analysis (1.5 pages)

- **Conflict Salience Regression:** Logistic regression on IC detection (refusal=1 vs confident=0). Features: token_distance, same_paragraph, authority_marker count, magnitude_ratio. Identify which features predict detection success/failure.
- **Error Pattern Analysis:** Categorize IC failure modes — does the model (a) ignore the conflict entirely, (b) silently adopt the conflicting value, (c) acknowledge discrepancy but proceed anyway?
  <!-- authority_marker가 dominant predictor일 경우, 권위 편향 메커니즘 설명 가능 -->

## 7. Discussion (1 page)

- **Authority Bias in LLMs:** Connect IC failure to human cognitive biases (authority bias, anchoring). LLMs inherit these patterns from training data.
- **Implications for Financial AI:** Real-world financial data frequently contains contradictions (restated filings, conflicting sources). IC failure = silent risk.
- **Limitations:** Single domain (finance), L5 excluded, no open-model attention analysis (deferred), Human Review sample size.

## 8. Conclusion (0.5 pages)

- Absence detection and conflict detection are distinct metacognitive capabilities. Current LLMs possess the former but largely lack the latter.
- The Conflict Difficulty Ladder provides a systematic framework for evaluating conflict detection.
- Recommendations: conflict-aware evaluation should be standard in LLM benchmarks; mitigation strategies show promise but are insufficient.

---

## Required Figures and Tables

| ID | Type | Content | Data Source | Status |
|----|------|---------|-------------|--------|
| Fig 1 | Grouped bar / radar | Absence vs IC detection rate per model | Phase B balanced results | **NEEDED** |
| Fig 2 | Line chart | L1-L4 detection rate gradient per model | L1-L4 experiments | **NEEDED** |
| Fig 3 | Scatter + regression | Conflict salience features vs detection | Salience scorer on IC data | **NEEDED** |
| Fig 4 | Bar chart | 6 prompt strategies IC detection comparison | Strategy experiments | **NEEDED** |
| Table 1 | Main results | MC Score by model x type x strategy | Phase A+B all | **NEEDED** |
| Table 2 | IC ladder | L1-L4 detection rate, # problems, examples | L1-L4 experiments | **NEEDED** |
| Table 3 | Human baseline | Human vs LLM detection rate on L3 IC | Human study | **NEEDED** |
| Table 4 | Transformation stats | # problems per type, Human Review agreement | Human Review data | Available (after review) |
| Table 5 | Salience regression | Feature coefficients, p-values, AUC | Salience scorer | **NEEDED** |

## Required Experiments

| # | Experiment | What | Est. Cost | Dependency | Priority |
|---|-----------|------|-----------|------------|----------|
| E1 | Balanced baseline (Phase A) | 120 problems x 3 models x 4 strategies | ~$5-10 | Human Review done | P1 |
| E2 | Balanced IC L3 (Phase B) | 120 IC-L3 x 3 models x 4 strategies | ~$5-10 | E1, L3 transforms | P1 |
| E3 | IC L1-L4 implementation | Extend transformation pipeline with 4 levels | $0 (code) | None | P1 |
| E4 | IC L1-L4 generation | 120 problems x 4 levels (LLM-based) | ~$2-5 | E3, Human Review | P2 |
| E5 | L1-L4 evaluation | L1-L4 transforms x 3 balanced models x 2 strategies | ~$15-25 | E4 | P2 |
| E6 | Mitigation strategies | ic_fewshot + ic_crosscheck on balanced models | ~$5-10 | E2 (for comparison) | P2 |
| E7 | Salience scorer | Feature extraction + logistic regression | $0 (analysis) | E2 or E5 | P3 |
| E8 | Human baseline study | 5-10 participants x 10-15 L3 IC problems | $0 (manual) | L3 transforms | P3 |
| **Total** | | | **~$32-60** | | |

<!-- 기존 economic 파일럿 데이터는 pilot으로 사용, 본 실험은 balanced 중심 -->

## Writing Order Recommendation

| Order | Section | Rationale |
|-------|---------|-----------|
| 1 | **3. Methodology** | Can write now: taxonomy, MC Score, pipeline are all defined. No experiment dependency. |
| 2 | **2. Related Work** | Can write now: literature review is independent of results. |
| 3 | **4. Experimental Setup** | Can write now: dataset, models, phases are decided. |
| 4 | **5. Results (5.1-5.2)** | After E1+E2: core asymmetry finding + model size effect. |
| 5 | **6. Analysis** | After E5+E7: salience regression needs IC data. |
| 6 | **5. Results (5.3-5.5)** | After E5+E6+E8: difficulty gradient, mitigation, human baseline. |
| 7 | **1. Introduction** | Write last: needs final numbers to frame the story precisely. |
| 8 | **7. Discussion + 8. Conclusion** | After all results: connect findings to implications. |

<!-- 핵심 원칙: 데이터 의존 없는 섹션(Methodology, Related Work)을 먼저 쓰고, 실험 결과가 나오는 대로 Results 채워감 -->

---

## Key Narrative Decisions

1. **Framing:** "IC detection is conditional" (not "all models fail"). The Difficulty Ladder and mitigation results show *when* and *why* detection succeeds or fails.
2. **Absence as control, not co-protagonist.** EA/SA results establish that models *can* do metacognition, making IC failure more striking.
3. **L5 excluded.** Implicit ratio inconsistency requires uncertain generation quality. Noted as future work. L1-L4 provide a clean gradient.
4. **Attention analysis deferred.** Mechanistic explanation via attention weights requires A100+ GPU. If available, add as Section 6.3. If not, note as future work.
5. **Economic data = pilot only.** All main results tables use balanced models. Economic results appear in appendix or supplementary.
