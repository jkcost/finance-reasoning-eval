# Research Framework: Metacognitive Evaluation via Controlled Information Manipulation

## 1. Academic References & Metric Foundations

### 1.1 Core Metrics — References

| Metric | Definition | Reference | Status |
|--------|-----------|-----------|--------|
| **Refusal Recall** | correctly_refused / total_unsolvable | Wen+2024 (EMNLP Findings) | Correct |
| **Refusal Precision** | correctly_refused / total_refused | Wen+2024 (EMNLP Findings) | Correct |
| **Refusal F1** | 2 x RP x RR / (RP + RR) | Standard F1 | Correct |
| **Over-Conservatism** | solvable_refused / total_solvable | Yang+2024 (NeurIPS) | Correct |
| **Hallucination Rate** | hallucinated / answering_responses | Inspired by FActScore (Min+2023, EMNLP) | Simplified |
| **MC Score** | Refusal_F1 x (1 - HR) | **Novel composite** | Our contribution |

> **Citation Fix Required**: Code references "Feng+2024" but correct source is **Wen et al. (2024/2025)**.
> Feng+2024b is ElectionQA23 dataset paper, not the abstention metric paper.

### 1.2 Full Reference List

**Abstention / Refusal Metrics**:
1. Wen, B. et al. "Characterizing LLM Abstention Behavior in Science QA with Context Perturbations" — *Findings of EMNLP 2024* — Context perturbation으로 abstention 행동 측정. **우리 방법론과 가장 직접적 관련**.
2. Wen, B. et al. "Know Your Limits: A Survey of Abstention in Large Language Models" — *TACL 2025* — Abstention 메트릭 체계화 서베이. P/R/F1/Acc, coverage, AUROC 등.
3. Kirichenko, P., Ibrahim, M., Chaudhuri, K., Bell, S.J. "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions" — *arXiv 2506.09038, 2025* — 35,000+ unanswerable queries, 6 abstention scenarios.

**Honesty & IDK**:
4. Yang, Y. et al. "Alignment for Honesty" — *NeurIPS 2024* — Prudence + Over-Conservatism → Honesty Score.
5. Cheng, Q. et al. "Can AI Assistants Know What They Don't Know?" — *ICML 2024* — Knowledge quadrant (Ik-Ik, Idk-Ik, etc.), model-specific IDK dataset.
6. Amayuelas, A. et al. "Knowledge of Knowledge: Exploring Known-Unknowns" — *Findings of ACL 2024* — KUQ dataset, uncertainty source taxonomy.

**Self-Knowledge & Calibration**:
7. Kadavath, S. et al. "Language Models (Mostly) Know What They Know" — *Anthropic, arXiv 2207.05221, 2022* — P(IK) probe, calibration curves.
8. Yin, Z. et al. "Do Large Language Models Know What They Don't Know?" — *Findings of ACL 2023* — Self-knowledge evaluation.

**Domain Metacognition**:
9. Griot, M., Hemptinne, C., Vanderdonckt, J., Yuksel, D. "Large Language Models lack essential metacognition for reliable medical reasoning" — *Nature Communications 16, 642, 2025* — MetaMedQA, 의료 LLM 메타인지 평가. **우리 IC 결과와 일관**.

**Foundational**:
10. Rajpurkar, P. et al. "Know What You Don't Know: Unanswerable Questions for SQuAD" — *ACL 2018* — SQuAD 2.0, unanswerable QA evaluation 원형.
11. Min, S. et al. "FActScore: Fine-grained Atomic Evaluation of Factual Precision" — *EMNLP 2023* — Atomic hallucination measurement.

**Base Dataset**:
12. "FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging" — *ACL 2025* — 우리 연구의 기반 데이터셋. 정보 부족 상황에서의 메타인지를 future work로 남겨둠.

### 1.3 Novelty Positioning

| Aspect | Prior Work | Ours |
|--------|-----------|------|
| Domain | General/Science QA (Wen+2024), Medical (Griot+2025) | **Financial numerical reasoning** |
| Perturbation | 3 types: removal/replace/augment (Wen+2024) | **5 types**: EA-p, EA-f, SA, IC, TA |
| Metacognition dimension | Absence detection only (SQuAD 2.0) | **Absence + Conflict detection** |
| Metric | Abstention F1 (Wen), Honesty Score (Yang) | **MC Score** = F1 x (1-HR) |
| Unanswerable generation | Human-written (SQuAD) or model-generated (Cheng) | **Rule-based transformation** of solvable → unsolvable |

> **Gap filled**: 금융 도메인에서의 LLM 메타인지 평가 논문은 **발견되지 않음**. 이것이 우리 연구의 핵심 novelty.

---

## 2. Hypotheses

### H1: Model Size Effect

**Background**: Balanced 모델 (gpt-4o, claude-sonnet-4, gemini-2.5-pro) vs Economic 모델 (gpt-4o-mini, claude-haiku-4, gemini-2.5-flash)

| Sub | Hypothesis | Prediction | Test Method |
|-----|-----------|------------|-------------|
| H1a | Provider alignment trait > model size effect | Claude 계열이 GPT/Gemini보다 일관되게 높은 거부율 | Same 30 problems, economic budget |
| H1b | Larger models → lower C3 (less memorization) | sonnet C3 < haiku C3 | Compare C3 rates |
| H1c | Larger models → lower C2 (less false confidence) | balanced C2 < economic C2 | Compare C2 rates |

**Current evidence**:
- claude-sonnet-4: 77.4% refusal (highest)
- gpt-4o: 64.3%
- gemini-2.5-pro: 63.5%
- Prior economic (n=5): all ~66.7%

### H2: Reasoning vs Non-Reasoning

| Sub | Hypothesis | Prediction | Test Method |
|-----|-----------|------------|-------------|
| H2a | Reasoning traces may cause over-rationalization | gemini-2.5-pro lowest refusal despite thinking mode | Examine reasoning traces in C2 cases |
| H2b | Reasoning models have higher C2 rate | gemini C2 > others | Current: gemini C2=14.8%, sonnet C2=9.6% |
| H2c | self_verification prompt boosts reasoning model more | gemini delta(self_verif - metacognitive) > others | Run with --prompt-strategy self_verification |

### H3: Transformation Type Difficulty Hierarchy

**Current ranking** (by C1 rate):

```
EA-full (85.7%) > EA-partial (66.7%) > IC (64.3%) > SA (63.1%) >> TA (11.1%)
```

| Sub | Hypothesis | Prediction | Rationale |
|-----|-----------|------------|-----------|
| H3a | Structural absence > marker-based absence | EA-full C1 > EA-partial C1 | Missing column is stronger signal than N/A in one cell |
| H3b | High C3 in EA-partial/SA is due to question leakage | After leakage fix, C3 drops 30%+ | 37.3% of C3 is question_leaks_data |
| H3c | TA low C1 is valid finding, not design flaw | TA C1 stays low even with more data | Temporal ambiguity is genuinely harder |

### H4: IC (Information Conflict) Detection

**Critical finding**: IC improved from ~0% (economic, prior) to 64.3% (balanced).

| Sub | Hypothesis | Prediction | Test Method |
|-----|-----------|------------|-------------|
| H4a | IC improvement is due to prompt refinement, not just model size | Economic models with new prompt also show improvement | Re-run economic with current prompt |
| H4b | gemini-2.5-pro best at IC due to reasoning traces | gemini IC C1 > others | Current: gemini IC=82%, gpt-4o=50%, sonnet=61% |
| H4c | IC C3 cases use original (not conflicting) values | C3 models ignore inserted 1.5x value | Inspect executed code for value sources |
| H4d | contradiction_aware prompt further improves IC | IC C1 with contradiction_aware > metacognitive | Run with --prompt-strategy contradiction_aware |

### H5: Question Data Leakage Impact

**Current C3**: 67/345 = 19.4%

| Sub | Hypothesis | Prediction | Test Method |
|-----|-----------|------------|-------------|
| H5a | 30-40% of C3 is question leakage | After filtering, ~40 C3 remain | Build question_leakage_detector |
| H5b | Adjusted refusal rate rises to ~75% | (236+~25) / (345-~25) ≈ 81% | Recompute excluding leakage cases |
| H5c | 15-20% of problems are untransformable for certain types | Pre-filter identifies them | Check all solution values vs question values |

---

## 3. Transformation Intensity Quantification

### 3.1 Proposed Metrics (Priority Order)

#### Metric 1: Critical Value Removal Ratio (CVRR) — Priority 1

```python
def critical_value_removal_ratio(
    original_context: str,
    transformed_context: str,
    python_solution: str,
    question: str,
) -> float:
    """Fraction of solution-required values that were removed."""
    # Extract values used in solution code
    solution_values = extract_values_from_code(python_solution)
    # If hardcoded, extract from question + context
    if not solution_values:
        solution_values = extract_values_from_question(question)

    original_values = extract_all_numbers(original_context)
    transformed_values = extract_all_numbers(transformed_context)
    removed_values = original_values - transformed_values

    critical_removed = solution_values & removed_values
    return len(critical_removed) / len(solution_values) if solution_values else 0
```

**Interpretation**:
- CVRR = 0.0 → 비핵심 데이터만 제거됨 (C3 예상)
- CVRR = 0.5 → 핵심 데이터 일부 제거
- CVRR = 1.0 → 모든 핵심 데이터 제거 (C1 보장해야 함)

#### Metric 2: Information Removal Ratio (IRR) — Priority 2

```python
def info_removal_ratio(original_context: str, transformed_context: str) -> float:
    """Fraction of numerical values removed."""
    orig_nums = extract_all_numbers(original_context)
    trans_nums = extract_all_numbers(transformed_context)
    removed = orig_nums - trans_nums
    return len(removed) / len(orig_nums) if orig_nums else 0
```

**Expected ranges by type**:
| Type | IRR Range | Description |
|------|-----------|-------------|
| EA-partial | 1-5% | One value → N/A |
| EA-full | 10-40% | Entire column removed |
| SA (table) | 50-100% | Data rows removed |
| SA (text) | 10-30% | Key sentences removed |
| IC | 0% (+ added) | Values added, not removed |

#### Metric 3: Character Edit Distance Ratio (CEDR) — Priority 3

```python
def char_edit_ratio(original: str, transformed: str) -> float:
    """Normalized Levenshtein distance."""
    from Levenshtein import distance
    return distance(original, transformed) / max(len(original), 1)
```

#### Metric 4: Question-Context Information Overlap (QCIO) — Priority 4

```python
def question_context_overlap(question: str, context: str) -> float:
    """Fraction of question's numerical values present in context."""
    q_nums = extract_all_numbers(question)
    c_nums = extract_all_numbers(context)
    if not q_nums:
        return 0
    return len(q_nums & c_nums) / len(q_nums)
```

**Purpose**: QCIO(question, transformed_context) = 1.0이면 변환이 무의미 (question에 모든 값 잔존).

### 3.2 Intensity-Response Correlation Analysis

```
변환 강도 (CVRR/IRR) vs 모델 반응 (C1/C2/C3)

CVRR    Expected    Actual?
0.0     C3 (풀이 가능)   → 확인 필요
0.3     C2/C3       → 모델 의존적
0.7     C1/C2       → 대부분 거부
1.0     C1 (풀이 불가)   → 확인 필요

Hypothesis: CVRR과 C1 비율 간 monotonic positive correlation
```

---

## 4. Experimental Plan

### Phase 1: Quantification Infrastructure (코드 구현, API 비용 $0)

1. `evaluation/transformation_intensity.py` — CVRR, IRR, CEDR, QCIO 계산
2. `evaluation/question_leakage_detector.py` — Question 데이터 누출 탐지
3. 기존 `batch_transformations_0_30.json`에 intensity metrics 추가

### Phase 2: Controlled Experiments (API 비용 ~$1-2)

| Experiment | Purpose | Command | Hypothesis |
|-----------|---------|---------|------------|
| Economic comparison | H1 (model size) | `--budget economic` | H1a/b/c |
| Prompt strategy comparison | H2c, H4d | `--prompt-strategy self_verification` | H2c, H4d |
| IC code inspection | H4c | Analyze executed_code in IC-C3 | H4c |

### Phase 3: Analysis & Visualization

1. CVRR vs C1 scatter plot (per type, per model)
2. Question leakage impact: before/after adjusted rates
3. Model × Type × Intensity 3D heatmap
4. Hypothesis test results summary table

### Phase 4: Paper Writing

1. Citation 수정 (Feng → Wen)
2. Novelty claim: 금융 도메인 첫 메타인지 평가
3. Transformation intensity as controlled variable
4. CVRR threshold for effective transformation

---

## 5. Key Insights from Current Data

### 5.1 C3 Root Cause Distribution (67건)

| Cause | Count | % | Actionable? |
|-------|-------|---|-------------|
| Question에 핵심 데이터 중복 | 25 | 37.3% | **Yes**: question 내 값 포함 시 변환 skip |
| 모순 무시 (IC) | 13 | 19.4% | **No**: IC 변환의 근본 한계 |
| 비핵심 데이터 제거 | 13 | 19.4% | **Yes**: CVRR=0 문제 식별 |
| 암기 가능성 | 8 | 11.9% | **No**: 모델 내부 특성 |
| 서술문만 제거 | 3 | 4.5% | **Yes**: 수치 없는 문장 제거 방지 |
| TA 연도 잔존 | 4 | 6.0% | **Partial**: 데이터 내 연도도 함께 변환 |
| 범용 숫자 제거 | 1 | 1.5% | **Yes**: 10, 100 등 skip |

**Actionable C3**: 42/67 (62.7%) — 변환 알고리즘 개선으로 해결 가능
**Non-actionable C3**: 25/67 (37.3%) — 모델 특성 (IC 무시, 암기)

### 5.2 Citation Error Summary

| Current (in code) | Correct | Action |
|-------------------|---------|--------|
| Feng+2024 (Abstention F1) | Wen+2024 (EMNLP Findings) | Replace |
| FActScore concept (HR) | Min+2023 (EMNLP) — but simplified | Clarify "inspired by" |
| Rajpurkar+2018 | Correct | Keep |
| Yang+2024 | Correct | Keep |
| Cheng+2024 | Correct | Keep |

### 5.3 New References to Add

- Wen+2025 (TACL) — Abstention survey (메트릭 체계의 근거)
- Griot+2025 (Nature Comms) — Medical metacognition failure (cross-domain comparison)
- AbstentionBench 2025 — Latest benchmark (scale comparison)
- Kadavath+2022 (Anthropic) — P(IK) calibration (alternative approach reference)
