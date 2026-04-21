# Finance LLM Project Memory

## Project Structure
- **Data**: `data/financereasoning/raw/FinanceReasoning/{easy,medium,hard}.json`
- **Evaluation**: `evaluation/` (model_runner, metrics, error_analysis/, refusal_detector, metacognitive_metrics)
- **Experiments**: `experiments/` (run_metacognitive_experiment.py, apply_transformations_full.py)
- **Results**: `experiments/results/metacognitive/`

## Key Files
| File | Purpose |
|------|---------|
| `evaluation/model_runner.py` | API providers (OpenAI, Anthropic, Google) with httpx |
| `evaluation/error_analysis/error_taxonomy.py` | 13 error categories + MODEL_REGISTRY + BUDGET_MODEL_SETS |
| `evaluation/metacognitive_metrics.py` | MetacognitiveResult dataclass, MC Score computation |
| `evaluation/refusal_detector.py` | Response classification (refused/caveat/confident/error) |
| `experiments/run_metacognitive_experiment.py` | Main experiment (Phase A~D) |
| `experiments/apply_transformations_full.py` | 4+1 transformation types (EA/SA/IC/TA) + validation |
| `experiments/generate_metacognitive_dashboard.py` | HTML dashboard generation |

## Model Registry (in error_taxonomy.py)
- **economic**: gpt-4o-mini, claude-haiku-4, gemini-2.5-flash
- **balanced**: gpt-4o, claude-sonnet-4, gemini-2.5-pro
- **full**: all models

## Metacognitive Metrics (updated 2026-02-23)
- MC Score = Refusal_F1 × (1 - Hallucination_Rate) — property name: `mc_score`
- v1 completely removed (2026-02-23)
- Refusal Recall = correctly_refused / total_unsolvable (Feng+2024)
- Refusal Precision = correctly_refused / total_refused (Feng+2024)
- Refusal F1 = harmonic mean of RP/RR
- Over-Conservatism Rate = solvable_refused / total_solvable (Yang+2024)
- Cross-phase: compute_cross_phase_metrics(phase_a, phase_b) merges solvable+unsolvable
- Graceful degradation: no Phase A → RP=1.0, F1=2RR/(1+RR)
- References: Feng+2024 Abstention F1, Yang+2024 Honesty Score, Cheng+2024 Balanced IDK

## Experiment State (2026-02-20)
- Phase A (economic, n=10, hard, 4 strategies): done (120 evals, $0.036)
- Phase B (economic, n=10, hard, 4 strategies): done (348 evals, $0.089)
- Phase D (cross-phase analysis): done (768 total evals)
- Best MC Score: claude-sonnet-4 (0.833), Best efficiency: gemini-2.5-flash (22.0 MC/$)
- Phase A/B (balanced): done (previous run)
- Phase C (RAG, metacognitive): done (previous run)
- contradiction_aware: gpt-4o-mini RefF1=0.926, gemini RefF1=0.906, haiku RefF1=0.863

## Dashboard Features (2026-02-20)
- Strategy × Type heatmap (CSS gradient cells)
- Strategy MC Score grouped bar chart (Chart.js)
- Type difficulty ranking (horizontal bar)
- Transformation design reference cards (collapsible)
- Phase A now supports --prompt-strategy all (4-strategy parallel)

## Known Issues (Fixed 2026-02-20)
- validate_transformation: hardcoded solutions bypass detection -> added _solution_uses_hardcoded_values()
- Type 5 "audited" label: models treat as authoritative -> changed to explicit discrepancy language
- model_runner semaphore: release() called in retry but ProviderSemaphore manages it via __aexit__
- model_runner 529 errors: insufficient retry for Anthropic overloaded -> extended backoff

## Transformation Taxonomy v2 (2026-03-03)
- **EA-partial** (Explicit Absence - Partial): [DATA MISSING]/N/A marker — was Type 1
- **EA-full** (Explicit Absence - Full): entire key/column deletion — was Type 2
- **SA** (Silent Absence): marker-free removal. Text: sentence deletion (new!) — was Type 4
- **IC** (Information Conflict): 1.5× contradictory value — was Type 5
- **TA** (Temporal Ambiguity): year → "end of period" (auxiliary, ~11 hard) — was Type 3
- Labels defined in `apply_transformations_full.py`: LABEL_EA_PARTIAL, LABEL_EA_FULL, etc.
- `normalize_transformation_label()` converts legacy "Type N" to new labels
- `LEGACY_LABEL_MAP` dict: old → new mapping
- SA text: `transform_sa_text()` — deletes question-keyword-matching sentences
- Dashboard/experiment loaders auto-convert legacy labels via normalize_transformation_label()

## Prompt Strategies
- standard, metacognitive, self_verification, contradiction_aware (added 2026-02-20)

## Submission Strategy (2026-04-15)
- [CIKM 2026 Short → AAAI 2027 확장 전략 (RQ-A/B/C CIKM, RQ-D/E AAAI)](project_submission_strategy.md)
- [RQ-B POT Faithfulness 측정 도구 (Value Provenance + Memorization Score)](project_pot_faithfulness_rqb.md) — 1million 모델별 재현 증거 포함 (from 4/15 sync)
- [RQ-C Conflict Salience Regression (4 feature logistic regression)](project_conflict_salience_rqc.md)

## Meeting-Derived Decisions (4/15 team sync)
- [연구 앵글 2축 확장 — 판정 이유 카테고리화 + Hard example 생성](project_rq_direction_shift_2026_04_15.md)
- [POT 프롬프트의 IC 판정 한계 — CoT/reasoning trace 전환](project_pot_limitation_ic.md)
- [원본 오답 문제는 메인 분석에서 제외](feedback_exclude_baseline_wrong.md)

## Feedback
- [변환은 반드시 LLM 기반으로 수행](feedback_llm_based_transformation.md)

## Project History
- [규칙 기반 변환 v1 작업 이력](project_rule_based_v1_archive.md)
- [회의록 보관소](../meetings/) — 주 1회 주간 회의 raw transcript + decisions 섹션

## Reasoning Trace Validation Pipeline (2026-03-05)
- **Purpose**: validate_transformation()의 python_solution 의존 한계 극복 (hard.json 64% 하드코딩)
- **Files**: `evaluation/reasoning_trace_analyzer.py`, `experiments/run_validation_pipeline.py`, `experiments/generate_validation_report.py`
- **Case 분류**: Case 1 (거부=유효), Case 2 (오답=유효), Case 3 (정답=암기/추론 분석)
- **Components**: NumberExtractor, ValueProvenanceClassifier, AwarenessDetector, MemorizationScorer, LLMJudge(optional)
- **Value Sources**: from_context, from_removed_data, fabricated, common_constant, derived
- **Verdict**: effective / compromised_by_memorization / transformation_insufficient
- **First run (1140 results)**: Case1=62%, Case2=15%, Case3=23%; SA 가장 유효, EA-full/TA 정답률 높음
- **validate_transformation()**: deprecated 주석 추가, 새 파이프라인으로 대체
