# FinanceReasoning Evaluation System

Reorganized project structure for FinanceReasoning benchmark evaluation.

## 📁 Project Structure

```
finance_LLM/
├── .env                          # API keys (gitignored)
├── experiments/                   # Experiment configurations and runners
│   ├── experiment_config.py      # Experiment configuration manager
│   └── results_analyzer.py      # Results analysis and visualization
├── config/                       # Configuration files
│   ├── experiments.yaml         # Experiment definitions
│   └── local_secrets.yaml    # Model overrides
├── docs/                         # Consolidated documentation
├── data/
│   ├── financereasoning/         # Evaluation system
│   │   ├── raw/              # Benchmark datasets
│   │   │   └── FinanceReasoning/  # FinanceReasoning data
│   │   ├── evaluation/          # Core evaluation components
│   │   │   ├── main.py         # NEW: Updated entry point
│   │   │   ├── config.py        # Model configurations
│   │   │   ├── dataset_loader.py
│   │   │   ├── model_runner.py
│   │   │   ├── prompt_builder.py
│   │   │   ├── response_parser.py
│   │   │   ├── pot_executor.py
│   │   │   ├── metrics_evaluator.py
│   │   │   └── result_store.py
│   │   └── results/           # Evaluation outputs
│   └── functions/              # Financial function library
│       └── functions-article-all.json
└── .gitignore
```

## 🎯 Key Components

### 1. Experiments Framework (`experiments/experiment_config.py`)

**Features:**
- Experiment configuration management (YAML-based)
- Baseline comparison support
- Methodology selection (COT, POT, COT+RAG, POT+RAG, FinanceReasoning-compliant)
- Model selection per experiment
- Dataset configuration (split, examples limit)

**Supported Methodologies:**
- `cot`: Chain-of-Thought (natural language reasoning)
- `pot`: Program-of-Thought (code execution)
- `cot_rag`: COT with function retrieval
- `pot_rag`: POT with function retrieval
- `finance_reasoning_compliant`: Paper-compliant structured output

### 2. Results Analyzer (`experiments/results_analyzer.py`)

**Features:**
- Load multiple experiment results
- Model performance comparison across experiments
- CSV export for analysis
- Interactive HTML dashboard with:
  - Model filter and comparison
  - Accuracy and cost metrics
  - Scatter plots for performance visualization
  - Detailed per-model tables

### 3. Updated Main Entry Point (`data/financereasoning/evaluation/main.py`)

**New Commands:**
```bash
# List available experiments
python -m financereasoning.eval list

# Run named experiment
python -m financereasoning.eval run <experiment_name>

# Create new experiment
python -m financereasoning.eval create <name> <methodology> <model1> <model2> ...

# Example:
python -m financereasoning.eval run finance_reasoning_paper_reproduction
python -m financereasoning.eval create compare_cot_vs_pot cot claude-sonnet-4.5 gemini-2.5-pro
```

## 📊 Supported Models

**Claude Models:**
- `claude-sonnet-4.5` - Latest Sonnet (4.5)
- `claude-opus-4.5` - Latest Opus (4.5)
- `claude-haiku-4.5` - Latest Haiku (4.5)
- `claude-opus-4.1` - Legacy Opus 4.1
- `claude-sonnet-4` - Legacy Sonnet 4
- `claude-3-haiku` - Legacy Haiku 3

**Gemini Models:**
- `gemini-2.5-pro` - Latest Pro
- `gemini-2.0-flash` - Flash
- `gemini-2.5-flash` - Flash

**OpenAI Models:** (Requires valid API key)
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-5-2025-08-07`

## 🚀 Quick Start

1. **Configure API Keys** in `.env`:
   ```bash
   OPENAI_API_KEY=sk-proj-...
   ANTHROPIC_API_KEY=sk-ant-api03-...
   GOOGLE_API_KEY=AIzaSy...
   ```

2. **Run Baseline Experiment** (Paper COT):
   ```bash
   python -m financereasoning.eval run finance_reasoning_paper_reproduction
   ```

3. **Compare Results**:
   ```bash
   python experiments/results_analyzer.py list
   # View dashboard at: results/dashboard.html
   ```

4. **Create Custom Experiment**:
   ```bash
   python -m financereasoning.eval create my_experiment pot claude-sonnet-4.5
   ```

## 📁 Data Locations

| Component | Location |
|-----------|----------|
| Benchmark data | `data/financereasoning/raw/FinanceReasoning/` |
| Evaluation output | `data/financereasoning/evaluation/results/` |
| Experiment configs | `config/experiments.yaml` |
| Analysis results | `results/` (root-level) |

## 🔑 API Status Notes

- **OpenAI**: Current key expired or invalid. Update `.env` with valid key
- **Anthropic**: ✅ Working with updated model IDs
- **Gemini**: ✅ Working with `GOOGLE_API_KEY`

## 📚 Documentation

Consolidated in `docs/` folder (moved from root/data).

**Last Updated:** 2026-01-21
