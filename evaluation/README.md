# FinanceReasoning Evaluation System

FinanceReasoning 평가 시스템 - arXiv paper 기반 구현

## 📋 요구사항

1. 논문 기반 평가 지표 구현 (Accuracy, Completeness, Order, Similarity, Hallucination)
2. 여러 LLM 제공업체 지원 (OpenAI, Anthropic, etc.)
3. 추론 과정(Reasoning Trace) 추출 및 평가
4. 비용 추적 및 예산 관리
5. 결과 저장 및 캐싱 (SQLite)
6. Paper 기준 보고서 생성

## 🏗️ 시스템 아키텍처

```
┌───────────────────────────────────────────┐
│                CLI / Config Entry Point     │
└───────────────────────┬───────────────────┘
                        │
┌───────────────▼─────────────────────────────────┐
│            Experiment Orchestrator        │
│  config, seeds, schedules, concurrency    │
└───────┬───────────────┬──────────────┘
        │               │               │
    ┌────────────▼────────────────────────────────────────────────┐
    │ Dataset Loader  │ Prompt Builder │ Model Registry │
    │ local JSON files│ templates/spec  │ providers/models │
    └───────┬────────┘   └───────┬──────────────────────┘
            │                   │               │
            └───────────▼───────────────────────────────────────────────────────────┘
                        │
            ┌───────────▼───────────────────────────────────────────────────────────────────┐
            │ Model Runner (Async, Rate-Limited)       │
            │  OpenAI/Anthropic/Other adapters       │
            │  retry + cost tracking                 │
            └───────────┬───────────────────────────────────────────────────────────────────────────┘
                        │
            ┌───────────▼───────────────────────────────────────────────────────────────────────────┐
            │ Response Parser / Trace Extractor         │
            │  final answer, steps, citations, etc.  │
            └───────────┬───────────────────────────────────────────────────────────────────────────┘
                        │
            ┌───────────▼───────────────────────────────────────────────────────────────────────────────────┐
            │ Metrics Evaluator                          │
            │  paper metrics, normalization, scoring    │
            └───────────┬───────────────────────────────────────────────────────────────────────────┘
                        │
            ┌───────────▼───────────────────────────────────────────────────────────────────────────────────┐
            │ Result Store + Cache                     │
            │  SQLite + JSON artifacts + blobs            │
            └───────────┬───────────────────────────────────────────────────────────────────────────┘
                        │
            ┌───────────────▼─────────────────────┐
            │ Aggregator / Comparator                 │
            │ per-model, per-category, overall         │
            └───────────────┬─────────────────────────────┘
                        │
            ┌───────────▼───────────────────────────────────────┐
            │ Report Generator                         │
            │ Markdown/HTML + figures                 │
            └───────────────────────────────────────────────┘
```

## 📦 컴포넌트

### Core Components

1. **Config**: API 키 관리 (env vars + config file)
2. **DatasetLoader**: FinanceReasoning 데이터 로드
3. **PromptBuilder**: 구조화된 프롬프트 생성
4. **ModelRunner**: 비동기 LLM API 호출, 동시성 관리
5. **ResponseParser**: LLM 응답 파싱 및 추론 단계 추출
6. **MetricsEvaluator**: 종합 평가 지표 계산
7. **ResultStore**: 결과 저장 및 캐싱 (SQLite)
8. **POTExecutor**: Program-of-Thought 코드 실행 샌드박스
9. **ReportGenerator**: Paper 기준 보고서 생성

### CLI Commands

- `python -m financereasoning.eval run-eval` - 새 평가 실행
- `python -m financereasoning.eval resume <id>` - 이전 평가 이어서 진행
- `python -m financereasoning.eval report <id>` - 보고서 생성
- `python -m financereasoning.eval inspect <id>` - 결과 검사

## 🔑 인증 설정

### 옵션 1: 환경 변수 (.env)
```bash
# 프로젝트 루트의 .env 파일
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 옵션 2: 설정 파일 (config/local_secrets.yaml)
```yaml
# .gitignore 포함되어야 함
models:
  openai:
    api_key: ${OPENAI_API_KEY}
    models:
      - id: "gpt-4o"
        name: "GPT-4o"
      - id: "gpt-4o-mini"
        name: "GPT-4o Mini"

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    models:
      - id: "claude-sonnet-4.5"
        name: "Claude Sonnet 4.5"
```

## 📊 평가 지표

### Paper Metrics

1. **Final Answer Accuracy**: 정답 정확도
2. **Step Completeness**: 추론 단계 포괄성 (ground_truth_steps 대비)
3. **Step Order Correctness**: 추론 순서 정확성
4. **Reasoning Similarity**: 추론 단계 유형 분포 유사성 (Jaccard index)
5. **Hallucination Rate**: 할루시네이션 비율
6. **Overall Reasoning Score**: 종합 점수 (가중평균)

### 점수 계산 공식

```
Overall Reasoning Score =
  (Final Answer Accuracy × 0.4) +
  (Step Completeness × 0.3) +
  (Step Order Correct × 0.1) +
  ((1 - Hallucination Rate) × 0.2)
```

## 📂 데이터 형식

### 입력 데이터 (hard.json)
```json
{
  "question": "...",
  "context": "...",
  "ground_truth": 1152,
  "python_solution": "shares_outstanding = 1328...",
  "question_id": "test-2000",
  "level": "hard"
}
```

### 추론 트레이스 (financial_reasoning_traced.json)
```json
{
  "reasoning_trace": {
    "total_steps": 9,
    "steps": [
      {
        "step_number": 1,
        "step_type": "variable_definition",
        "description": "...",
        "code_snippet": "..."
      }
    ]
  }
}
```

## 🎯 평가 전략 (Paper 기준)

### Chain-of-Thought (COT)

**Status**: ✅ Implemented

Models reason step-by-step in natural language and output final answer.

**Prompt Format**:
```
You are a financial reasoning assistant. Answer question step-by-step.

Output format:
{
  "final_answer": [number or string],
  "reasoning_steps": [
    {
      "step_number": 1,
      "step_type": "variable_definition|calculation|conditional|return_result",
      "description": "...",
      "code_snippet": "..."
    }
  ]
}
```

### Program-of-Thought (POT)

**Status**: ✅ Implemented

Models generate Python code which is executed to produce final answer.

**Features:**
- **Code Extraction**: Supports ` ```python ... ```, code blocks, or plain code
- **Sandboxed Execution**: Runs code in isolated subprocess
- **Timeout Protection**: Default 10 second timeout (configurable)
- **Output Capture**: Captures stdout and extracts `answer` variable or print output
- **Error Handling**: Comprehensive error handling with detailed messages

**Prompt Format**:
```
You are a financial reasoning assistant with strong Python programming skills.

Generate a Python program to solve the given financial problem:
1. Define all necessary variables
2. Perform calculations step-by-step
3. Include comments explaining each step
4. The final answer should be stored in a variable called 'answer'
5. Use `return answer` as the last statement

Output ONLY Python code, no explanations outside code block.
```

**Example Execution**:
```python
# LLM Output:
shares_outstanding = 1328
acquisition_cost = 176
answer = shares_outstanding - acquisition_cost

# Execution Result:
Success: True
Output: 1152.0
Time: 0.024s
```

### COT + RAG / POT + RAG

**Status**: ⏳ Pending Implementation

Models can retrieve relevant financial functions from function library to assist reasoning.

**Function Library**:
- Location: `raw/functions/functions-article-all.json`
- 3,133 Python-formatted financial functions
- Examples: NPV calculation, amortization, ratio analysis

## 📁 파일 구조

```
finance_LLM/
├── .env                              # API keys (gitignored)
├── data/
│   └── financereasoning/
│       ├── raw/
│       │   ├── FinanceReasoning/
│       │   │   ├── easy.json      # 1,000 examples
│       │   │   ├── medium.json    # 1,000 examples
│       │   │   └── hard.json      # 238 examples
│       │   ├── functions/
│       │   │   └── functions-article-all.json  # 3,133 functions
│       │   └── documents/
│       │       └── financial_documents.json
│       ├── evaluation/
│       │   ├── main.py                 # Entry point
│       │   ├── config.py               # Model configuration
│       │   ├── config/
│       │   │   └── local_secrets.yaml  # Your model config
│       │   ├── dataset_loader.py        # Load datasets
│       │   ├── model_runner.py          # API orchestration
│       │   ├── prompt_builder.py        # Prompt templates
│       │   ├── response_parser.py       # Parse responses
│       │   ├── pot_executor.py          # POT code execution sandbox
│       │   ├── metrics_evaluator.py    # Calculate metrics
│       │   └── result_store.py         # SQLite storage
│       └── results/
│           └── evaluations.db         # Cached results
└── test_financereasoning.py          # Quick test script
```

## 📖 상태 문서

- **Completed**:
  - Basic evaluation framework
  - COT (Chain-of-Thought) strategy
  - POT (Program-of-Thought) code execution sandbox
  - Response parsing for both strategies
  - Multiple model support (OpenAI, Anthropic)
  - Configuration system with model overrides
  - Documentation

- **In Progress**:
  - RAG (Retrieval-Augmented Generation) with function library
  - Full dataset benchmarking
  - Paper-compliant report generation

## API 인증 문제 (현재 상태)

### 알려진 이슈:

1. **OpenAI API**: 현재 `.env`의 API 키가 무효 또는 만료됨
   - HTTP 401 Unauthorized 오류 발생
   - 새로운 API 키 필요 (https://platform.openai.com/account/api-keys)

2. **Anthropic API**: 현재 모델 ID 문제
   - `claude-3-5-sonnet-20240620` 모델이 없음
   - `claude-sonnet-4.5`로 변경 필요

### 해결 방안:

1. **OpenAI API 키 업데이트**:
   ```bash
   # .env 파일에 새로운 API 키 추가
   OPENAI_API_KEY=sk-proj-새키...
   ```

2. **Anthropic 모델 ID 수정**:
   - `config/local_secrets.yaml`에 `claude-sonnet-4.5` 사용 (이미 수정됨)
   - 또는 `claude-3-opus-20240229` 사용 가능

3. **GPT 모델 사용**:
   - `gpt-4o` 및 `gpt-4o-mini` 모델들은 API 키가 유효하면 정상 작동
   - 우선적으로 이들 모델로 테스트 권장

## 퀵 테스트

```bash
# 단일 예제 테스트
python test_financereasoning.py

# 전체 평가 실행
cd data/financereasoning/evaluation
python main.py
```

## Paper 결과와 비교

| Metric | Paper Top (GPT-5) | Target |
|---------|---------------------|--------|
| Hard Set Accuracy | ~65% | 60%+ |
| Overall Score | ~0.72 | 0.65+ |

**Note**: 모든 논문 모델을 구현할 필요는 없습니다. 접근 가능한 모델로 최대한 유사한 실험 환경을 구성했습니다.

---

**Last Updated**: 2026-01-20
**Status**: COT 및 POT 프레임워크 구현 완료, RAG 보류 중
