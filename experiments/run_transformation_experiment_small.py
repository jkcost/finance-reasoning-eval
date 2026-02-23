"""
Run evaluation on SMALL SAMPLE of transformed data (1-2 problems only).

This is a quick test to:
1. Validate the experiment pipeline
2. Generate sample data for visualization
3. Estimate costs before full run

Based on paper_error_cases_experiment.py pattern.
"""

import asyncio
import json
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import ConfigManager, ModelConfig
from model_runner import OpenAIProvider, AnthropicProvider, GoogleProvider


# ============================================================================
# OFFICIAL PAPER PROMPTS
# ============================================================================

COT_SYSTEM_INPUT = """You are a financial expert, you are supposed to answer the given question. You need to first think through the problem step by step, identifying the exact variables and values, and documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. The final answer should be a numeric value."""

COT_PROGRAM_PREFIX_INPUT = (
    """Let's think step by step to answer the given question.\n"""
)

POT_SYSTEM_INPUT = """You are a financial expert, you are supposed to generate a Python program to answer the given question. The returned value of the program is supposed to be the answer. Here is an example of the Python program:
```python
def solution():
    # Define variables name and value
    revenue = 600000
    avg_account_receivable = 50000
    
    # Do math calculation to get the answer
    receivables_turnover = revenue / avg_account_receivable
    answer = 365 / receivables_turnover
    
    # return answer
    return answer
```
"""

POT_PROGRAM_PREFIX_INPUT = """Please generate a Python program to answer the given question. The format of the program should be the following:
```python
def solution():
    # Define variables name and value
    
    # Do math calculation to get the answer
    
    # return answer
```

Continue your output:
```python
def solution():
    # Define variables name and value
"""


# ============================================================================
# MODEL CONFIGURATIONS (Budget models only for quick test)
# ============================================================================

BUDGET_MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "temperature": 0.0,
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "temperature": 0.0,
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def extract_answer_from_cot(response: str) -> Optional[float]:
    """Extract numerical answer from COT response."""
    # Look for "Therefore, the answer is X" pattern
    patterns = [
        r"Therefore,?\s+the answer is\s+([-+]?[\d,]+\.?\d*)",
        r"the answer is\s+([-+]?[\d,]+\.?\d*)",
        r"answer:\s*([-+]?[\d,]+\.?\d*)",
        r"final answer:\s*([-+]?[\d,]+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                answer_str = match.group(1).replace(",", "")
                return float(answer_str)
            except ValueError:
                continue

    return None


def extract_code_from_pot(response: str) -> Optional[str]:
    """Extract Python code from POT response."""
    # Look for code block
    code_pattern = r"```python\s*(.*?)\s*```"
    match = re.search(code_pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()

    # If no code block, try to extract function definition
    if "def solution():" in response:
        # Extract from "def solution():" to the end or next ```
        start = response.find("def solution():")
        end = response.find("```", start)
        if end == -1:
            end = len(response)
        return response[start:end].strip()

    return None


def execute_pot_code(code: str) -> Optional[float]:
    """Execute POT code and return answer."""
    try:
        # Create execution environment
        exec_globals = {}
        exec(code, exec_globals)

        # Call solution function
        if "solution" in exec_globals:
            result = exec_globals["solution"]()
            return float(result)
    except Exception as e:
        print(f"    [POT Execution Error]: {str(e)}")
        return None

    return None


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


async def evaluate_cot(provider: Any, problem: Dict, model_config: Dict) -> Dict:
    """Evaluate using Chain-of-Thought."""

    # Construct prompt
    user_message = f"{COT_PROGRAM_PREFIX_INPUT}\nQuestion: {problem['question']}\n\nContext: {problem['context']}"

    full_prompt = f"System: {COT_SYSTEM_INPUT}\n\nUser: {user_message}"

    # Call model
    try:
        response = await provider.generate(
            messages=[
                {"role": "system", "content": COT_SYSTEM_INPUT},
                {"role": "user", "content": user_message},
            ],
            temperature=model_config.get("temperature", 0.0),
            max_tokens=2000,
        )

        raw_response = response.get("content", "")

        # Extract answer
        extracted_answer = extract_answer_from_cot(raw_response)

        # Check correctness
        is_correct = False
        if extracted_answer is not None:
            ground_truth = float(problem["ground_truth"])
            # Allow 1% tolerance
            is_correct = (
                abs(extracted_answer - ground_truth) / max(abs(ground_truth), 1e-10)
                < 0.01
            )

        return {
            "method": "COT",
            "prompt_template": COT_SYSTEM_INPUT,
            "full_prompt": full_prompt,
            "raw_response": raw_response,
            "extracted_answer": extracted_answer,
            "ground_truth": problem["ground_truth"],
            "is_correct": is_correct,
            "error": None,
        }

    except Exception as e:
        return {
            "method": "COT",
            "error": str(e),
            "raw_response": "",
            "extracted_answer": None,
            "is_correct": False,
        }


async def evaluate_pot(provider: Any, problem: Dict, model_config: Dict) -> Dict:
    """Evaluate using Program-of-Thought."""

    # Construct prompt
    user_message = f"{POT_PROGRAM_PREFIX_INPUT}\n\nQuestion: {problem['question']}\n\nContext: {problem['context']}"

    full_prompt = f"System: {POT_SYSTEM_INPUT}\n\nUser: {user_message}"

    # Call model
    try:
        response = await provider.generate(
            messages=[
                {"role": "system", "content": POT_SYSTEM_INPUT},
                {"role": "user", "content": user_message},
            ],
            temperature=model_config.get("temperature", 0.0),
            max_tokens=2000,
        )

        raw_response = response.get("content", "")

        # Extract code
        code = extract_code_from_pot(raw_response)

        # Execute code
        extracted_answer = None
        execution_error = None
        if code:
            extracted_answer = execute_pot_code(code)
            if extracted_answer is None:
                execution_error = "Code execution failed"
        else:
            execution_error = "No code found in response"

        # Check correctness
        is_correct = False
        if extracted_answer is not None:
            ground_truth = float(problem["ground_truth"])
            is_correct = (
                abs(extracted_answer - ground_truth) / max(abs(ground_truth), 1e-10)
                < 0.01
            )

        return {
            "method": "POT",
            "prompt_template": POT_SYSTEM_INPUT,
            "full_prompt": full_prompt,
            "raw_response": raw_response,
            "executed_code": code,
            "execution_error": execution_error,
            "extracted_answer": extracted_answer,
            "ground_truth": problem["ground_truth"],
            "is_correct": is_correct,
            "error": None,
        }

    except Exception as e:
        return {
            "method": "POT",
            "error": str(e),
            "raw_response": "",
            "executed_code": None,
            "extracted_answer": None,
            "is_correct": False,
        }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


async def run_experiment():
    """Run small sample experiment."""

    # Load environment
    load_dotenv()

    # Load transformed samples
    samples_file = Path(
        "experiments/transformation_samples/transformation_samples_v2.json"
    )
    with open(samples_file, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # Take only FIRST problem from EASY (with all its transformations)
    samples = {
        "easy": [all_samples["easy"][0]]  # Just first problem
    }

    print("=" * 70)
    print("SMALL SAMPLE TRANSFORMATION EXPERIMENT")
    print("=" * 70)
    print(
        f"Testing: 1 problem with {len(samples['easy'][0]['transformations'])} transformations"
    )
    print(f"Models: {', '.join(BUDGET_MODELS.keys())}")
    print(f"Methods: COT, POT")
    print(
        f"Total API calls: {len(samples['easy'][0]['transformations']) * len(BUDGET_MODELS) * 2}"
    )
    print("=" * 70)
    print()

    # Initialize providers
    config_manager = ConfigManager()
    providers = {}

    for model_name, config in BUDGET_MODELS.items():
        # Create ModelConfig object
        model_config = ModelConfig(
            id=model_name,
            name=model_name,
            provider=config["provider"],
            model_id=config["model_id"],
            api_key_env_var="OPENAI_API_KEY"
            if config["provider"] == "openai"
            else "ANTHROPIC_API_KEY",
            max_tokens=2000,
            temperature=config.get("temperature", 0.0),
            cost_per_million_tokens=0.0,  # Not tracking cost for now
        )

        if config["provider"] == "openai":
            providers[model_name] = OpenAIProvider(model_config)
        elif config["provider"] == "anthropic":
            providers[model_name] = AnthropicProvider(model_config)
        elif provider_type == "anthropic":
            providers[model_name] = AnthropicProvider(
                model_id=model_config["model_id"], config_manager=config_manager
            )

    # Results structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": list(BUDGET_MODELS.keys()),
            "methods": ["COT", "POT"],
            "sample_size": "small (1 problem)",
        },
        "problems": [],
    }

    # Process the single problem
    for problem_data in samples["easy"]:
        original = problem_data["original"]
        transformations = problem_data["transformations"]

        print(f"\n{'=' * 70}")
        print(f"Problem: {original['question_id']}")
        print(f"Question: {original['question'][:80]}...")
        print(f"Transformations: {len(transformations)}")
        print(f"{'=' * 70}\n")

        problem_results = {"original": original, "transformations": []}

        # Evaluate each transformation
        for trans_idx, transformation in enumerate(transformations):
            print(
                f"\nTransformation {trans_idx + 1}/{len(transformations)}: {transformation['transformation_type']}"
            )
            print(f"Description: {transformation['transformation_description']}")

            trans_results = {
                "transformation_info": {
                    "type": transformation["transformation_type"],
                    "description": transformation["transformation_description"],
                    "expected_behavior": transformation["expected_behavior"],
                },
                "transformed_problem": transformation,
                "model_results": [],
            }

            # Test each model with each method
            for model_name, provider in providers.items():
                model_config = BUDGET_MODELS[model_name]

                # COT
                print(f"  {model_name} (COT)...", end=" ", flush=True)
                cot_result = await evaluate_cot(provider, transformation, model_config)
                cot_result["model"] = model_name
                trans_results["model_results"].append(cot_result)

                status = "[OK]" if cot_result.get("is_correct") else "[FAIL]"
                print(f"{status}")

                # POT
                print(f"  {model_name} (POT)...", end=" ", flush=True)
                pot_result = await evaluate_pot(provider, transformation, model_config)
                pot_result["model"] = model_name
                trans_results["model_results"].append(pot_result)

                status = "[OK]" if pot_result.get("is_correct") else "[FAIL]"
                print(f"{status}")

            problem_results["transformations"].append(trans_results)

        results["problems"].append(problem_results)

    # Save results
    output_dir = Path("experiments/transformation_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"small_sample_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"[SUCCESS] Results saved to: {output_file}")
    print(f"{'=' * 70}")

    # Print summary
    total_evals = (
        sum(len(p["transformations"]) for p in results["problems"])
        * len(BUDGET_MODELS)
        * 2
    )
    print(f"\nTotal evaluations: {total_evals}")
    print("\nNext step: Generate visualization from these results")

    return output_file


if __name__ == "__main__":
    output_file = asyncio.run(run_experiment())
    print(f"\nVisualization input: {output_file}")
