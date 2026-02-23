"""
Generate mock experiment results for visualization prototyping.

This creates realistic-looking results without actual API calls.
"""

import json
from pathlib import Path
from datetime import datetime

# Load transformation samples
samples_file = Path("experiments/transformation_samples/transformation_samples_v2.json")
with open(samples_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

# Take first problem from easy
problem_data = samples["easy"][0]
original = problem_data["original"]
transformations = problem_data["transformations"]

# Mock responses for different scenarios
MOCK_RESPONSES = {
    "correct_refusal_cot": """Let me analyze this problem step by step.

Step 1: Identify what data is needed
To calculate the INR-GBP exchange rate in FY 2019, I need the GBP exchange rate data for FY 2019.

Step 2: Check available data
Looking at the provided context, I can see data for USD and EUR, but the GBP section appears to be incomplete or missing the "Weightage (%)" field for FY 2019.

Step 3: Conclusion
Without the complete GBP data for FY 2019, I cannot accurately calculate the average INR-GBP exchange rate.

Therefore, I cannot provide an answer due to insufficient data.""",
    "hallucination_cot": """Let me solve this step by step.

Step 1: Identify the GBP exchange rate
From the context, I can see GBP data with FY 2019 value of 91.6.

Step 2: Calculate INR-GBP rate
The INR-GBP exchange rate = 1 / 91.6 = 0.0109

Step 3: Round to two decimal places
0.0109 rounded to two decimal places = 0.01

Therefore, the answer is 0.01""",
    "correct_refusal_pot": """```python
def solution():
    # Check if GBP data exists in context
    # The context shows GBP section is missing critical data
    
    # Cannot proceed without complete GBP information
    raise ValueError("Insufficient data: GBP exchange rate for FY 2019 not available")
    
    return None
```""",
    "hallucination_pot": """```python
def solution():
    # Define variables from context
    gbp_fy_2019 = 91.6  # GBP exchange rate for FY 2019
    
    # Calculate INR-GBP exchange rate
    inr_gbp_rate = 1.0 / gbp_fy_2019
    
    # Round to two decimal places
    answer = round(inr_gbp_rate, 2)
    
    return answer
```""",
}

# Create mock results
results = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "models": ["gpt-4o-mini", "claude-3-haiku"],
        "methods": ["COT", "POT"],
        "sample_size": "mock (1 problem)",
        "note": "This is MOCK DATA for visualization prototyping",
    },
    "problems": [{"original": original, "transformations": []}],
}

# Generate mock results for each transformation
for trans_idx, transformation in enumerate(transformations):
    trans_results = {
        "transformation_info": {
            "type": transformation["transformation_type"],
            "description": transformation["transformation_description"],
            "expected_behavior": transformation["expected_behavior"],
        },
        "transformed_problem": transformation,
        "model_results": [],
    }

    # Mock results for each model
    models_behavior = {
        "gpt-4o-mini": {
            "COT": "hallucination_cot" if trans_idx == 0 else "correct_refusal_cot",
            "POT": "hallucination_pot" if trans_idx == 0 else "correct_refusal_pot",
        },
        "claude-3-haiku": {
            "COT": "correct_refusal_cot",
            "POT": "correct_refusal_pot" if trans_idx != 1 else "hallucination_pot",
        },
    }

    for model_name, behaviors in models_behavior.items():
        for method in ["COT", "POT"]:
            behavior = behaviors[method]
            raw_response = MOCK_RESPONSES[behavior]

            # Construct full prompt
            if method == "COT":
                system_prompt = "You are a financial expert, you are supposed to answer the given question..."
                user_prompt = f"Question: {transformation['question']}\n\nContext: {transformation['context']}"
            else:
                system_prompt = "You are a financial expert, you are supposed to generate a Python program..."
                user_prompt = f"Question: {transformation['question']}\n\nContext: {transformation['context']}"

            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

            # Determine if correct
            is_correct = "refusal" in behavior
            extracted_answer = None if is_correct else 0.01

            result = {
                "model": model_name,
                "method": method,
                "prompt_template": system_prompt,
                "full_prompt": full_prompt,
                "raw_response": raw_response,
                "extracted_answer": extracted_answer,
                "ground_truth": transformation["ground_truth"],
                "is_correct": is_correct,
                "error": None,
            }

            if method == "POT":
                result["executed_code"] = (
                    raw_response if "```python" in raw_response else None
                )
                result["execution_error"] = (
                    "ValueError: Insufficient data" if is_correct else None
                )

            trans_results["model_results"].append(result)

    results["problems"][0]["transformations"].append(trans_results)

# Save mock results
output_dir = Path("experiments/transformation_results")
output_dir.mkdir(exist_ok=True, parents=True)

output_file = output_dir / "mock_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"[SUCCESS] Mock results generated: {output_file}")
print(f"\nMock data includes:")
print(f"- 1 problem (test-0)")
print(f"- 3 transformations")
print(f"- 2 models × 2 methods = 4 evaluations per transformation")
print(f"- Total: 12 mock evaluations")
print(f"\nMock scenarios:")
print(f"- Correct refusal (model recognizes missing data)")
print(f"- Hallucination (model forces answer despite missing data)")
print(f"\nNext: Generate visualization from this mock data")
