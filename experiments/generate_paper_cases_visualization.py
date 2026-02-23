"""
Generate improved visualization for paper error cases results.

Applies the same format as transformation visualization:
- Full prompts (System + User)
- Model reasoning process
- Why wrong/correct analysis
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def analyze_why_wrong_or_correct(result: Dict) -> str:
    """
    Analyze why the model got it wrong or correct.

    Returns HTML explanation.
    """
    is_correct = result.get("is_correct", False)
    method = result.get("method", "COT")
    raw_response = result.get("raw_response", "").lower()
    final_answer = result.get("final_answer")
    ground_truth = result.get("ground_truth")
    error_type = result.get("paper_error_type", "Unknown")

    if is_correct:
        # Analyze why correct
        reasons = []

        if method == "POT":
            reasons.append(
                "✅ <strong>Correct Code Logic</strong>: Generated code correctly implements the required calculation"
            )
            code = result.get("raw_response", "")
            if "expected" in code.lower() or "probability" in code.lower():
                reasons.append(
                    "✅ <strong>Proper Probability Handling</strong>: Code correctly handles probabilistic scenarios"
                )
        else:
            reasons.append(
                "✅ <strong>Correct Reasoning</strong>: Step-by-step logic leads to the right answer"
            )

        if abs(float(final_answer) - float(ground_truth)) < 0.01:
            reasons.append(
                f"✅ <strong>Accurate Calculation</strong>: Final answer ({final_answer}) matches ground truth ({ground_truth})"
            )

        html = f"""
        <div class="analysis-correct">
            <h6>✅ Why This is Correct:</h6>
            <p><strong>Error Type from Paper:</strong> {error_type}</p>
            <p><strong>What Happened:</strong></p>
            <ul>
                {"".join(f"<li>{reason}</li>" for reason in reasons)}
            </ul>
            <p><strong>Key Success Factor:</strong> Model successfully avoided the error pattern identified in the paper.</p>
        </div>
        """
        return html

    # Analyze why wrong
    reasons = []

    # Check for common error patterns
    if "risk premium" in raw_response and error_type == "Misunderstanding of Problem":
        reasons.append(
            "🔴 <strong>Misunderstanding of Problem</strong>: Model may have misapplied the risk premium in the calculation"
        )

    if "formula" in error_type.lower():
        reasons.append(
            "🔴 <strong>Formula Application Error</strong>: Model used incorrect formula or applied it incorrectly"
        )

    if "numerical" in error_type.lower() and "extraction" in error_type.lower():
        reasons.append(
            "🔴 <strong>Numerical Extraction Error</strong>: Model extracted wrong values from the problem statement"
        )

    if "calculation" in error_type.lower():
        reasons.append(
            "🔴 <strong>Numerical Calculation Error</strong>: Model made mistakes in arithmetic or computation"
        )

    if method == "POT":
        exec_error = result.get("execution_error")
        if exec_error:
            reasons.append(f"🔴 <strong>Code Execution Error</strong>: {exec_error}")
        else:
            reasons.append(
                "🔴 <strong>Logic Error in Code</strong>: Code executed but produced wrong result"
            )

    # Calculate error magnitude
    try:
        error_pct = (
            abs(float(final_answer) - float(ground_truth)) / float(ground_truth) * 100
        )
        reasons.append(
            f"📊 <strong>Error Magnitude</strong>: {error_pct:.2f}% deviation from ground truth"
        )
    except:
        pass

    if not reasons:
        reasons.append(
            "🔴 <strong>Incorrect Answer</strong>: Model produced wrong result"
        )

    html = f"""
    <div class="analysis-wrong">
        <h6>❌ Why This is Wrong:</h6>
        <p><strong>Error Type from Paper:</strong> {error_type}</p>
        <p><strong>Expected Answer:</strong> {ground_truth}</p>
        <p><strong>Model's Answer:</strong> {final_answer}</p>
        <p><strong>What Went Wrong:</strong></p>
        <ul>
            {"".join(f"<li>{reason}</li>" for reason in reasons)}
        </ul>
    </div>
    """

    return html


def format_model_result(result: Dict) -> str:
    """Format a single model's result."""

    model = result["model_name"]
    method = result["method"]
    is_correct = result.get("is_correct", False)

    # Status badge
    if is_correct:
        status_class = "correct-answer"
        status_text = "✓ Correct"
    else:
        status_class = "incorrect-answer"
        status_text = "✗ Incorrect"

    # Full prompt
    system_prompt = escape_html(result.get("system_prompt", "N/A"))
    user_prompt = escape_html(result.get("user_prompt", "N/A"))
    full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

    if len(full_prompt) > 1500:
        full_prompt = full_prompt[:1500] + "...\n[Truncated for display]"

    # Raw response
    raw_response = escape_html(result.get("raw_response", "N/A"))
    if len(raw_response) > 2000:
        raw_response = raw_response[:2000] + "...\n[Truncated for display]"

    # For POT, show code
    pot_section = ""
    if method == "POT":
        code = result.get("raw_response", "")
        # Extract code block
        code_match = re.search(r"```python\s*(.*?)```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1)

        exec_error = result.get("execution_error")

        pot_section = f"""
        <div class="pot-section">
            <h6>Generated Code:</h6>
            <pre class="code-block">{escape_html(code) if code else "No code extracted"}</pre>
            {f'<p class="execution-error"><strong>Execution Error:</strong> {escape_html(exec_error)}</p>' if exec_error else ""}
        </div>
        """

    html = f"""
    <div class="model-result {status_class}">
        <div class="model-header">
            <h5>{model} ({method})</h5>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
        
        <div class="result-body">
            <details class="collapsible">
                <summary>Full Prompt Sent to Model</summary>
                <pre class="prompt-box">{full_prompt}</pre>
            </details>
            
            <details class="collapsible" open>
                <summary>Model's Reasoning Process</summary>
                <pre class="response-box">{raw_response}</pre>
            </details>
            
            {pot_section}
            
            <div class="evaluation">
                <p><strong>Final Answer:</strong> {result.get("final_answer", "None")}</p>
                <p><strong>Ground Truth:</strong> {result.get("ground_truth", "N/A")}</p>
                <p><strong>Correct:</strong> {"Yes" if is_correct else "No"}</p>
            </div>
            
            {analyze_why_wrong_or_correct(result)}
        </div>
    </div>
    """

    return html


def generate_html(results_file: Path, output_file: Path):
    """Generate HTML visualization."""

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    timestamp = data.get("timestamp", "Unknown")
    methodology = data.get("methodology", "Unknown")
    description = data.get("description", "")
    paper_cases = data.get("paper_error_cases", {})
    results = data.get("results", [])

    # Group results by example_id
    grouped_results = {}
    for result in results:
        example_id = result["example_id"]
        if example_id not in grouped_results:
            grouped_results[example_id] = {
                "question": result["question"],
                "ground_truth": result["ground_truth"],
                "error_info": paper_cases.get(example_id, {}),
                "results": [],
            }
        grouped_results[example_id]["results"].append(result)

    # Start HTML
    html_parts = []

    # Header with CSS (reuse from transformation visualization)
    html_parts.append(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Paper Error Cases - Detailed Analysis</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #333;
                background: #f5f5f5;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            h1 {{
                color: #2c3e50;
                margin-bottom: 10px;
            }}
            
            .metadata {{
                background: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
            
            .metadata p {{
                margin: 5px 0;
            }}
            
            .problem-card {{
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 25px;
                margin-bottom: 30px;
                background: #fff;
            }}
            
            .problem-header {{
                background: #3498db;
                color: white;
                padding: 15px;
                margin: -25px -25px 20px -25px;
                border-radius: 6px 6px 0 0;
            }}
            
            .error-info {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
            }}
            
            .error-info h4 {{
                margin-top: 0;
                color: #856404;
            }}
            
            .question-box {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 4px solid #2196f3;
            }}
            
            .model-result {{
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                background: white;
            }}
            
            .model-result.correct-answer {{
                border-color: #4caf50;
                background-color: #f1f8f4;
            }}
            
            .model-result.incorrect-answer {{
                border-color: #f44336;
                background-color: #fef5f5;
            }}
            
            .model-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid #ddd;
            }}
            
            .status-badge {{
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
            }}
            
            .status-badge.correct-answer {{
                background: #4caf50;
                color: white;
            }}
            
            .status-badge.incorrect-answer {{
                background: #f44336;
                color: white;
            }}
            
            .collapsible {{
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            
            .collapsible summary {{
                background: #ecf0f1;
                padding: 10px;
                cursor: pointer;
                font-weight: bold;
                user-select: none;
            }}
            
            .collapsible summary:hover {{
                background: #d5dbdb;
            }}
            
            .collapsible[open] summary {{
                border-bottom: 1px solid #ddd;
            }}
            
            .collapsible > *:not(summary) {{
                padding: 10px;
            }}
            
            .prompt-box, .response-box, .code-block {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.4;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            
            .evaluation {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
            }}
            
            .analysis-wrong {{
                background: #ffebee;
                border-left: 4px solid #f44336;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }}
            
            .analysis-wrong h6 {{
                margin-top: 0;
                color: #c62828;
                font-size: 16px;
            }}
            
            .analysis-wrong ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            
            .analysis-wrong li {{
                margin: 8px 0;
                line-height: 1.6;
            }}
            
            .analysis-correct {{
                background: #e8f5e9;
                border-left: 4px solid #4caf50;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }}
            
            .analysis-correct h6 {{
                margin-top: 0;
                color: #2e7d32;
                font-size: 16px;
            }}
            
            .analysis-correct ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            
            .pot-section {{
                margin: 15px 0;
            }}
            
            .execution-error {{
                color: #d32f2f;
                background: #ffebee;
                padding: 8px;
                border-radius: 4px;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Paper Error Cases - Detailed Analysis</h1>
            
            <div class="metadata">
                <p><strong>Timestamp:</strong> {timestamp}</p>
                <p><strong>Methodology:</strong> {methodology}</p>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Total Problems:</strong> {len(grouped_results)}</p>
                <p><strong>Total Evaluations:</strong> {len(results)}</p>
            </div>
    """)

    # Process each problem
    for example_id, problem_data in grouped_results.items():
        error_info = problem_data["error_info"]

        html_parts.append(f"""
            <div class="problem-card">
                <div class="problem-header">
                    <h2>Problem: {example_id}</h2>
                    <p><strong>Table {error_info.get("table", "N/A")}</strong> - {error_info.get("error_type", "Unknown Error Type")}</p>
                </div>
                
                <div class="error-info">
                    <h4>Error Type from Paper (Appendix B)</h4>
                    <p><strong>Type:</strong> {error_info.get("error_type", "Unknown")}</p>
                    <p><strong>Description:</strong> {error_info.get("description", "N/A")}</p>
                    <p><strong>Source:</strong> {error_info.get("source", "N/A")}</p>
                </div>
                
                <div class="question-box">
                    <h4>Question:</h4>
                    <p>{escape_html(problem_data["question"])}</p>
                    <p><strong>Ground Truth:</strong> {problem_data["ground_truth"]}</p>
                </div>
                
                <h4>Model Results:</h4>
        """)

        # Add each model result
        for result in problem_data["results"]:
            html_parts.append(format_model_result(result))

        html_parts.append("</div>")  # Close problem-card

    # Close HTML
    html_parts.append("""
        </div>
    </body>
    </html>
    """)

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"[SUCCESS] Visualization generated: {output_file}")


if __name__ == "__main__":
    import sys

    # Check if results file provided as argument
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    else:
        # Use most recent paper_error_cases file
        results_dir = Path("experiments/quick_results")
        paper_files = list(results_dir.glob("paper_error_cases_*.json"))
        if paper_files:
            results_file = max(paper_files, key=lambda p: p.stat().st_mtime)
            print(f"Using most recent results: {results_file.name}")
        else:
            print("No paper error cases results found!")
            sys.exit(1)

    output_file = Path("experiments/quick_results/paper_cases_detailed_analysis.html")

    generate_html(results_file, output_file)

    print(f"\nOpen the file in your browser to view:")
    print(f"  {output_file.absolute()}")
