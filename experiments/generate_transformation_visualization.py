"""
Generate HTML visualization for transformation evaluation results.

Shows:
1. Original vs Transformed comparison (with highlights)
2. Full prompts sent to models
3. Model reasoning processes
4. Evaluation results
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def analyze_transformation_changes(
    original: str, transformed: str, trans_type: str
) -> Dict:
    """
    Analyze what changed between original and transformed.

    Returns: {
        'changes': [list of change descriptions],
        'removed_items': [list of removed items],
        'modified_items': [list of modified items]
    }
    """
    changes = []
    removed_items = []
    modified_items = []

    try:
        import json as json_lib

        orig_json = json_lib.loads(original)
        trans_json = json_lib.loads(transformed)

        # Find removed keys/values
        def find_changes(orig, trans, path=""):
            if isinstance(orig, dict) and isinstance(trans, dict):
                for key in orig:
                    current_path = f"{path}.{key}" if path else key

                    if key not in trans:
                        removed_items.append(f"Entire key '{key}' removed")
                        changes.append(f"🔴 Removed: {current_path}")
                    elif isinstance(orig[key], dict) and isinstance(trans[key], dict):
                        # Check nested dict
                        orig_subkeys = set(orig[key].keys())
                        trans_subkeys = set(trans[key].keys())

                        removed_subkeys = orig_subkeys - trans_subkeys
                        for subkey in removed_subkeys:
                            removed_items.append(
                                f"{key}['{subkey}'] = {orig[key][subkey]}"
                            )
                            changes.append(f"🔴 Removed: {current_path}['{subkey}']")

                        # Recurse
                        find_changes(orig[key], trans[key], current_path)
                    elif orig[key] != trans[key]:
                        modified_items.append(f"{key}: {orig[key]} → {trans[key]}")
                        changes.append(f"🟡 Modified: {current_path}")

        find_changes(orig_json, trans_json)

    except:
        # Not JSON - try simple text diff
        if original != transformed:
            changes.append("Text content modified")

    return {
        "changes": changes,
        "removed_items": removed_items,
        "modified_items": modified_items,
    }


def highlight_diff(original: str, transformed: str, changes_info: Dict) -> tuple:
    """
    Highlight differences between original and transformed text.

    Returns: (highlighted_original, highlighted_transformed)
    """
    if original == transformed:
        return escape_html(original), escape_html(transformed)

    # Try to parse as JSON and highlight differences
    try:
        import json as json_lib

        orig_json = json_lib.loads(original)
        trans_json = json_lib.loads(transformed)

        # Pretty print for better readability
        orig_pretty = json_lib.dumps(orig_json, indent=2, ensure_ascii=False)
        trans_pretty = json_lib.dumps(trans_json, indent=2, ensure_ascii=False)

        # Highlight removed items
        orig_highlighted = escape_html(orig_pretty)
        trans_highlighted = escape_html(trans_pretty)

        # Highlight removed keys in original
        for removed in changes_info.get("removed_items", []):
            # Extract key name
            if "[" in removed:
                key = removed.split("[")[0]
                subkey = removed.split("'")[1] if "'" in removed else None

                if subkey:
                    # Highlight the specific subkey line
                    pattern = f'"{subkey}"'
                    orig_highlighted = orig_highlighted.replace(
                        pattern,
                        f'<span class="removed-data" title="This data was removed">{pattern}</span>',
                        1,
                    )
            elif "Entire key" in removed:
                key = removed.split("'")[1]
                # Highlight entire key section
                pattern = f'"{key}"'
                orig_highlighted = orig_highlighted.replace(
                    pattern,
                    f'<span class="removed-data" title="This entire key was removed">{pattern}</span>',
                    1,
                )

        return orig_highlighted, trans_highlighted

    except:
        # Not JSON, just escape
        return escape_html(original), escape_html(transformed)

    # Try to parse as JSON and highlight differences
    try:
        import json as json_lib

        orig_json = json_lib.loads(original)
        trans_json = json_lib.loads(transformed)

        # Find removed keys
        removed_keys = set()

        def find_removed(orig, trans, path=""):
            if isinstance(orig, dict) and isinstance(trans, dict):
                for key in orig:
                    if key not in trans:
                        removed_keys.add(f"{path}.{key}" if path else key)
                    elif isinstance(orig[key], dict):
                        find_removed(
                            orig[key],
                            trans.get(key, {}),
                            f"{path}.{key}" if path else key,
                        )

        find_removed(orig_json, trans_json)

        # Highlight removed parts in original
        orig_highlighted = escape_html(original)
        trans_highlighted = escape_html(transformed)

        for key in removed_keys:
            # Simple highlighting - mark the key
            orig_highlighted = orig_highlighted.replace(
                f'"{key}"', f'<span class="removed-data">"{key}"</span>'
            )

        return orig_highlighted, trans_highlighted

    except:
        # Not JSON, just escape
        return escape_html(original), escape_html(transformed)


def format_context_comparison(
    original_context: str, transformed_context: str, trans_info: Dict
) -> str:
    """Format side-by-side context comparison with change highlights."""

    # Analyze changes
    changes_info = analyze_transformation_changes(
        original_context, transformed_context, trans_info["type"]
    )

    # Highlight differences
    orig_highlighted, trans_highlighted = highlight_diff(
        original_context, transformed_context, changes_info
    )

    # Truncate if too long
    max_len = 800
    if len(orig_highlighted) > max_len:
        orig_highlighted = orig_highlighted[:max_len] + "...\n[Truncated for display]"
    if len(trans_highlighted) > max_len:
        trans_highlighted = trans_highlighted[:max_len] + "...\n[Truncated for display]"

    # Build change summary
    change_summary = ""
    if changes_info["changes"]:
        change_summary = f"""
        <div class="change-summary">
            <h6>🔍 What Changed:</h6>
            <ul>
                {"".join(f"<li>{change}</li>" for change in changes_info["changes"][:5])}
            </ul>
        </div>
        """

    if changes_info["removed_items"]:
        change_summary += f"""
        <div class="removed-items-list">
            <h6>🗑️ Removed Data:</h6>
            <ul>
                {"".join(f"<li><code>{item}</code></li>" for item in changes_info["removed_items"][:5])}
            </ul>
        </div>
        """

    html = f"""
    <div class="context-comparison">
        {change_summary}
        <div class="comparison-row">
            <div class="comparison-col">
                <h5>Original Context</h5>
                <pre class="context-box">{orig_highlighted}</pre>
            </div>
            <div class="comparison-col">
                <h5>Transformed Context</h5>
                <pre class="context-box">{trans_highlighted}</pre>
            </div>
        </div>
    </div>
    """

    return html


def format_question_comparison(original_q: str, transformed_q: str) -> str:
    """Format question comparison."""

    if original_q == transformed_q:
        return f"""
        <div class="question-comparison">
            <p><strong>Question:</strong> {escape_html(original_q)}</p>
            <p class="note">(Question unchanged)</p>
        </div>
        """

    return f"""
    <div class="question-comparison">
        <div class="comparison-row">
            <div class="comparison-col">
                <h5>Original Question</h5>
                <p>{escape_html(original_q)}</p>
            </div>
            <div class="comparison-col">
                <h5>Transformed Question</h5>
                <p class="changed-data">{escape_html(transformed_q)}</p>
            </div>
        </div>
    </div>
    """


def analyze_why_wrong(result: Dict, trans_info: Dict) -> str:
    """
    Analyze why the model got it wrong.

    Returns HTML explanation.
    """
    if result.get("is_correct"):
        return """
        <div class="analysis-correct">
            <h6>✅ Why This is Correct:</h6>
            <p>The model correctly recognized that the data is insufficient to answer the question and either:</p>
            <ul>
                <li>Explicitly stated it cannot answer, OR</li>
                <li>Produced an error/exception indicating missing data, OR</li>
                <li>Returned a result that matches the expected "refusal" behavior</li>
            </ul>
        </div>
        """

    # Analyze the wrong answer
    raw_response = result.get("raw_response", "").lower()
    extracted_answer = result.get("extracted_answer")
    method = result.get("method")

    reasons = []

    # Check for hallucination patterns
    if "assume" in raw_response or "assuming" in raw_response:
        reasons.append(
            "🔴 <strong>Assumption Hallucination</strong>: Model explicitly assumed data that doesn't exist (e.g., 'Assume USD-INR = 73.5')"
        )

    if extracted_answer is not None and extracted_answer != 0:
        reasons.append(
            f"🔴 <strong>Forced Answer</strong>: Model provided a numeric answer ({extracted_answer}) despite insufficient data"
        )

    if (
        "not determinable" in raw_response
        or "cannot" in raw_response
        or "insufficient" in raw_response
    ):
        if extracted_answer is not None:
            reasons.append(
                "🔴 <strong>Contradictory Behavior</strong>: Model recognized the problem ('not determinable') but still provided a numeric answer"
            )

    if method == "POT":
        code = result.get("executed_code", "")
        if "assume" in code.lower():
            reasons.append(
                "🔴 <strong>Code-level Hallucination</strong>: Generated code with assumed/hardcoded values not present in context"
            )

        exec_error = result.get("execution_error")
        if exec_error and "KeyError" not in str(exec_error):
            reasons.append(
                f"🟡 <strong>Execution Error</strong>: Code failed to run ({exec_error}), but not due to missing data"
            )

    if not reasons:
        reasons.append(
            "🔴 <strong>General Hallucination</strong>: Model generated an answer without recognizing missing data"
        )

    # Build explanation
    html = f"""
    <div class="analysis-wrong">
        <h6>❌ Why This is Wrong:</h6>
        <p><strong>Expected Behavior:</strong> {escape_html(trans_info["expected_behavior"])}</p>
        <p><strong>What Happened:</strong></p>
        <ul>
            {"".join(f"<li>{reason}</li>" for reason in reasons)}
        </ul>
    </div>
    """

    return html


def format_model_result(result: Dict, trans_info: Dict) -> str:
    """Format a single model's result."""

    model = result["model"]
    method = result["method"]
    is_correct = result.get("is_correct", False)

    # Status badge
    if is_correct:
        status_class = "correct-refusal"
        status_text = "✓ Correct Refusal"
        status_desc = "Model correctly recognized insufficient data"
    else:
        status_class = "incorrect-hallucination"
        status_text = "✗ Hallucination"
        status_desc = "Model forced an answer despite missing data"

    # Full prompt
    full_prompt = escape_html(result.get("full_prompt", "N/A"))
    if len(full_prompt) > 1000:
        full_prompt = full_prompt[:1000] + "..."

    # Raw response
    raw_response = escape_html(result.get("raw_response", "N/A"))

    # For POT, show code
    pot_section = ""
    if method == "POT":
        code = result.get("executed_code", "N/A")
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
            <div class="section">
                <h6>Expected Behavior:</h6>
                <p class="expected-behavior">{escape_html(trans_info["expected_behavior"])}</p>
            </div>
            
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
                <p class="status-description">{status_desc}</p>
                <p><strong>Extracted Answer:</strong> {result.get("extracted_answer", "None")}</p>
                <p><strong>Ground Truth:</strong> {result.get("ground_truth", "N/A")}</p>
            </div>
            
            {analyze_why_wrong(result, trans_info)}
        </div>
    </div>
    """

    return html


def generate_html(results_file: Path, output_file: Path):
    """Generate HTML visualization."""

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    problems = data["problems"]

    # Start HTML
    html_parts = []

    # Header
    html_parts.append(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transformation Evaluation Results</title>
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
            
            .transformation-card {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                background: #fafafa;
            }}
            
            .transformation-header {{
                background: #34495e;
                color: white;
                padding: 10px 15px;
                margin: -20px -20px 15px -20px;
                border-radius: 4px 4px 0 0;
            }}
            
            .transformation-info {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 10px;
                margin-bottom: 15px;
            }}
            
            .comparison-row {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 15px 0;
            }}
            
            .comparison-col {{
                background: white;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }}
            
            .comparison-col h5 {{
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            
            .context-box, .prompt-box, .response-box, .code-block {{
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
            
            .removed-data {{
                background-color: #ffebee;
                text-decoration: line-through;
                color: #c62828;
                padding: 2px 4px;
                border-radius: 3px;
                font-weight: bold;
                border: 1px dashed #c62828;
            }}
            
            .change-summary {{
                background: #fff3cd;
                border-left: 4px solid #ff9800;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }}
            
            .change-summary h6 {{
                margin-top: 0;
                color: #e65100;
            }}
            
            .change-summary ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            
            .removed-items-list {{
                background: #ffebee;
                border-left: 4px solid #c62828;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }}
            
            .removed-items-list h6 {{
                margin-top: 0;
                color: #c62828;
            }}
            
            .removed-items-list code {{
                background: #fff;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
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
            
            .changed-data {{
                background-color: #fff9c4;
                color: #f57f17;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            
            .model-result {{
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                background: white;
            }}
            
            .model-result.correct-refusal {{
                border-color: #4caf50;
                background-color: #f1f8f4;
            }}
            
            .model-result.incorrect-hallucination {{
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
            
            .status-badge.correct-refusal {{
                background: #4caf50;
                color: white;
            }}
            
            .status-badge.incorrect-hallucination {{
                background: #f44336;
                color: white;
            }}
            
            .section {{
                margin: 15px 0;
            }}
            
            .expected-behavior {{
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 10px;
                font-style: italic;
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
            
            .evaluation {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
            }}
            
            .status-description {{
                font-weight: bold;
                margin-bottom: 5px;
            }}
            
            .execution-error {{
                color: #d32f2f;
                background: #ffebee;
                padding: 8px;
                border-radius: 4px;
                margin-top: 10px;
            }}
            
            .note {{
                color: #666;
                font-style: italic;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Transformation Evaluation Results</h1>
            
            <div class="metadata">
                <p><strong>Generated:</strong> {metadata["timestamp"]}</p>
                <p><strong>Models:</strong> {", ".join(metadata["models"])}</p>
                <p><strong>Methods:</strong> {", ".join(metadata["methods"])}</p>
                <p><strong>Sample Size:</strong> {metadata.get("sample_size", "N/A")}</p>
                {f'<p class="note"><strong>Note:</strong> {metadata["note"]}</p>' if "note" in metadata else ""}
            </div>
    """)

    # Process each problem
    for prob_idx, problem in enumerate(problems):
        original = problem["original"]
        transformations = problem["transformations"]

        html_parts.append(f"""
            <div class="problem-card">
                <div class="problem-header">
                    <h2>Problem {prob_idx + 1}: {original["question_id"]}</h2>
                    <p><strong>Level:</strong> {original.get("level", "N/A").upper()}</p>
                </div>
                
                <div class="original-problem">
                    <h4>Original Problem</h4>
                    <p><strong>Question:</strong> {escape_html(original["question"])}</p>
                    <p><strong>Ground Truth:</strong> {original["ground_truth"]}</p>
                </div>
        """)

        # Process each transformation
        for trans_idx, trans in enumerate(transformations):
            trans_info = trans["transformation_info"]
            trans_problem = trans["transformed_problem"]
            model_results = trans["model_results"]

            html_parts.append(f"""
                <div class="transformation-card">
                    <div class="transformation-header">
                        <h3>Transformation {trans_idx + 1}: {trans_info["type"]}</h3>
                    </div>
                    
                    <div class="transformation-info">
                        <p><strong>Description:</strong> {escape_html(trans_info["description"])}</p>
                        <p><strong>Expected Behavior:</strong> {escape_html(trans_info["expected_behavior"])}</p>
                    </div>
                    
                    <h4>Data Comparison</h4>
                    {format_question_comparison(original["question"], trans_problem["question"])}
                    {format_context_comparison(original["context"], trans_problem["context"], trans_info)}
                    
                    <h4>Model Results</h4>
            """)

            # Add each model result
            for result in model_results:
                html_parts.append(format_model_result(result, trans_info))

            html_parts.append("</div>")  # Close transformation-card

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
        # Use most recent real_experiment file
        results_dir = Path("experiments/transformation_results")
        real_files = list(results_dir.glob("real_experiment_*.json"))
        if real_files:
            results_file = max(real_files, key=lambda p: p.stat().st_mtime)
            print(f"Using most recent results: {results_file.name}")
        else:
            results_file = Path("experiments/transformation_results/mock_results.json")
            print(f"No real results found, using mock data")

    output_file = Path(
        "experiments/transformation_results/transformation_visualization.html"
    )

    generate_html(results_file, output_file)

    print(f"\nOpen the file in your browser to view:")
    print(f"  {output_file.absolute()}")
