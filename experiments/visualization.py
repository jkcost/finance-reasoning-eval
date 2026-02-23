"""
FinanceReasoning Evaluation Visualization

Generates interactive HTML dashboards for visualizing:
1. COT vs POT performance comparison
2. Step-by-step reasoning process
3. Per-example detailed analysis
4. Aggregate metrics and trends

Based on FinanceReasoning paper methodology:
- Accuracy with 0.2% tolerance
- Execution rate tracking
- Step completeness visualization
- Error analysis

Usage:
    python experiments/visualization.py <results_json_file>
    python experiments/visualization.py  # Uses latest results
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import html


def generate_html_dashboard(results: Dict[str, Any], output_path: Path) -> Path:
    """
    Generate interactive HTML dashboard from evaluation results.

    Features:
    - COT vs POT comparison charts
    - Per-example detail cards
    - Reasoning step visualization
    - Metrics summary tables
    """

    timestamp = results.get("timestamp", datetime.now().isoformat())
    model_id = results.get("model_id", "Unknown")
    num_examples = results.get("num_examples", 0)
    examples = results.get("examples", [])
    metrics = results.get("aggregate_metrics", {})

    # Calculate summary stats
    cot_correct = sum(1 for ex in examples if ex.get("cot", {}).get("correct", False))
    pot_correct = sum(1 for ex in examples if ex.get("pot", {}).get("correct", False))

    cot_accuracy = (cot_correct / num_examples * 100) if num_examples > 0 else 0
    pot_accuracy = (pot_correct / num_examples * 100) if num_examples > 0 else 0

    total_cost = metrics.get("total_cost_usd", 0)
    total_time = metrics.get("total_time_seconds", 0)

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinanceReasoning Evaluation Results</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --error-color: #dc2626;
            --warning-color: #d97706;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 8px;
        }}
        
        .header-meta {{
            display: flex;
            gap: 24px;
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .metric-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border-color);
        }}
        
        .metric-card .label {{
            font-size: 13px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .metric-card .value {{
            font-size: 32px;
            font-weight: 700;
            color: var(--primary-color);
        }}
        
        .metric-card .value.success {{
            color: var(--success-color);
        }}
        
        .metric-card .value.error {{
            color: var(--error-color);
        }}
        
        .metric-card .subtitle {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        
        .section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border-color);
        }}
        
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--border-color);
        }}
        
        .chart-container {{
            width: 100%;
            height: 400px;
        }}
        
        .example-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            border: 1px solid var(--border-color);
            transition: box-shadow 0.2s;
        }}
        
        .example-card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}
        
        .example-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }}
        
        .example-id {{
            font-weight: 600;
            font-size: 16px;
        }}
        
        .example-badges {{
            display: flex;
            gap: 8px;
        }}
        
        .badge {{
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .badge.success {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .badge.error {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .badge.cot {{
            background: #dbeafe;
            color: #1e40af;
        }}
        
        .badge.pot {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .question-text {{
            font-size: 14px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            padding: 12px;
            background: #f1f5f9;
            border-radius: 8px;
        }}
        
        .results-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 12px;
        }}
        
        .result-column {{
            padding: 16px;
            border-radius: 8px;
        }}
        
        .result-column.cot {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
        }}
        
        .result-column.pot {{
            background: #fffbeb;
            border: 1px solid #fde68a;
        }}
        
        .result-column h4 {{
            font-size: 14px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .answer-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }}
        
        .answer-row:last-child {{
            border-bottom: none;
        }}
        
        .answer-label {{
            font-size: 13px;
            color: var(--text-secondary);
        }}
        
        .answer-value {{
            font-weight: 600;
            font-size: 14px;
        }}
        
        .steps-container {{
            margin-top: 12px;
        }}
        
        .step {{
            display: flex;
            padding: 8px 12px;
            background: white;
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 13px;
            border-left: 3px solid var(--primary-color);
        }}
        
        .step-number {{
            font-weight: 600;
            margin-right: 8px;
            color: var(--primary-color);
        }}
        
        .step-type {{
            background: #e2e8f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            margin-right: 8px;
        }}
        
        .code-block {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            overflow-x: auto;
            margin-top: 8px;
            white-space: pre-wrap;
        }}
        
        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}
        
        .collapsible::before {{
            content: '▶ ';
            font-size: 10px;
            transition: transform 0.2s;
        }}
        
        .collapsible.active::before {{
            content: '▼ ';
        }}
        
        .collapsible-content {{
            display: none;
            padding-top: 12px;
        }}
        
        .collapsible-content.show {{
            display: block;
        }}
        
        .methodology-note {{
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
            font-size: 14px;
        }}
        
        .methodology-note h3 {{
            color: #166534;
            margin-bottom: 8px;
        }}
        
        .formula {{
            font-family: 'Monaco', 'Menlo', monospace;
            background: #ecfdf5;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        
        @media (max-width: 768px) {{
            .results-comparison {{
                grid-template-columns: 1fr;
            }}
            
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FinanceReasoning Evaluation Results</h1>
            <div class="header-meta">
                <span>Model: {html.escape(model_id)}</span>
                <span>Examples: {num_examples}</span>
                <span>Generated: {timestamp[:19]}</span>
            </div>
        </div>
        
        <div class="methodology-note">
            <h3>Paper Methodology</h3>
            <p>Evaluation follows FinanceReasoning (ACL 2025) methodology:</p>
            <ul style="margin-top: 8px; margin-left: 20px;">
                <li><strong>Accuracy</strong>: Correct answers within <span class="formula">0.2% tolerance</span></li>
                <li><strong>Overall Score</strong>: <span class="formula">(Accuracy × 0.4) + (Step Completeness × 0.3) + (Order × 0.1) + ((1 - Hallucination) × 0.2)</span></li>
            </ul>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">COT Accuracy</div>
                <div class="value {"success" if cot_accuracy >= 50 else "error"}">{cot_accuracy:.1f}%</div>
                <div class="subtitle">{cot_correct}/{num_examples} correct</div>
            </div>
            <div class="metric-card">
                <div class="label">POT Accuracy</div>
                <div class="value {"success" if pot_accuracy >= 50 else "error"}">{pot_accuracy:.1f}%</div>
                <div class="subtitle">{pot_correct}/{num_examples} correct</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Cost</div>
                <div class="value">${total_cost:.4f}</div>
                <div class="subtitle">API usage</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Time</div>
                <div class="value">{total_time:.1f}s</div>
                <div class="subtitle">Execution time</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">COT vs POT Performance Comparison</h2>
            <div id="comparison-chart" class="chart-container"></div>
        </div>
        
        <div class="section">
            <h2 class="section-title">Detailed Metrics</h2>
            <div id="metrics-chart" class="chart-container"></div>
        </div>
        
        <div class="section">
            <h2 class="section-title">Per-Example Results ({num_examples} examples)</h2>
            <div id="examples-container">
"""

    # Add example cards
    for i, ex in enumerate(examples):
        example_id = ex.get("example_id", f"Example {i + 1}")
        question = ex.get("question", "No question")[:200]
        ground_truth = ex.get("ground_truth", "N/A")

        cot = ex.get("cot", {})
        pot = ex.get("pot", {})

        cot_correct = cot.get("correct", False)
        pot_correct = pot.get("correct", False)
        cot_answer = cot.get("answer", "N/A")
        pot_answer = pot.get("answer", "N/A")

        cot_badge = "success" if cot_correct else "error"
        pot_badge = "success" if pot_correct else "error"

        cot_steps = cot.get("steps", [])
        cot_metrics = cot.get("metrics", {})
        pot_code = pot.get("raw_response", "")

        # Extract code from POT response
        if "```python" in pot_code:
            code_start = pot_code.find("```python") + 9
            code_end = pot_code.find("```", code_start)
            pot_code_display = (
                pot_code[code_start:code_end].strip()
                if code_end > code_start
                else pot_code[:500]
            )
        else:
            pot_code_display = pot_code[:500] if pot_code else "No code generated"

        html_content += f"""
                <div class="example-card">
                    <div class="example-header">
                        <span class="example-id">{html.escape(str(example_id))}</span>
                        <div class="example-badges">
                            <span class="badge cot {cot_badge}">COT: {"✓" if cot_correct else "✗"}</span>
                            <span class="badge pot {pot_badge}">POT: {"✓" if pot_correct else "✗"}</span>
                        </div>
                    </div>
                    
                    <div class="question-text">
                        <strong>Question:</strong> {html.escape(question)}{"..." if len(question) >= 200 else ""}
                    </div>
                    
                    <div class="answer-row">
                        <span class="answer-label">Ground Truth:</span>
                        <span class="answer-value">{html.escape(str(ground_truth))}</span>
                    </div>
                    
                    <div class="results-comparison">
                        <div class="result-column cot">
                            <h4><span class="badge cot">COT</span> Chain-of-Thought</h4>
                            <div class="answer-row">
                                <span class="answer-label">Answer:</span>
                                <span class="answer-value">{html.escape(str(cot_answer))}</span>
                            </div>
                            <div class="answer-row">
                                <span class="answer-label">Correct:</span>
                                <span class="answer-value">{"Yes ✓" if cot_correct else "No ✗"}</span>
                            </div>
                            <div class="answer-row">
                                <span class="answer-label">Overall Score:</span>
                                <span class="answer-value">{(cot_metrics or {}).get("overall_score", 0):.3f}</span>
                            </div>
                            <div class="answer-row">
                                <span class="answer-label">Step Completeness:</span>
                                <span class="answer-value">{(cot_metrics or {}).get("step_completeness", 0) * 100:.1f}%</span>
                            </div>
"""

        # Add reasoning steps if available
        if cot_steps:
            html_content += f"""
                            <div class="steps-container">
                                <span class="collapsible" onclick="toggleCollapsible(this)">View {len(cot_steps)} Reasoning Steps</span>
                                <div class="collapsible-content">
"""
            for step in cot_steps[:5]:  # Limit to 5 steps
                step_num = step.get("step_number", "?")
                step_type = step.get("step_type", "unknown")
                step_desc = step.get("description", "")[:100]
                html_content += f"""
                                    <div class="step">
                                        <span class="step-number">#{step_num}</span>
                                        <span class="step-type">{html.escape(step_type)}</span>
                                        {html.escape(step_desc)}
                                    </div>
"""
            html_content += """
                                </div>
                            </div>
"""

        html_content += f"""
                        </div>
                        
                        <div class="result-column pot">
                            <h4><span class="badge pot">POT</span> Program-of-Thought</h4>
                            <div class="answer-row">
                                <span class="answer-label">Answer:</span>
                                <span class="answer-value">{html.escape(str(pot_answer))}</span>
                            </div>
                            <div class="answer-row">
                                <span class="answer-label">Correct:</span>
                                <span class="answer-value">{"Yes ✓" if pot_correct else "No ✗"}</span>
                            </div>
                            <div class="answer-row">
                                <span class="answer-label">Execution:</span>
                                <span class="answer-value">{"Success ✓" if pot.get("execution_success", False) else "Failed ✗"}</span>
                            </div>
                            <span class="collapsible" onclick="toggleCollapsible(this)">View Generated Code</span>
                            <div class="collapsible-content">
                                <div class="code-block">{html.escape(pot_code_display)}</div>
                            </div>
                        </div>
                    </div>
                </div>
"""

    # Close examples container and add charts script
    html_content += f"""
            </div>
        </div>
    </div>
    
    <script>
        // Toggle collapsible sections
        function toggleCollapsible(element) {{
            element.classList.toggle('active');
            const content = element.nextElementSibling;
            content.classList.toggle('show');
        }}
        
        // Data for charts
        const examples = {json.dumps([ex.get("example_id", f"Ex{i}") for i, ex in enumerate(examples)])};
        const cotCorrect = {json.dumps([1 if ex.get("cot", {}).get("correct", False) else 0 for ex in examples])};
        const potCorrect = {json.dumps([1 if ex.get("pot", {}).get("correct", False) else 0 for ex in examples])};
        
        // Comparison Chart
        const comparisonTrace1 = {{
            x: examples,
            y: cotCorrect,
            type: 'bar',
            name: 'COT',
            marker: {{ color: '#3b82f6' }}
        }};
        
        const comparisonTrace2 = {{
            x: examples,
            y: potCorrect,
            type: 'bar',
            name: 'POT',
            marker: {{ color: '#f59e0b' }}
        }};
        
        const comparisonLayout = {{
            title: 'Correctness by Example',
            barmode: 'group',
            yaxis: {{ 
                title: 'Correct (1) / Incorrect (0)',
                tickvals: [0, 1],
                ticktext: ['Incorrect', 'Correct']
            }},
            xaxis: {{ title: 'Example ID' }},
            legend: {{ orientation: 'h', y: 1.1 }}
        }};
        
        Plotly.newPlot('comparison-chart', [comparisonTrace1, comparisonTrace2], comparisonLayout, {{responsive: true}});
        
        // Metrics Summary Chart
        const metricsData = {{
            'COT Accuracy': {cot_accuracy:.1f},
            'POT Accuracy': {pot_accuracy:.1f},
            'COT Overall Score': {metrics.get("cot_avg_overall_score", 0) * 100:.1f},
            'Step Completeness': {metrics.get("cot_avg_step_completeness", 0) * 100:.1f},
        }};
        
        const metricsTrace = {{
            type: 'bar',
            x: Object.keys(metricsData),
            y: Object.values(metricsData),
            marker: {{
                color: ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6']
            }},
            text: Object.values(metricsData).map(v => v.toFixed(1) + '%'),
            textposition: 'outside'
        }};
        
        const metricsLayout = {{
            title: 'Aggregate Metrics (%)',
            yaxis: {{ title: 'Percentage', range: [0, 100] }},
            showlegend: false
        }};
        
        Plotly.newPlot('metrics-chart', [metricsTrace], metricsLayout, {{responsive: true}});
    </script>
</body>
</html>
"""

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def find_latest_results(results_dir: Path) -> Optional[Path]:
    """Find the most recent results file"""
    results_files = list(results_dir.glob("quick_eval_*.json"))
    if not results_files:
        return None
    return max(results_files, key=lambda p: p.stat().st_mtime)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate visualization dashboard from evaluation results"
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        help="Path to results JSON file (uses latest if not specified)",
    )
    parser.add_argument("--output", "-o", help="Output HTML file path")
    args = parser.parse_args()

    # Find results file
    results_dir = Path(__file__).parent / "quick_results"

    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = find_latest_results(results_dir)
        if not results_path:
            print("[ERROR] No results files found. Run quick_evaluation.py first.")
            sys.exit(1)

    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        sys.exit(1)

    print(f"[INFO] Loading results from: {results_path}")

    # Load results
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_path.with_suffix(".html")

    # Generate dashboard
    print(f"[INFO] Generating dashboard...")
    generate_html_dashboard(results, output_path)

    print(f"[OK] Dashboard saved to: {output_path}")
    print(f"\n[TIP] Open in browser: file://{output_path.absolute()}")

    return output_path


if __name__ == "__main__":
    main()
