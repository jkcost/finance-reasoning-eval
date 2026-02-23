"""
Model Comparison Visualization

Generates an HTML dashboard comparing multiple LLM models on the same questions.
Shows prompts, reasoning steps, and accuracy comparisons.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def generate_comparison_dashboard(results_path: str) -> str:
    """Generate HTML dashboard from comparison results"""

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    timestamp = data.get("timestamp", "unknown")
    prompts = data.get("prompts", {})
    results = data.get("results", [])
    summary = data.get("summary", {})

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Model Comparison - FinanceReasoning</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; 
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #1a1a2e; margin-bottom: 5px; }}
        h2 {{ color: #16213e; border-bottom: 2px solid #e94560; padding-bottom: 10px; }}
        h3 {{ color: #0f3460; }}
        .header-info {{ color: #666; margin-bottom: 20px; }}
        
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .grid {{ display: grid; gap: 20px; }}
        .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
        .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
        
        @media (max-width: 900px) {{
            .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        
        .correct {{ color: #28a745; font-weight: bold; }}
        .incorrect {{ color: #dc3545; font-weight: bold; }}
        
        .prompt-box {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .response-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            font-family: monospace;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        .steps-list {{
            list-style: none;
            padding: 0;
        }}
        .steps-list li {{
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 3px solid #007bff;
        }}
        .step-type {{
            font-size: 11px;
            color: #666;
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 3px;
            margin-left: 10px;
        }}
        
        .tabs {{
            display: flex;
            border-bottom: 2px solid #ddd;
            margin-bottom: 15px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
        }}
        .tab:hover {{ background: #f5f5f5; }}
        .tab.active {{
            border-bottom-color: #007bff;
            color: #007bff;
            font-weight: bold;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        .model-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
        }}
        .model-claude {{ background: #ffd700; color: #333; }}
        .model-gpt {{ background: #74aa9c; color: white; }}
        .model-gemini {{ background: #4285f4; color: white; }}
        
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #1a1a2e;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
        }}
        
        .collapsible {{
            cursor: pointer;
            padding: 10px;
            background: #f8f9fa;
            border: none;
            width: 100%;
            text-align: left;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .collapsible:hover {{ background: #e9ecef; }}
        .collapsible-content {{
            display: none;
            padding: 10px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 0 0 5px 5px;
        }}
        .collapsible-content.show {{ display: block; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Comparison Dashboard</h1>
        <div class="header-info">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
            Results from: {timestamp}
        </div>
"""

    # Summary Cards
    html += """
        <h2>Summary</h2>
        <div class="grid grid-3">
"""

    # Calculate summary stats by model
    model_stats = {}
    for key, stats in summary.items():
        model = stats["model"]
        if model not in model_stats:
            model_stats[model] = {"COT": None, "POT": None}
        model_stats[model][stats["method"]] = stats

    for model, methods in model_stats.items():
        model_class = (
            "model-claude"
            if "claude" in model
            else ("model-gpt" if "gpt" in model else "model-gemini")
        )
        cot = methods.get("COT", {})
        pot = methods.get("POT", {})

        html += f"""
            <div class="card">
                <span class="model-badge {model_class}">{model}</span>
                <table>
                    <tr>
                        <th>Method</th>
                        <th>Accuracy</th>
                        <th>Avg Cost</th>
                        <th>Avg Latency</th>
                    </tr>
                    <tr>
                        <td>COT</td>
                        <td class="{"correct" if cot.get("accuracy", 0) >= 50 else "incorrect"}">{cot.get("accuracy", 0):.1f}%</td>
                        <td>${cot.get("avg_cost", 0):.4f}</td>
                        <td>{cot.get("avg_latency", 0):.1f}s</td>
                    </tr>
                    <tr>
                        <td>POT</td>
                        <td class="{"correct" if pot.get("accuracy", 0) >= 50 else "incorrect"}">{pot.get("accuracy", 0):.1f}%</td>
                        <td>${pot.get("avg_cost", 0):.4f}</td>
                        <td>{pot.get("avg_latency", 0):.1f}s</td>
                    </tr>
                </table>
            </div>
"""

    html += """
        </div>
"""

    # Prompts Section
    html += """
        <h2>Prompts Used</h2>
        <div class="grid grid-2">
"""

    for method, prompt_info in prompts.items():
        html += f"""
            <div class="card">
                <h3>{method} Prompt</h3>
                <h4>System Message:</h4>
                <div class="prompt-box">{prompt_info.get("system_prompt", "N/A")}</div>
                <h4>User Template:</h4>
                <div class="prompt-box">{prompt_info.get("user_template", "N/A")}</div>
            </div>
"""

    html += """
        </div>
"""

    # Per-Question Results
    html += """
        <h2>Per-Question Analysis</h2>
"""

    for result in results:
        example_id = result.get("example_id", "unknown")
        question = result.get("question", "")
        ground_truth = result.get("ground_truth", "")
        context = result.get("context", "")[:500]

        html += f"""
        <div class="card">
            <h3>Question: {example_id}</h3>
            <p><strong>Question:</strong> {question}</p>
            <p><strong>Ground Truth:</strong> <span class="correct">{ground_truth}</span></p>
            
            <button class="collapsible">Show Context</button>
            <div class="collapsible-content">
                <div class="response-box">{context}...</div>
            </div>
            
            <h4>Model Responses</h4>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Method</th>
                    <th>Answer</th>
                    <th>Correct</th>
                    <th>Reasoning Steps</th>
                    <th>Cost</th>
                    <th>Latency</th>
                </tr>
"""

        for mr in result.get("model_results", []):
            model = mr.get("model_name", "")
            method = mr.get("method", "")
            answer = mr.get("final_answer", "")
            is_correct = mr.get("is_correct", False)
            steps = mr.get("reasoning_steps", [])
            cost = mr.get("cost_usd", 0)
            latency = mr.get("latency_seconds", 0)

            model_class = (
                "model-claude"
                if "claude" in model
                else ("model-gpt" if "gpt" in model else "model-gemini")
            )
            correct_class = "correct" if is_correct else "incorrect"

            html += f"""
                <tr>
                    <td><span class="model-badge {model_class}">{model}</span></td>
                    <td>{method}</td>
                    <td class="{correct_class}">{answer}</td>
                    <td class="{correct_class}">{"Yes" if is_correct else "No"}</td>
                    <td>{len(steps)}</td>
                    <td>${cost:.4f}</td>
                    <td>{latency:.1f}s</td>
                </tr>
"""

        html += """
            </table>
"""

        # Reasoning steps details
        for mr in result.get("model_results", []):
            model = mr.get("model_name", "")
            method = mr.get("method", "")
            steps = mr.get("reasoning_steps", [])
            raw_response = mr.get("raw_response", "")

            if steps:
                html += f"""
            <button class="collapsible">{model} - {method} Reasoning Steps ({len(steps)} steps)</button>
            <div class="collapsible-content">
                <ul class="steps-list">
"""
                for step in steps:
                    html += f"""
                    <li>
                        <strong>Step {step.get("step_number", "?")}</strong>
                        <span class="step-type">{step.get("step_type", "")}</span><br>
                        {step.get("description", "")}<br>
                        <code>{step.get("code_snippet", "")}</code>
                    </li>
"""
                html += """
                </ul>
            </div>
"""

            # Raw response
            html += f"""
            <button class="collapsible">{model} - {method} Raw Response</button>
            <div class="collapsible-content">
                <div class="response-box">{raw_response[:2000]}{"..." if len(raw_response) > 2000 else ""}</div>
            </div>
"""

        html += """
        </div>
"""

    # Charts
    html += """
        <h2>Visualizations</h2>
        <div class="grid grid-2">
            <div class="card">
                <div id="accuracy-chart"></div>
            </div>
            <div class="card">
                <div id="cost-chart"></div>
            </div>
        </div>
        <div class="card">
            <div id="latency-chart"></div>
        </div>
"""

    # JavaScript for charts and collapsibles
    models = list(model_stats.keys())
    cot_accuracies = [model_stats[m].get("COT", {}).get("accuracy", 0) for m in models]
    pot_accuracies = [model_stats[m].get("POT", {}).get("accuracy", 0) for m in models]
    cot_costs = [model_stats[m].get("COT", {}).get("avg_cost", 0) for m in models]
    pot_costs = [model_stats[m].get("POT", {}).get("avg_cost", 0) for m in models]
    cot_latencies = [
        model_stats[m].get("COT", {}).get("avg_latency", 0) for m in models
    ]
    pot_latencies = [
        model_stats[m].get("POT", {}).get("avg_latency", 0) for m in models
    ]

    html += f"""
    <script>
        // Collapsibles
        document.querySelectorAll('.collapsible').forEach(btn => {{
            btn.addEventListener('click', function() {{
                this.classList.toggle('active');
                var content = this.nextElementSibling;
                content.classList.toggle('show');
            }});
        }});
        
        // Accuracy Chart
        var accuracyData = [
            {{
                x: {json.dumps(models)},
                y: {json.dumps(cot_accuracies)},
                name: 'COT',
                type: 'bar',
                marker: {{ color: '#007bff' }}
            }},
            {{
                x: {json.dumps(models)},
                y: {json.dumps(pot_accuracies)},
                name: 'POT',
                type: 'bar',
                marker: {{ color: '#28a745' }}
            }}
        ];
        Plotly.newPlot('accuracy-chart', accuracyData, {{
            title: 'Accuracy by Model & Method',
            yaxis: {{ title: 'Accuracy (%)', range: [0, 100] }},
            barmode: 'group'
        }});
        
        // Cost Chart
        var costData = [
            {{
                x: {json.dumps(models)},
                y: {json.dumps(cot_costs)},
                name: 'COT',
                type: 'bar',
                marker: {{ color: '#007bff' }}
            }},
            {{
                x: {json.dumps(models)},
                y: {json.dumps(pot_costs)},
                name: 'POT',
                type: 'bar',
                marker: {{ color: '#28a745' }}
            }}
        ];
        Plotly.newPlot('cost-chart', costData, {{
            title: 'Average Cost by Model & Method',
            yaxis: {{ title: 'Cost (USD)' }},
            barmode: 'group'
        }});
        
        // Latency Chart
        var latencyData = [
            {{
                x: {json.dumps(models)},
                y: {json.dumps(cot_latencies)},
                name: 'COT',
                type: 'bar',
                marker: {{ color: '#007bff' }}
            }},
            {{
                x: {json.dumps(models)},
                y: {json.dumps(pot_latencies)},
                name: 'POT',
                type: 'bar',
                marker: {{ color: '#28a745' }}
            }}
        ];
        Plotly.newPlot('latency-chart', latencyData, {{
            title: 'Average Latency by Model & Method',
            yaxis: {{ title: 'Latency (seconds)' }},
            barmode: 'group'
        }});
    </script>
"""

    html += """
    </div>
</body>
</html>
"""

    return html


def main():
    if len(sys.argv) < 2:
        # Find most recent comparison result
        results_dir = Path(__file__).parent / "quick_results"
        files = list(results_dir.glob("model_comparison_*.json"))
        if not files:
            print("Usage: python comparison_visualization.py <results.json>")
            print("No comparison results found in quick_results/")
            return
        results_path = max(files, key=lambda p: p.stat().st_mtime)
    else:
        results_path = Path(sys.argv[1])

    print(f"[INFO] Loading results from: {results_path}")

    html = generate_comparison_dashboard(str(results_path))

    # Save HTML
    output_path = results_path.with_suffix(".html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Dashboard saved to: {output_path}")
    print(f"\n[TIP] Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
