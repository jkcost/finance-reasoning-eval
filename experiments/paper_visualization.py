"""
Paper Experiment Visualization Dashboard

Generates an HTML dashboard for FinanceReasoning paper-compliant experiment results.
Shows COT vs POT comparison across multiple models with accuracy, cost, and latency charts.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def generate_paper_dashboard(results_path: str) -> str:
    """Generate HTML dashboard from paper experiment results"""

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    timestamp = data.get("timestamp", "unknown")
    prompts = data.get("prompts", {})
    results = data.get("results", [])
    summary = data.get("summary", {})
    methodology = data.get("methodology", "FinanceReasoning Paper")

    # Group results by example
    by_example = {}
    for r in results:
        ex_id = r.get("example_id", "unknown")
        if ex_id not in by_example:
            by_example[ex_id] = {
                "question": r.get("question", ""),
                "ground_truth": r.get("ground_truth", ""),
                "results": [],
            }
        by_example[ex_id]["results"].append(r)

    # Calculate summary stats by model
    model_stats = {}
    for key, stats in summary.items():
        model = stats["model"]
        if model not in model_stats:
            model_stats[model] = {"COT": None, "POT": None}
        model_stats[model][stats["method"]] = stats

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FinanceReasoning Paper Experiment Results</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        
        h1 {{ 
            color: white; 
            margin-bottom: 5px; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        h2 {{ 
            color: #1a1a2e; 
            border-bottom: 3px solid #e94560; 
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        h3 {{ color: #0f3460; }}
        .header-info {{ 
            color: rgba(255,255,255,0.9); 
            margin-bottom: 20px;
            font-size: 14px;
        }}
        
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        
        .grid {{ display: grid; gap: 20px; }}
        .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
        .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
        .grid-6 {{ grid-template-columns: repeat(6, 1fr); }}
        
        @media (max-width: 1200px) {{
            .grid-6 {{ grid-template-columns: repeat(3, 1fr); }}
        }}
        @media (max-width: 900px) {{
            .grid-2, .grid-3, .grid-6 {{ grid-template-columns: 1fr; }}
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background: #f8f9ff; }}
        
        .correct {{ color: #28a745; font-weight: bold; }}
        .incorrect {{ color: #dc3545; font-weight: bold; }}
        .neutral {{ color: #6c757d; }}
        
        .prompt-box {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
            line-height: 1.5;
        }}
        
        .model-card {{
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .model-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }}
        
        .model-name {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
            padding: 5px 10px;
            border-radius: 20px;
            display: inline-block;
        }}
        .model-openai {{ background: #74aa9c; color: white; }}
        .model-anthropic {{ background: #d4a574; color: white; }}
        .model-google {{ background: #4285f4; color: white; }}
        
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-sub {{
            font-size: 14px;
            color: #888;
            margin-top: 5px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            margin: 2px;
        }}
        .badge-cot {{ background: #007bff; color: white; }}
        .badge-pot {{ background: #28a745; color: white; }}
        .badge-correct {{ background: #d4edda; color: #155724; }}
        .badge-incorrect {{ background: #f8d7da; color: #721c24; }}
        
        .example-card {{
            border-left: 4px solid #667eea;
            padding-left: 20px;
            margin-bottom: 30px;
        }}
        .example-id {{
            font-size: 12px;
            color: #667eea;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .example-question {{
            font-size: 16px;
            color: #333;
            margin: 10px 0;
            line-height: 1.5;
        }}
        .example-gt {{
            background: #e8f5e9;
            padding: 8px 15px;
            border-radius: 8px;
            display: inline-block;
            color: #2e7d32;
            font-weight: bold;
        }}
        
        .response-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .response-item {{
            padding: 12px;
            border-radius: 8px;
            background: #f8f9fa;
            text-align: center;
        }}
        .response-model {{
            font-size: 11px;
            color: #666;
            margin-bottom: 5px;
        }}
        .response-answer {{
            font-size: 18px;
            font-weight: bold;
        }}
        
        .collapsible {{
            cursor: pointer;
            padding: 12px 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: none;
            width: 100%;
            text-align: left;
            border-radius: 8px;
            margin-top: 10px;
            font-weight: 500;
            transition: background 0.2s;
        }}
        .collapsible:hover {{ 
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        }}
        .collapsible::after {{
            content: '\\25BC';
            float: right;
            transition: transform 0.2s;
        }}
        .collapsible.active::after {{
            transform: rotate(180deg);
        }}
        .collapsible-content {{
            display: none;
            padding: 15px;
            background: #fff;
            border: 1px solid #e9ecef;
            border-radius: 0 0 8px 8px;
            margin-top: -5px;
        }}
        .collapsible-content.show {{ display: block; }}
        
        .winner-badge {{
            background: linear-gradient(135deg, #ffd700 0%, #ffb347 100%);
            color: #333;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 10px;
            font-weight: bold;
            margin-left: 5px;
        }}
        
        .section-divider {{
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            margin: 40px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FinanceReasoning Paper Experiment Results</h1>
        <div class="header-info">
            <strong>Methodology:</strong> {methodology} | 
            <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
            <strong>Experiment:</strong> {timestamp} |
            <strong>Examples:</strong> {len(by_example)} |
            <strong>Models:</strong> {len(model_stats)}
        </div>
"""

    # Model Summary Cards
    html += """
        <div class="card">
            <h2>Model Performance Summary</h2>
            <div class="grid grid-6">
"""

    # Find best performers
    best_cot = max(
        model_stats.items(),
        key=lambda x: x[1].get("COT", {}).get("accuracy", 0) if x[1].get("COT") else 0,
    )
    best_pot = max(
        model_stats.items(),
        key=lambda x: x[1].get("POT", {}).get("accuracy", 0) if x[1].get("POT") else 0,
    )

    for model, methods in sorted(model_stats.items()):
        provider = (
            "openai"
            if "gpt" in model
            else ("anthropic" if "claude" in model else "google")
        )
        cot = methods.get("COT", {})
        pot = methods.get("POT", {})

        cot_acc = cot.get("accuracy", 0) if cot else 0
        pot_acc = pot.get("accuracy", 0) if pot else 0
        avg_acc = (cot_acc + pot_acc) / 2

        cot_cost = cot.get("avg_cost", 0) if cot else 0
        pot_cost = pot.get("avg_cost", 0) if pot else 0

        is_best_cot = model == best_cot[0] and cot_acc > 0
        is_best_pot = model == best_pot[0] and pot_acc > 0

        html += f"""
            <div class="model-card card">
                <div class="model-name model-{provider}">{model}</div>
                <div class="metric-value">{avg_acc:.0f}%</div>
                <div class="metric-label">Average Accuracy</div>
                <div class="metric-sub">
                    <span class="badge badge-cot">COT {cot_acc:.0f}%</span>
                    {"<span class='winner-badge'>BEST</span>" if is_best_cot else ""}
                </div>
                <div class="metric-sub">
                    <span class="badge badge-pot">POT {pot_acc:.0f}%</span>
                    {"<span class='winner-badge'>BEST</span>" if is_best_pot else ""}
                </div>
                <div class="metric-sub" style="margin-top: 10px;">
                    ${(cot_cost + pot_cost) / 2:.4f}/query
                </div>
            </div>
"""

    html += """
            </div>
        </div>
"""

    # Detailed Results Table
    html += """
        <div class="card">
            <h2>Detailed Results Table</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Provider</th>
                    <th>COT Accuracy</th>
                    <th>POT Accuracy</th>
                    <th>COT Cost</th>
                    <th>POT Cost</th>
                    <th>COT Latency</th>
                    <th>POT Latency</th>
                </tr>
"""

    for model, methods in sorted(model_stats.items()):
        provider = (
            "OpenAI"
            if "gpt" in model
            else ("Anthropic" if "claude" in model else "Google")
        )
        cot = methods.get("COT", {})
        pot = methods.get("POT", {})

        html += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td>{provider}</td>
                    <td class="{"correct" if cot.get("accuracy", 0) >= 60 else "incorrect" if cot.get("accuracy", 0) < 40 else "neutral"}">{cot.get("accuracy", 0):.1f}%</td>
                    <td class="{"correct" if pot.get("accuracy", 0) >= 60 else "incorrect" if pot.get("accuracy", 0) < 40 else "neutral"}">{pot.get("accuracy", 0):.1f}%</td>
                    <td>${cot.get("avg_cost", 0):.5f}</td>
                    <td>${pot.get("avg_cost", 0):.5f}</td>
                    <td>{cot.get("avg_latency", 0):.1f}s</td>
                    <td>{pot.get("avg_latency", 0):.1f}s</td>
                </tr>
"""

    html += """
            </table>
        </div>
"""

    # Charts Section
    models = list(model_stats.keys())
    cot_accuracies = [
        model_stats[m].get("COT", {}).get("accuracy", 0)
        if model_stats[m].get("COT")
        else 0
        for m in models
    ]
    pot_accuracies = [
        model_stats[m].get("POT", {}).get("accuracy", 0)
        if model_stats[m].get("POT")
        else 0
        for m in models
    ]
    cot_costs = [
        model_stats[m].get("COT", {}).get("avg_cost", 0)
        if model_stats[m].get("COT")
        else 0
        for m in models
    ]
    pot_costs = [
        model_stats[m].get("POT", {}).get("avg_cost", 0)
        if model_stats[m].get("POT")
        else 0
        for m in models
    ]
    cot_latencies = [
        model_stats[m].get("COT", {}).get("avg_latency", 0)
        if model_stats[m].get("COT")
        else 0
        for m in models
    ]
    pot_latencies = [
        model_stats[m].get("POT", {}).get("avg_latency", 0)
        if model_stats[m].get("POT")
        else 0
        for m in models
    ]

    html += """
        <div class="card">
            <h2>Visualizations</h2>
            <div class="grid grid-2">
                <div id="accuracy-chart"></div>
                <div id="cost-chart"></div>
            </div>
            <div class="grid grid-2" style="margin-top: 20px;">
                <div id="latency-chart"></div>
                <div id="scatter-chart"></div>
            </div>
        </div>
"""

    # Per-Example Analysis
    html += """
        <div class="card">
            <h2>Per-Example Analysis</h2>
"""

    for ex_id, ex_data in by_example.items():
        # Count correct by method
        cot_correct = sum(
            1
            for r in ex_data["results"]
            if r.get("method") == "COT" and r.get("is_correct")
        )
        pot_correct = sum(
            1
            for r in ex_data["results"]
            if r.get("method") == "POT" and r.get("is_correct")
        )
        cot_total = sum(1 for r in ex_data["results"] if r.get("method") == "COT")
        pot_total = sum(1 for r in ex_data["results"] if r.get("method") == "POT")

        html += f"""
            <div class="example-card">
                <div class="example-id">{ex_id}</div>
                <div class="example-question">{ex_data["question"][:200]}{"..." if len(ex_data["question"]) > 200 else ""}</div>
                <div class="example-gt">Ground Truth: {ex_data["ground_truth"]}</div>
                <div style="margin-top: 10px;">
                    <span class="badge badge-cot">COT: {cot_correct}/{cot_total} correct</span>
                    <span class="badge badge-pot">POT: {pot_correct}/{pot_total} correct</span>
                </div>
                <div class="response-grid">
"""

        for r in ex_data["results"]:
            model = r.get("model_name", "")
            method = r.get("method", "")
            answer = r.get("final_answer", "N/A")
            is_correct = r.get("is_correct", False)

            html += f"""
                    <div class="response-item">
                        <div class="response-model">{model} ({method})</div>
                        <div class="response-answer {"correct" if is_correct else "incorrect"}">{answer}</div>
                    </div>
"""

        html += """
                </div>
            </div>
"""

    html += """
        </div>
"""

    # Prompts Section
    html += """
        <div class="card">
            <h2>Paper Prompts Used</h2>
            <p style="color: #666;">These are the exact prompts from the official FinanceReasoning GitHub repository.</p>
            <div class="grid grid-2">
"""

    for method, prompt_info in prompts.items():
        system = prompt_info.get("system", "N/A")
        prefix = prompt_info.get("prefix", "N/A")

        html += f"""
                <div>
                    <h3>{method} Prompt</h3>
                    <h4>System Message:</h4>
                    <div class="prompt-box">{system}</div>
                    <h4 style="margin-top: 15px;">Prefix Template:</h4>
                    <div class="prompt-box">{prefix}</div>
                </div>
"""

    html += """
            </div>
        </div>
"""

    # JavaScript for charts
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
        
        // Chart colors
        const cotColor = '#667eea';
        const potColor = '#28a745';
        
        // Accuracy Chart
        Plotly.newPlot('accuracy-chart', [
            {{
                x: {json.dumps(models)},
                y: {json.dumps(cot_accuracies)},
                name: 'COT (Chain-of-Thought)',
                type: 'bar',
                marker: {{ color: cotColor }}
            }},
            {{
                x: {json.dumps(models)},
                y: {json.dumps(pot_accuracies)},
                name: 'POT (Program-of-Thought)',
                type: 'bar',
                marker: {{ color: potColor }}
            }}
        ], {{
            title: '<b>Accuracy by Model & Method</b>',
            yaxis: {{ title: 'Accuracy (%)', range: [0, 100] }},
            barmode: 'group',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: '-apple-system, BlinkMacSystemFont, sans-serif' }}
        }});
        
        // Cost Chart
        Plotly.newPlot('cost-chart', [
            {{
                x: {json.dumps(models)},
                y: {json.dumps(cot_costs)},
                name: 'COT',
                type: 'bar',
                marker: {{ color: cotColor }}
            }},
            {{
                x: {json.dumps(models)},
                y: {json.dumps(pot_costs)},
                name: 'POT',
                type: 'bar',
                marker: {{ color: potColor }}
            }}
        ], {{
            title: '<b>Average Cost per Query (USD)</b>',
            yaxis: {{ title: 'Cost ($)' }},
            barmode: 'group',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }});
        
        // Latency Chart
        Plotly.newPlot('latency-chart', [
            {{
                x: {json.dumps(models)},
                y: {json.dumps(cot_latencies)},
                name: 'COT',
                type: 'bar',
                marker: {{ color: cotColor }}
            }},
            {{
                x: {json.dumps(models)},
                y: {json.dumps(pot_latencies)},
                name: 'POT',
                type: 'bar',
                marker: {{ color: potColor }}
            }}
        ], {{
            title: '<b>Average Latency (seconds)</b>',
            yaxis: {{ title: 'Latency (s)' }},
            barmode: 'group',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }});
        
        // Scatter: Cost vs Accuracy
        const avgAccuracies = {json.dumps([(cot_accuracies[i] + pot_accuracies[i]) / 2 for i in range(len(models))])};
        const avgCosts = {json.dumps([(cot_costs[i] + pot_costs[i]) / 2 for i in range(len(models))])};
        
        Plotly.newPlot('scatter-chart', [{{
            x: avgCosts,
            y: avgAccuracies,
            mode: 'markers+text',
            type: 'scatter',
            text: {json.dumps(models)},
            textposition: 'top center',
            marker: {{
                size: 20,
                color: avgAccuracies,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {{ title: 'Accuracy' }}
            }}
        }}], {{
            title: '<b>Cost vs Accuracy Trade-off</b>',
            xaxis: {{ title: 'Average Cost ($)', type: 'log' }},
            yaxis: {{ title: 'Average Accuracy (%)', range: [0, 100] }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
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
        # Find most recent paper experiment result
        results_dir = Path(__file__).parent / "quick_results"
        files = list(results_dir.glob("paper_experiment_*.json"))
        if not files:
            print("Usage: python paper_visualization.py <results.json>")
            print("No paper experiment results found in quick_results/")
            return
        results_path = max(files, key=lambda p: p.stat().st_mtime)
    else:
        results_path = Path(sys.argv[1])

    print(f"[INFO] Loading results from: {results_path}")

    html = generate_paper_dashboard(str(results_path))

    # Save HTML
    output_path = results_path.with_suffix(".html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Dashboard saved to: {output_path}")
    print(f"\n[TIP] Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
