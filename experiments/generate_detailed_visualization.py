"""
Generate Detailed Visualization for Paper Error Cases

Shows:
1. Each problem's question and context
2. Each model's reasoning process (raw_response / executed_code)
3. Where and why each model failed
"""

import json
import html
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def escape_html(text: str) -> str:
    """Escape HTML special characters"""
    if text is None:
        return ""
    return html.escape(str(text))


def format_code(code: str) -> str:
    """Format code with syntax highlighting placeholder"""
    if not code:
        return ""
    return escape_html(code)


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text with ellipsis"""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def generate_detailed_html(results_path: str, output_path: str = None):
    """Generate detailed HTML visualization"""

    # Load results
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    paper_error_cases = data.get("paper_error_cases", {})

    # Group results by example_id
    by_example = {}
    for r in results:
        ex_id = r["example_id"]
        if ex_id not in by_example:
            by_example[ex_id] = {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "paper_table": r.get("paper_table"),
                "paper_error_type": r.get("paper_error_type"),
                "paper_source": r.get("paper_source"),
                "level": r.get("level"),
                "difficulty": r.get("difficulty"),
                "results": [],
            }
        by_example[ex_id]["results"].append(r)

    # Sort by paper table number
    sorted_examples = sorted(
        by_example.items(), key=lambda x: x[1].get("paper_table", 0)
    )

    # Calculate statistics
    total_results = len(results)
    correct_results = sum(1 for r in results if r["is_correct"])
    overall_accuracy = correct_results / total_results * 100 if total_results > 0 else 0

    # Model statistics
    model_stats = {}
    for r in results:
        key = f"{r['model_name']}_{r['method']}"
        if key not in model_stats:
            model_stats[key] = {
                "model": r["model_name"],
                "method": r["method"],
                "correct": 0,
                "total": 0,
            }
        model_stats[key]["total"] += 1
        model_stats[key]["correct"] += 1 if r["is_correct"] else 0

    # Error type statistics
    error_type_stats = {}
    for r in results:
        et = r.get("paper_error_type", "Unknown")
        if et not in error_type_stats:
            error_type_stats[et] = {"correct": 0, "total": 0}
        error_type_stats[et]["total"] += 1
        error_type_stats[et]["correct"] += 1 if r["is_correct"] else 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Error Cases - Detailed Analysis</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        
        /* Header */
        .header {{
            text-align: center;
            padding: 40px 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            margin-bottom: 30px;
        }}
        h1 {{ 
            color: #00d4ff;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{ color: #888; font-size: 1.1em; }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .stat-label {{ color: #888; margin-top: 5px; }}
        
        /* Navigation */
        .nav-section {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .nav-title {{ color: #00d4ff; margin-bottom: 15px; font-size: 1.2em; }}
        .nav-links {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .nav-link {{
            background: rgba(0,212,255,0.2);
            color: #00d4ff;
            padding: 8px 16px;
            border-radius: 20px;
            text-decoration: none;
            font-size: 0.9em;
            transition: all 0.3s;
        }}
        .nav-link:hover {{
            background: rgba(0,212,255,0.4);
            transform: translateY(-2px);
        }}
        
        /* Problem Card */
        .problem-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 20px;
            margin-bottom: 40px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .problem-header {{
            background: rgba(0,212,255,0.15);
            padding: 20px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .problem-title {{
            font-size: 1.4em;
            color: #00d4ff;
        }}
        .problem-badges {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .badge {{
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-table {{ background: rgba(0,212,255,0.3); color: #00d4ff; }}
        .badge-error {{ background: rgba(248,113,113,0.3); color: #f87171; }}
        .badge-level {{ background: rgba(74,222,128,0.3); color: #4ade80; }}
        .badge-score {{ background: rgba(250,204,21,0.3); color: #facc15; }}
        
        /* Question Section */
        .question-section {{
            padding: 25px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .section-title {{
            color: #00d4ff;
            font-size: 1.1em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .section-title::before {{
            content: '';
            width: 4px;
            height: 20px;
            background: #00d4ff;
            border-radius: 2px;
        }}
        .question-text {{
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 12px;
            font-size: 1em;
            line-height: 1.8;
            white-space: pre-wrap;
        }}
        .ground-truth {{
            margin-top: 15px;
            padding: 15px;
            background: rgba(74,222,128,0.15);
            border-radius: 10px;
            border-left: 4px solid #4ade80;
        }}
        .ground-truth strong {{ color: #4ade80; }}
        
        /* Model Results Section */
        .results-section {{
            padding: 25px;
        }}
        .model-results-grid {{
            display: grid;
            gap: 20px;
        }}
        
        /* Individual Model Result */
        .model-result {{
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .model-result.correct {{
            border-color: rgba(74,222,128,0.5);
        }}
        .model-result.incorrect {{
            border-color: rgba(248,113,113,0.5);
        }}
        .model-result-header {{
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.05);
        }}
        .model-info {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .model-name {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        .method-badge {{
            padding: 4px 10px;
            border-radius: 10px;
            font-size: 0.8em;
            background: rgba(147,51,234,0.3);
            color: #a78bfa;
        }}
        .result-status {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .status-icon {{
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
        }}
        .status-icon.correct {{
            background: rgba(74,222,128,0.3);
            color: #4ade80;
        }}
        .status-icon.incorrect {{
            background: rgba(248,113,113,0.3);
            color: #f87171;
        }}
        .answer-comparison {{
            font-size: 0.9em;
        }}
        .answer-comparison .predicted {{ color: #facc15; }}
        .answer-comparison .expected {{ color: #4ade80; }}
        
        /* Reasoning Content */
        .reasoning-content {{
            padding: 20px;
        }}
        .reasoning-toggle {{
            background: rgba(0,212,255,0.2);
            color: #00d4ff;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95em;
            margin-bottom: 15px;
            transition: all 0.3s;
        }}
        .reasoning-toggle:hover {{
            background: rgba(0,212,255,0.4);
        }}
        .reasoning-text {{
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 400px;
            overflow-y: auto;
            display: none;
        }}
        .reasoning-text.visible {{
            display: block;
        }}
        .code-block {{
            background: rgba(0,0,0,0.5);
            padding: 20px;
            border-radius: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-wrap: break-word;
            border-left: 4px solid #a78bfa;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        /* Charts Section */
        .charts-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
        }}
        .chart-title {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        /* Summary Table */
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .summary-table th, .summary-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .summary-table th {{
            background: rgba(0,212,255,0.2);
            color: #00d4ff;
        }}
        .summary-table tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(0,212,255,0.5);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(0,212,255,0.7);
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .problem-header {{
                flex-direction: column;
                align-items: flex-start;
            }}
            .model-result-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }}
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Paper Error Cases - Detailed Analysis</h1>
            <p class="subtitle">FinanceReasoning Paper Appendix B (Table 6-14) Error Cases</p>
        </div>
        
        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(sorted_examples)}</div>
                <div class="stat-label">Error Cases</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(model_stats)}</div>
                <div class="stat-label">Model-Method Combinations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_results}</div>
                <div class="stat-label">Total Evaluations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overall_accuracy:.1f}%</div>
                <div class="stat-label">Overall Accuracy</div>
            </div>
        </div>
        
        <!-- Navigation -->
        <div class="nav-section">
            <div class="nav-title">Quick Navigation</div>
            <div class="nav-links">
"""

    for ex_id, ex_data in sorted_examples:
        table = ex_data.get("paper_table", "?")
        error_type = ex_data.get("paper_error_type", "Unknown")
        html_content += f'<a href="#problem-{ex_id}" class="nav-link">Table {table}: {error_type}</a>\n'

    html_content += """
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-section">
            <div class="chart-card">
                <div class="chart-title">Accuracy by Model (POT)</div>
                <canvas id="modelAccuracyChart"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">Accuracy by Error Type</div>
                <canvas id="errorTypeChart"></canvas>
            </div>
        </div>
        
        <!-- Summary Tables -->
        <div class="charts-section">
            <div class="chart-card">
                <div class="chart-title">Model Performance Summary</div>
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Method</th>
                            <th>Correct</th>
                            <th>Total</th>
                            <th>Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for key, stats in sorted(model_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        acc_color = "#4ade80" if acc >= 50 else "#f87171"
        html_content += f"""
                        <tr>
                            <td>{stats["model"]}</td>
                            <td>{stats["method"]}</td>
                            <td>{stats["correct"]}</td>
                            <td>{stats["total"]}</td>
                            <td style="color: {acc_color}">{acc:.1f}%</td>
                        </tr>
"""

    html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="chart-card">
                <div class="chart-title">Error Type Analysis</div>
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Error Type</th>
                            <th>Correct</th>
                            <th>Total</th>
                            <th>Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for et, stats in sorted(error_type_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        acc_color = "#4ade80" if acc >= 50 else "#f87171"
        html_content += f"""
                        <tr>
                            <td>{et}</td>
                            <td>{stats["correct"]}</td>
                            <td>{stats["total"]}</td>
                            <td style="color: {acc_color}">{acc:.1f}%</td>
                        </tr>
"""

    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Problem Cards -->
        <h2 style="color: #00d4ff; margin: 30px 0 20px;">Detailed Results by Problem</h2>
"""

    # Generate each problem card
    for ex_id, ex_data in sorted_examples:
        table = ex_data.get("paper_table", "?")
        error_type = ex_data.get("paper_error_type", "Unknown")
        level = ex_data.get("level", "unknown")
        gt = ex_data.get("ground_truth")
        question = ex_data.get("question", "")

        correct_count = sum(1 for r in ex_data["results"] if r["is_correct"])
        total_count = len(ex_data["results"])

        html_content += f"""
        <div class="problem-card" id="problem-{ex_id}">
            <div class="problem-header">
                <div class="problem-title">{ex_id}</div>
                <div class="problem-badges">
                    <span class="badge badge-table">Table {table}</span>
                    <span class="badge badge-error">{escape_html(error_type)}</span>
                    <span class="badge badge-level">{level}</span>
                    <span class="badge badge-score">Score: {correct_count}/{total_count}</span>
                </div>
            </div>
            
            <div class="question-section">
                <div class="section-title">Question</div>
                <div class="question-text">{escape_html(question)}</div>
                <div class="ground-truth">
                    <strong>Ground Truth:</strong> {gt}
                </div>
            </div>
            
            <div class="results-section">
                <div class="section-title">Model Results & Reasoning</div>
                <div class="model-results-grid">
"""

        # Sort results by model name and method
        sorted_results = sorted(
            ex_data["results"], key=lambda x: (x["model_name"], x["method"])
        )

        for idx, r in enumerate(sorted_results):
            is_correct = r["is_correct"]
            status_class = "correct" if is_correct else "incorrect"
            status_icon = "✓" if is_correct else "✗"

            raw_response = r.get("raw_response", "")
            executed_code = r.get("executed_code", "")
            final_answer = r.get("final_answer")

            # Unique ID for toggle
            toggle_id = f"{ex_id}-{r['model_name']}-{r['method']}-{idx}"

            html_content += f'''
                    <div class="model-result {status_class}">
                        <div class="model-result-header">
                            <div class="model-info">
                                <span class="model-name">{r["model_name"]}</span>
                                <span class="method-badge">{r["method"]}</span>
                            </div>
                            <div class="result-status">
                                <div class="answer-comparison">
                                    <span class="predicted">Predicted: {final_answer}</span> | 
                                    <span class="expected">Expected: {gt}</span>
                                </div>
                                <div class="status-icon {status_class}">{status_icon}</div>
                            </div>
                        </div>
                        <div class="reasoning-content">
                            <button class="reasoning-toggle" onclick="toggleReasoning('{toggle_id}')">
                                Show Reasoning Process
                            </button>
                            <div id="{toggle_id}" class="reasoning-text">
'''

            if r["method"] == "POT" and executed_code:
                html_content += (
                    f'<div class="code-block">{format_code(executed_code)}</div>'
                )
            elif raw_response:
                html_content += f"{escape_html(raw_response)}"

            html_content += """
                            </div>
                        </div>
                    </div>
"""

        html_content += """
                </div>
            </div>
        </div>
"""

    # Chart data
    pot_models = [s for k, s in model_stats.items() if s["method"] == "POT"]
    model_labels = [s["model"] for s in pot_models]
    model_accuracies = [
        s["correct"] / s["total"] * 100 if s["total"] > 0 else 0 for s in pot_models
    ]

    error_labels = list(error_type_stats.keys())
    error_accuracies = [
        error_type_stats[et]["correct"] / error_type_stats[et]["total"] * 100
        if error_type_stats[et]["total"] > 0
        else 0
        for et in error_labels
    ]

    html_content += f"""
        <script>
            function toggleReasoning(id) {{
                const el = document.getElementById(id);
                el.classList.toggle('visible');
                const btn = el.previousElementSibling;
                btn.textContent = el.classList.contains('visible') ? 'Hide Reasoning Process' : 'Show Reasoning Process';
            }}
            
            // Model Accuracy Chart
            new Chart(document.getElementById('modelAccuracyChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(model_labels)},
                    datasets: [{{
                        label: 'Accuracy (%)',
                        data: {json.dumps(model_accuracies)},
                        backgroundColor: 'rgba(0, 212, 255, 0.6)',
                        borderColor: 'rgba(0, 212, 255, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        x: {{
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#888' }} }}
                    }}
                }}
            }});
            
            // Error Type Chart
            new Chart(document.getElementById('errorTypeChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(error_labels)},
                    datasets: [{{
                        label: 'Accuracy (%)',
                        data: {json.dumps(error_accuracies)},
                        backgroundColor: error_accuracies => {{
                            return {json.dumps(error_accuracies)}.map(v => v >= 50 ? 'rgba(74, 222, 128, 0.6)' : 'rgba(248, 113, 113, 0.6)');
                        }},
                        borderColor: {json.dumps(error_accuracies)}.map(v => v >= 50 ? 'rgba(74, 222, 128, 1)' : 'rgba(248, 113, 113, 1)'),
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    indexAxis: 'y',
                    scales: {{
                        x: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        y: {{
                            ticks: {{ color: '#888', font: {{ size: 10 }} }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ display: false }}
                    }}
                }}
            }});
        </script>
    </div>
</body>
</html>
"""

    # Determine output path
    if output_path is None:
        output_path = Path(results_path).parent / f"detailed_analysis_{timestamp}.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated detailed visualization: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import sys

    # Find the most recent results file
    results_dir = Path(__file__).parent / "quick_results"

    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Find most recent paper_error_cases file
        results_files = list(results_dir.glob("paper_error_cases_*.json"))
        if not results_files:
            print("No paper_error_cases results found!")
            sys.exit(1)
        results_path = max(results_files, key=lambda p: p.stat().st_mtime)

    print(f"Using results file: {results_path}")
    output_path = generate_detailed_html(str(results_path))
    print(f"\nOpen in browser: {output_path}")
