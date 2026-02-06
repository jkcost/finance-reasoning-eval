"""
Error Report Generator for FinanceReasoning Evaluation

Generates HTML and JSON reports for error analysis results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from .error_taxonomy import (
    ErrorCategory,
    ErrorClassification,
    ModelType,
    MODEL_REGISTRY,
    ERROR_METADATA,
)


@dataclass
class ErrorAnalysisResult:
    """Complete error analysis result for one example"""

    example_id: str
    model_name: str
    method: str
    ground_truth: Any
    predicted_answer: Any
    is_correct: bool
    classification: Optional[ErrorClassification]
    raw_response: str
    cost_usd: float
    latency_seconds: float
    paper_table: Optional[int] = None
    paper_error_type: Optional[str] = None


@dataclass
class ModelTypeComparison:
    """Comparison between model types"""

    general_accuracy: float
    reasoning_accuracy: float
    general_top_errors: List[tuple]  # (ErrorCategory, count)
    reasoning_top_errors: List[tuple]
    general_total: int
    reasoning_total: int


class ErrorReportGenerator:
    """Generates comprehensive error analysis reports"""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_json_report(
        self,
        results: List[ErrorAnalysisResult],
        output_path: Path,
        experiment_name: str = "error_analysis",
    ) -> Path:
        """Generate JSON report"""

        # Calculate statistics
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        incorrect = total - correct

        # Error distribution
        error_distribution = {}
        for r in results:
            if not r.is_correct and r.classification:
                cat = r.classification.category.value
                error_distribution[cat] = error_distribution.get(cat, 0) + 1

        # Model comparison
        model_stats = {}
        for r in results:
            key = f"{r.model_name}_{r.method}"
            if key not in model_stats:
                model_stats[key] = {
                    "model": r.model_name,
                    "method": r.method,
                    "total": 0,
                    "correct": 0,
                    "cost": 0.0,
                    "errors": {},
                }
            model_stats[key]["total"] += 1
            model_stats[key]["correct"] += 1 if r.is_correct else 0
            model_stats[key]["cost"] += r.cost_usd

            if not r.is_correct and r.classification:
                cat = r.classification.category.value
                model_stats[key]["errors"][cat] = (
                    model_stats[key]["errors"].get(cat, 0) + 1
                )

        # Calculate accuracy
        for key in model_stats:
            total_m = model_stats[key]["total"]
            model_stats[key]["accuracy"] = (
                model_stats[key]["correct"] / total_m * 100 if total_m > 0 else 0
            )

        # Model type comparison
        model_type_comparison = self._calculate_model_type_comparison(results)

        # Build report
        report = {
            "experiment_id": f"{experiment_name}_{self.timestamp}",
            "timestamp": self.timestamp,
            "summary": {
                "total_examples": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": correct / total * 100 if total > 0 else 0,
            },
            "model_type_comparison": {
                "general": {
                    "accuracy": model_type_comparison.general_accuracy,
                    "total": model_type_comparison.general_total,
                    "top_errors": [
                        {"category": cat.value, "count": cnt}
                        for cat, cnt in model_type_comparison.general_top_errors
                    ],
                },
                "reasoning": {
                    "accuracy": model_type_comparison.reasoning_accuracy,
                    "total": model_type_comparison.reasoning_total,
                    "top_errors": [
                        {"category": cat.value, "count": cnt}
                        for cat, cnt in model_type_comparison.reasoning_top_errors
                    ],
                },
            },
            "error_distribution": error_distribution,
            "model_stats": model_stats,
            "detailed_cases": [
                {
                    "example_id": r.example_id,
                    "model_name": r.model_name,
                    "method": r.method,
                    "ground_truth": r.ground_truth,
                    "predicted_answer": r.predicted_answer,
                    "is_correct": r.is_correct,
                    "classification": (
                        r.classification.to_dict() if r.classification else None
                    ),
                    "paper_table": r.paper_table,
                    "paper_error_type": r.paper_error_type,
                }
                for r in results
            ],
        }

        # Save
        output_file = output_path / f"{experiment_name}_{self.timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return output_file

    def generate_html_report(
        self,
        results: List[ErrorAnalysisResult],
        output_path: Path,
        experiment_name: str = "error_analysis",
    ) -> Path:
        """Generate HTML report with visualizations"""

        # Calculate statistics
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)

        # Error distribution
        error_distribution = {}
        for r in results:
            if not r.is_correct and r.classification:
                cat = r.classification.category
                error_distribution[cat] = error_distribution.get(cat, 0) + 1

        # Model stats
        model_stats = {}
        for r in results:
            key = f"{r.model_name}_{r.method}"
            if key not in model_stats:
                model_stats[key] = {
                    "model": r.model_name,
                    "method": r.method,
                    "total": 0,
                    "correct": 0,
                }
            model_stats[key]["total"] += 1
            model_stats[key]["correct"] += 1 if r.is_correct else 0

        # Model type comparison
        model_type_comparison = self._calculate_model_type_comparison(results)

        # Generate HTML
        html_content = self._generate_html_content(
            results,
            total,
            correct,
            error_distribution,
            model_stats,
            model_type_comparison,
            experiment_name,
        )

        output_file = output_path / f"{experiment_name}_{self.timestamp}.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_file

    def _calculate_model_type_comparison(
        self, results: List[ErrorAnalysisResult]
    ) -> ModelTypeComparison:
        """Calculate comparison between general and reasoning models"""

        general_results = []
        reasoning_results = []

        for r in results:
            model_info = MODEL_REGISTRY.get(r.model_name)
            if model_info:
                if model_info.model_type == ModelType.GENERAL:
                    general_results.append(r)
                else:
                    reasoning_results.append(r)
            else:
                # Default to general if not in registry
                general_results.append(r)

        # Calculate accuracies
        general_total = len(general_results)
        general_correct = sum(1 for r in general_results if r.is_correct)
        general_accuracy = (
            general_correct / general_total * 100 if general_total > 0 else 0
        )

        reasoning_total = len(reasoning_results)
        reasoning_correct = sum(1 for r in reasoning_results if r.is_correct)
        reasoning_accuracy = (
            reasoning_correct / reasoning_total * 100 if reasoning_total > 0 else 0
        )

        # Top errors
        general_errors = {}
        for r in general_results:
            if not r.is_correct and r.classification:
                cat = r.classification.category
                general_errors[cat] = general_errors.get(cat, 0) + 1

        reasoning_errors = {}
        for r in reasoning_results:
            if not r.is_correct and r.classification:
                cat = r.classification.category
                reasoning_errors[cat] = reasoning_errors.get(cat, 0) + 1

        general_top = sorted(general_errors.items(), key=lambda x: -x[1])[:5]
        reasoning_top = sorted(reasoning_errors.items(), key=lambda x: -x[1])[:5]

        return ModelTypeComparison(
            general_accuracy=general_accuracy,
            reasoning_accuracy=reasoning_accuracy,
            general_top_errors=general_top,
            reasoning_top_errors=reasoning_top,
            general_total=general_total,
            reasoning_total=reasoning_total,
        )

    def _generate_html_content(
        self,
        results: List[ErrorAnalysisResult],
        total: int,
        correct: int,
        error_distribution: Dict[ErrorCategory, int],
        model_stats: Dict[str, Dict],
        model_type_comparison: ModelTypeComparison,
        experiment_name: str,
    ) -> str:
        """Generate HTML content"""

        # Prepare chart data
        error_labels = [cat.value for cat in error_distribution.keys()]
        error_counts = list(error_distribution.values())

        model_labels = list(model_stats.keys())
        model_accuracies = [
            (s["correct"] / s["total"] * 100 if s["total"] > 0 else 0)
            for s in model_stats.values()
        ]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error Analysis Report - {experiment_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #00d4ff;
            font-size: 2.5em;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            backdrop-filter: blur(10px);
        }}
        .summary-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .summary-label {{
            color: #888;
            margin-top: 5px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }}
        .chart-title {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        .comparison-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .comparison-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }}
        .comparison-title {{
            color: #00d4ff;
            font-size: 1.3em;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .accuracy-badge {{
            background: rgba(0,212,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            background: rgba(0,212,255,0.2);
            color: #00d4ff;
            font-weight: 600;
        }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        .correct {{ color: #4ade80; }}
        .incorrect {{ color: #f87171; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        .badge-error {{ background: rgba(248,113,113,0.2); color: #f87171; }}
        .badge-success {{ background: rgba(74,222,128,0.2); color: #4ade80; }}
        .badge-warning {{ background: rgba(251,191,36,0.2); color: #fbbf24; }}
        .badge-info {{ background: rgba(0,212,255,0.2); color: #00d4ff; }}
        .error-detail {{
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 0.9em;
        }}
        .section-title {{
            color: #00d4ff;
            font-size: 1.4em;
            margin: 30px 0 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Error Analysis Report</h1>
        <p class="subtitle">{experiment_name} | Generated: {self.timestamp}</p>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-value">{total}</div>
                <div class="summary-label">Total Examples</div>
            </div>
            <div class="summary-card">
                <div class="summary-value correct">{correct}</div>
                <div class="summary-label">Correct</div>
            </div>
            <div class="summary-card">
                <div class="summary-value incorrect">{total - correct}</div>
                <div class="summary-label">Incorrect</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{correct/total*100:.1f}%</div>
                <div class="summary-label">Accuracy</div>
            </div>
        </div>

        <!-- Model Type Comparison -->
        <h2 class="section-title">Model Type Comparison</h2>
        <div class="comparison-section">
            <div class="comparison-card">
                <div class="comparison-title">
                    <span>General-Purpose Models</span>
                    <span class="accuracy-badge">{model_type_comparison.general_accuracy:.1f}% accuracy</span>
                </div>
                <p style="color: #888; margin-bottom: 15px;">N = {model_type_comparison.general_total}</p>
                <h4 style="margin-bottom: 10px;">Top Error Types:</h4>
                <ul style="list-style: none;">
                    {"".join(f'<li style="margin: 5px 0;"><span class="badge badge-error">{cat.value}</span> {cnt}</li>' for cat, cnt in model_type_comparison.general_top_errors)}
                </ul>
            </div>
            <div class="comparison-card">
                <div class="comparison-title">
                    <span>Reasoning Models</span>
                    <span class="accuracy-badge">{model_type_comparison.reasoning_accuracy:.1f}% accuracy</span>
                </div>
                <p style="color: #888; margin-bottom: 15px;">N = {model_type_comparison.reasoning_total}</p>
                <h4 style="margin-bottom: 10px;">Top Error Types:</h4>
                <ul style="list-style: none;">
                    {"".join(f'<li style="margin: 5px 0;"><span class="badge badge-error">{cat.value}</span> {cnt}</li>' for cat, cnt in model_type_comparison.reasoning_top_errors) if model_type_comparison.reasoning_top_errors else '<li style="color: #888;">No errors (or no reasoning models tested)</li>'}
                </ul>
            </div>
        </div>

        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Error Distribution</div>
                <canvas id="errorDistChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Model Accuracy</div>
                <canvas id="modelAccuracyChart"></canvas>
            </div>
        </div>

        <!-- Model Results Table -->
        <h2 class="section-title">Results by Model</h2>
        <table>
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
            acc_class = "correct" if acc >= 50 else "incorrect"
            html += f"""
                <tr>
                    <td>{stats["model"]}</td>
                    <td>{stats["method"]}</td>
                    <td>{stats["correct"]}</td>
                    <td>{stats["total"]}</td>
                    <td class="{acc_class}">{acc:.1f}%</td>
                </tr>
"""

        html += """
            </tbody>
        </table>

        <!-- Error Cases Detail -->
        <h2 class="section-title">Error Cases Detail</h2>
        <table>
            <thead>
                <tr>
                    <th>Example ID</th>
                    <th>Model</th>
                    <th>Method</th>
                    <th>GT</th>
                    <th>Predicted</th>
                    <th>Error Type</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
"""

        for r in results:
            if r.is_correct:
                continue

            error_type = r.classification.category.value if r.classification else "N/A"
            confidence = (
                f"{r.classification.confidence*100:.0f}%"
                if r.classification
                else "N/A"
            )
            html += f"""
                <tr>
                    <td>{r.example_id}</td>
                    <td>{r.model_name}</td>
                    <td>{r.method}</td>
                    <td>{r.ground_truth}</td>
                    <td>{r.predicted_answer}</td>
                    <td><span class="badge badge-error">{error_type}</span></td>
                    <td>{confidence}</td>
                </tr>
"""

        html += f"""
            </tbody>
        </table>

        <script>
            // Error Distribution Chart
            new Chart(document.getElementById('errorDistChart'), {{
                type: 'pie',
                data: {{
                    labels: {json.dumps(error_labels)},
                    datasets: [{{
                        data: {json.dumps(error_counts)},
                        backgroundColor: [
                            'rgba(248, 113, 113, 0.8)',
                            'rgba(251, 191, 36, 0.8)',
                            'rgba(74, 222, 128, 0.8)',
                            'rgba(0, 212, 255, 0.8)',
                            'rgba(167, 139, 250, 0.8)',
                            'rgba(236, 72, 153, 0.8)',
                            'rgba(34, 211, 238, 0.8)',
                            'rgba(163, 230, 53, 0.8)',
                            'rgba(249, 115, 22, 0.8)',
                        ]
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'right',
                            labels: {{ color: '#888' }}
                        }}
                    }}
                }}
            }});

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
                    indexAxis: 'y',
                    scales: {{
                        x: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        y: {{
                            ticks: {{ color: '#888' }},
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
        return html

    def generate_reports(
        self,
        results: List[ErrorAnalysisResult],
        output_path: Path,
        experiment_name: str = "error_analysis",
    ) -> Dict[str, Path]:
        """Generate both JSON and HTML reports"""

        output_path.mkdir(parents=True, exist_ok=True)

        json_path = self.generate_json_report(results, output_path, experiment_name)
        html_path = self.generate_html_report(results, output_path, experiment_name)

        return {
            "json": json_path,
            "html": html_path,
        }
