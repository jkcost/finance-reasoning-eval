"""
Experiment Results Analyzer and Visualizer

Organizes and visualizes experiment results from evaluation runs.

Features:
- Compare multiple experiments
- Model performance comparison
- Error analysis
- CSV/Markdown export
- HTML visualization dashboard
"""

import json
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class ModelPerformance:
    """Performance metrics for a single model"""
    model_id: str
    total_examples: int = 0
    correct_answers: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_response_time: float = 0.0
    avg_overall_score: float = 0.0
    hallucination_rate: float = 0.0

    @property
    def accuracy(self) -> float:
        """Answer accuracy percentage"""
        if self.total_examples == 0:
            return 0.0
        return (self.correct_answers / self.total_examples) * 100


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    experiment_name: str
    methodology: str
    results_file: Path
    start_time: str
    end_time: str
    models: Dict[str, ModelPerformance]
    total_cost_usd: float = 0.0
    total_examples: int = 0


class ResultsAnalyzer:
    """Analyzes and organizes experiment results"""

    def __init__(self, results_dir: Optional[Path] = None):
        # Default results directory is now at root level (evaluation/ creates it)
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / "results"
        self.results_dir = results_dir

    def load_results(self, results_file: Path) -> ExperimentResult:
        """Load results from JSON file"""
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Initialize model performance
        models_performance = {}
        for model_data in data.get("models", []):
            model_id = model_data["model_id"]
            perf = ModelPerformance(
                model_id=model_id,
                total_examples=len(data.get("examples", [])),
                correct_answers=sum(1 for ex in data.get("examples", [])
                                   if ex.get("final_answer_correct", False)),
                total_tokens=sum(ex.get("total_tokens", 0) for ex in data.get("examples", [])),
                total_cost_usd=sum(ex.get("cost_usd", 0) for ex in data.get("examples", [])),
                avg_response_time=sum(ex.get("response_time_seconds", 0) for ex in data.get("examples", [])) / len(data.get("examples", [])),
                avg_overall_score=sum(ex.get("overall_reasoning_score", 0) for ex in data.get("examples", [])) / len(data.get("examples", [])),
                hallucination_rate=sum(1 for ex in data.get("examples", [])
                                           if ex.get("has_hallucination", False)) / len(data.get("examples", [])),
            )
            models_performance[model_id] = perf

        return ExperimentResult(
            experiment_name=data.get("experiment_name", "Unknown"),
            methodology=data.get("evaluation_config", {}).get("methodology", "Unknown"),
            results_file=results_file,
            start_time=data.get("timestamp", ""),
            end_time=data.get("end_timestamp", ""),
            models=models_performance,
            total_cost_usd=data.get("total_cost_usd", 0.0),
            total_examples=len(data.get("examples", [])),
        )

    def compare_experiments(self, experiments: List[ExperimentResult]) -> None:
        """Compare multiple experiment runs"""
        if len(experiments) < 2:
            print("[INFO] Need at least 2 experiments to compare")
            return

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPARISON")
        print("=" * 80)

        # Model comparison across experiments
        all_models = set()
        for exp in experiments:
            all_models.update(exp.models.keys())

        for model_id in sorted(all_models):
            print(f"\n{model_id}:")
            for exp in experiments:
                perf = exp.models.get(model_id)
                if perf:
                    print(f"  [{exp.experiment_name}] "
                          f"Acc: {perf.accuracy:.1f}% | "
                          f"Score: {perf.avg_overall_score:.3f} | "
                          f"Cost: ${perf.total_cost_usd:.2f}")

    def generate_comparison_table(self, experiments: List[ExperimentResult], output_file: Path) -> None:
        """Generate CSV comparison table"""
        rows = []

        for exp in experiments:
            for model_id, perf in exp.models.items():
                rows.append({
                    "Experiment": exp.experiment_name,
                    "Methodology": exp.methodology,
                    "Model": model_id,
                    "Accuracy (%)": f"{perf.accuracy:.2f}",
                    "Avg Score": f"{perf.avg_overall_score:.3f}",
                    "Avg Time (s)": f"{perf.avg_response_time:.2f}",
                    "Total Cost ($)": f"{perf.total_cost_usd:.4f}",
                    "Hallucination (%)": f"{perf.hallucination_rate:.2f}",
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"[OK] Comparison table saved to: {output_file}")

    def generate_html_dashboard(self, experiments: List[ExperimentResult], output_file: Path) -> None:
        """Generate interactive HTML dashboard for visualization"""

        # Prepare data
        model_data = []
        for exp in experiments:
            for model_id, perf in exp.models.items():
                model_data.append({
                    "experiment": exp.experiment_name,
                    "methodology": exp.methodology,
                    "model": model_id,
                    "accuracy": perf.accuracy,
                    "avg_score": perf.avg_overall_score,
                    "total_cost": perf.total_cost_usd,
                    "hallucination_rate": perf.hallucination_rate,
                })

        df = pd.DataFrame(model_data)

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experiment Results Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: #f0f9f8; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f5f5f5; }}
        .metric {{ display: inline-block; margin: 5px 15px 5px 0; padding: 5px 10px; background: #f0f0f0; border-radius: 4px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .experiment-tag {{ display: inline-block; background: #007bff; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; }}
        .model-selector {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Experiment Results Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="card">
            <h2>Experiment Summary</h2>
            <table>
                <tr>
                    <th>Experiments</th>
                    <th>Total Models</th>
                    <th>Total Cost</th>
                </tr>
                <tr>
                    <td>{len(experiments)}</td>
                    <td>{sum(len(exp.models) for exp in experiments)}</td>
                    <td>${sum(exp.total_cost_usd for exp in experiments):.2f}</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h2>Model Performance</h2>
            <div class="model-selector">
                <select id="modelFilter" onchange="filterByModel(this.value)">
                    <option value="all">All Models</option>
                    {"".join(f'<option value="{{m}}">{{m}}</option>' for m in sorted(all_models))}
                </select>
            </div>
            <div id="modelComparison"></div>
        </div>

        <div class="card">
            <h2>Model Details</h2>
            <div id="modelDetails"></div>
        </div>
    </div>

    <script>
        const modelData = {json.dumps(model_data)};

        function filterByModel(modelId) {{
            const filtered = modelId === 'all' ? modelData : modelData.filter(d => d.model === modelId);
            updateCharts(filtered);
            updateModelDetails(filtered);
        }}

        function updateCharts(data) {{
            const traces = {{
                x: data.map(d => d.experiment),
                y: data.map(d => d.accuracy),
                mode: 'markers',
                type: 'scatter',
                name: 'Accuracy',
                marker: {{ size: 10 }}
            }};

            const costData = {{
                x: data.map(d => d.experiment),
                y: data.map(d => d.total_cost),
                type: 'bar',
                name: 'Total Cost'
            }};

            const scoreData = {{
                x: data.map(d => d.experiment),
                y: data.map(d => d.avg_score),
                mode: 'lines+markers',
                type: 'scatter',
                name: 'Avg Score'
            }};

            Plotly.newPlot('modelComparison', []).addTraces([traces]);
            Plotly.newPlot('costChart', []).addTraces([{{x: [costData.map(d => d.experiment)], y: [costData.map(d => d.total_cost)], type: 'bar'}]);
            Plotly.newPlot('scoreChart', []).addTraces([scoreData]);
        }}

        function updateModelDetails(data) {{
            const container = document.getElementById('modelDetails');
            let html = '<table class="details-table"><tr><th>Metric</th><th>Value</th></tr>';

            const grouped = data.reduce((acc, d) {{
                if (!acc[d.model]) acc[d.model] = [];
                acc[d.model].push(d);
                return acc;
            }}, {{}});

            for (const [model, values] of Object.entries(grouped)) {{
                html += `<tr><td colspan="2"><strong>{{model}}</strong></td></tr>`;
                html += `<tr><td>Accuracy</td><td>{{values[0].accuracy.toFixed(1)}}%</td></tr>`;
                html += `<tr><td>Avg Score</td><td>{{values[0].avg_score.toFixed(3)}}</td></tr>`;
                html += `<tr><td>Total Cost</td><td>${{values[0].total_cost.toFixed(2)}}</td></tr>`;
                html += `<tr><td>Hallucinations</td><td>{{values[0].hallucination_rate.toFixed(1)}}%</td></tr>`;
                html += `<tr><td>Avg Time</td><td>{{values[0].avg_response_time.toFixed(2)}}s</td></tr>`;
                html += `<tr><td>Total Tokens</td><td>{{values[0].total_tokens.toLocaleString()}}</td></tr>`;
                html += `<tr><td>Experiments</td><td>{{values.map(v => v.experiment).join(', ')}}</td></tr>`;
            }}

            html += '</table>';
            container.innerHTML = html;
        }}

        // Initial load
        filterByModel('all');
    </script>
</body>
</html>"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[OK] HTML dashboard saved to: {output_file}")

    def list_all_results(self) -> None:
        """List all available result files"""
        if not self.results_dir.exists():
            print("[INFO] No results directory found")
            return

        result_files = sorted(self.results_dir.glob("**/evaluation_results_*.json"))

        print("\n" + "=" * 60)
        print(f"Found {len(result_files)} result files in {self.results_dir}")
        print("=" * 60)

        for i, result_file in enumerate(result_files, 1, 1):
            size = result_file.stat().st_size / 1024  # KB
            modified = datetime.fromtimestamp(result_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            print(f"  [{i}] {result_file.name} ({size:.0f} KB, {modified})")


if __name__ == "__main__":
    analyzer = ResultsAnalyzer()

    if len(__import__("sys").argv) > 1:
        command = __import__("sys").argv[1]

        if command == "list":
            analyzer.list_all_results()

        elif command == "compare":
            if len(__import__("sys").argv) < 3:
                print("Usage: python results_analyzer.py compare <result1.json> <result2.json> ...")
                return

            result_files = [Path(__import__("sys").argv[i]) for i in range(2, len(__import__("sys").argv))]
            experiments = [analyzer.load_results(f) for f in result_files]

            print(f"\nLoading {len(experiments)} experiments...")
            for exp in experiments:
                print(f"  - {exp.experiment_name}: {len(exp.models)} models, {exp.total_examples} examples")

            # Compare
            analyzer.compare_experiments(experiments)

            # Save comparison table
            comparison_file = analyzer.results_dir / "comparison_table.csv"
            analyzer.generate_comparison_table(experiments, comparison_file)

            # Generate dashboard
            dashboard_file = analyzer.results_dir / "dashboard.html"
            analyzer.generate_html_dashboard(experiments, dashboard_file)

            print(f"\n[OK] Comparison table: {comparison_file}")
            print(f"[OK] Dashboard: {dashboard_file}")

        else:
            print("Available commands:")
            print("  list               - List all result files")
            print("  compare <files...> - Compare multiple experiments")
            print("  visualize <files...> - Generate HTML dashboard")
