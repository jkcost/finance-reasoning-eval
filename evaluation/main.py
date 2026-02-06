"""
FinanceReasoning Evaluation System - Main Entry Point

Reorganized for modular experiment management:
- experiments/ folder for experiment configurations
- results/ folder for organized output
- Methodology implementations (COT, POT variants)
- Integrated result analysis and visualization
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from dataclasses import dataclass

from config import ConfigManager, EvaluationConfig
from dataset_loader import DatasetLoader, Example
from model_runner import ModelRunner
from response_parser import ResponseParser
from metrics_evaluator import MetricsEvaluator
from result_store import ResultStore
from experiment_config import ExperimentConfigManager, ExperimentConfig, Methodology


@dataclass
class ExperimentRun:
    """Configuration for a single experiment run"""

    name: str
    description: str
    methodology: Methodology
    models: List[str]
    dataset_name: str
    output_dir: Optional[str] = None


class EvaluationOrchestrator:
    """Orchestrates evaluation experiments with methodology support"""

    def __init__(self):
        # Load environment
        project_root = Path(__file__).parent
        env_file = project_root / ".env"
        load_dotenv(env_file)

        # Initialize managers
        self.config_manager = ConfigManager()
        self.experiment_manager = ExperimentConfigManager()
        self.result_store = ResultStore()

        # Set directories
        self.results_dir = project_root / "results"

    async def run_experiment(self, config: ExperimentConfig) -> str:
        """Run a single experiment"""
        print("\n" + "=" * 70)
        print(f"[EXPERIMENT] {config.name}")
        print("=" * 70)
        print(f"Description: {config.description}")
        print(f"Methodology: {config.methodology.name}")
        print(f"Models: {', '.join(config.models)}")
        print(f"Dataset: {config.dataset_name}")
        print(f"Examples: {config.examples_limit or 'All'}")

        # Get evaluation config
        eval_config = self.config_manager.get_evaluation_config()

        # Override with experiment config
        if config.examples_limit:
            # Note: dataset_loader doesn't support limit directly, would need to filter
            print(f"[INFO] Examples limit: {config.examples_limit}")

        # Determine prompt template based on methodology
        from prompt_builder import PromptBuilder

        prompt_builder = PromptBuilder()

        methodology_to_template = {
            Methodology.COT: "cot",
            Methodology.POT: "pot",
            Methodology.COT_RAG: "cot_rag",
            Methodology.POT_RAG: "pot_rag",
            Methodology.COT_ONLY: "cot_only",
            Methodology.FINANCE_REASONING: "finance_reasoning_compliant",
        }

        template_name = methodology_to_template.get(
            config.methodology, "finance_reasoning_compliant"
        )
        print(f"[INFO] Using prompt template: {template_name}")

        # Load dataset
        dataset_loader = DatasetLoader()
        examples = dataset_loader.load_dataset(
            dataset_name=config.dataset_name,
            split=config.dataset_split,
        )

        if config.examples_limit and len(examples) > config.examples_limit:
            examples = examples[: config.examples_limit]
            print(f"[INFO] Limited to {len(examples)} examples")

        print(f"[INFO] Loaded {len(examples)} examples from dataset")

        # Initialize model runner
        model_runner = ModelRunner(self.config_manager, eval_config)
        print(f"[INFO] Configured {len(model_runner.providers)} model providers")

        # Initialize other components
        response_parser = ResponseParser()
        metrics_evaluator = MetricsEvaluator()

        # Run evaluation
        all_results = []
        total_cost = 0.0

        for i, example in enumerate(examples, 1):
            print(f"\n[{i}/{len(examples)}] Processing: {example.id}")

            # Build prompt
            prompt = prompt_builder.build_prompt(
                template_type=template_name,
                question=example.question,
                context=example.context,
                python_solution=example.python_solution,
            )

            # Evaluate with all models
            responses = await model_runner.evaluate_example(
                example=example,
                template_type=template_name,
            )

            # Parse responses and evaluate metrics
            example_results = {}
            for model_id, llm_response in responses.items():
                if llm_response.parse_status != "success":
                    example_results[model_id] = {
                        "model_id": model_id,
                        "parse_status": llm_response.parse_status,
                        "error": llm_response.raw_response,
                    }
                    continue

                # Parse response
                parsed = response_parser.parse(llm_response.raw_response, example)
                example_results[model_id] = parsed

                # Track cost
                total_cost += llm_response.cost_usd

            # Store results
            self.result_store.save_example_result(
                experiment_name=config.name,
                example_id=example.id,
                model_results=example_results,
            )

            # Show progress
            if i % 5 == 0:
                cost_summary = f"Total cost so far: ${total_cost:.4f}"
                print(
                    f"[PROGRESS] {i}/{len(examples)} examples processed | {cost_summary}"
                )

        # Generate summary report
        evaluation_id = f"eval_{config.name}"

        summary_file = self.result_store.generate_summary(evaluation_id)

        print("\n" + "=" * 70)
        print("[COMPLETE] Evaluation Finished")
        print("=" * 70)
        print(f"Results saved to: {summary_file}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Total examples: {len(examples)}")
        print(f"Evaluation ID: {evaluation_id}")

        return evaluation_id

    def list_experiments(self):
        """List available experiments"""
        experiments = self.experiment_manager.load_experiments()

        print("\n" + "=" * 70)
        print("Available Experiments")
        print("=" * 70)

        for name, config in experiments.items():
            print(f"\n{name}")
            print(f"  Description: {config.description}")
            print(f"  Methodology: {config.methodology.name}")
            print(f"  Models: {', '.join(config.models)}")
            if config.baseline:
                print(f"  Baseline: {config.baseline.name}")

        print(f"\nTotal: {len(experiments)} experiments configured")

    def create_experiment_config(
        self,
        name: str,
        methodology: str,
        models: List[str],
        description: str = "",
    ):
        """Create a new experiment configuration"""
        config = ExperimentConfig(
            name=name,
            description=description,
            methodology=Methodology(methodology),
            models=models,
            dataset_name="financereasoning",
        )

        self.experiment_manager.save_experiment(config)
        print(f"[OK] Experiment '{name}' created")


async def main():
    """Main entry point"""
    import sys

    orchestrator = EvaluationOrchestrator()

    # CLI commands
    if len(__import__("sys").argv) < 2:
        print("FinanceReasoning Evaluation System")
        print("\nCommands:")
        print("  run <experiment_name>     Run a named experiment")
        print("  list                       List all experiments")
        print("  create <name> <methodology> <models...>")
        print("                            Create new experiment config")
        print(
            "                            Methods: cot, pot, cot_rag, pot_rag, finance_reasoning"
        )
        print("\nExamples:")
        print("  python main.py run finance_reasoning_paper_reproduction")
        print(
            "  python main.py create compare_cot_vs_pot cot claude-sonnet-4.5 gemini-2.5-pro"
        )
        print("  python main.py create test_baseline pot gemini-2.5-pro")
        return

    command = __import__("sys").argv[1]

    if command == "list":
        orchestrator.list_experiments()

    elif command == "run":
        if len(__import__("sys").argv) < 3:
            print("[ERROR] Usage: python main.py run <experiment_name>")
            return

        experiment_name = __import__("sys").argv[2]

        experiments = orchestrator.experiment_manager.load_experiments()

        if experiment_name not in experiments:
            print(f"[ERROR] Experiment '{experiment_name}' not found")
            print(f"Available experiments: {', '.join(experiments.keys())}")
            return

        config = experiments[experiment_name]

        # Run experiment
        evaluation_id = await orchestrator.run_experiment(config)

        print(f"\n[INFO] Evaluation ID: {evaluation_id}")
        print(f"[INFO] View results: python experiments/results_analyzer.py list")

    elif command == "create":
        if len(__import__("sys").argv) < 4:
            print(
                "[ERROR] Usage: python main.py create <name> <methodology> <model1> <model2> ..."
            )
            return

        name = __import__("sys").argv[2]
        methodology = __import__("sys").argv[3]
        models = __import__("sys").argv[4:]

        try:
            orchestrator.create_experiment_config(
                name=name,
                methodology=methodology,
                models=models,
                description=f"Custom experiment - {len(models)} models",
            )
        except ValueError as e:
            print(f"[ERROR] {e}")
            return

        print(f"[OK] Experiment '{name}' configured with methodology: {methodology}")
        print(f"[INFO] Available experiments:")
        orchestrator.list_experiments()

    else:
        print(f"[ERROR] Unknown command: {command}")
        print("Run 'python main.py' for help")


if __name__ == "__main__":
    asyncio.run(main())
