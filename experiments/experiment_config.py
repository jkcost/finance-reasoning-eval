"""
Experiment Configuration Manager

Manages experiment configurations including:
- Baseline definitions
- Model configurations
- Methodology strategies (COT, POT, COT variants)
- Dataset configurations
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class Methodology(Enum):
    """Evaluation methodology strategies"""

    COT = "cot"  # Chain-of-Thought (natural language)
    POT = "pot"  # Program-of-Thought (code execution)
    COT_RAG = "cot_rag"  # COT with function retrieval
    POT_RAG = "pot_rag"  # POT with function retrieval
    COT_ONLY = "cot_only"  # COT without structured JSON
    FINANCE_REASONING = "finance_reasoning_compliant"  # Paper-compliant COT


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""

    name: str
    description: str
    methodology: Methodology
    models: List[str]  # Model IDs to evaluate
    baseline: Optional[str] = None  # Baseline name for comparison

    # Dataset configuration
    dataset_name: str = "financereasoning"
    dataset_split: Optional[str] = None  # "easy", "medium", "hard", or None for all
    examples_limit: Optional[int] = None  # Limit number of examples

    # Execution configuration
    concurrency: int = 3
    max_cost_usd: Optional[float] = None
    max_tokens: Optional[int] = None

    # Output configuration
    output_dir: Optional[str] = None  # Override default output location
    save_intermediate_results: bool = True


@dataclass
class BaselineConfig:
    """Baseline comparison configuration"""

    name: str
    results_file: str  # Path to baseline results for comparison
    source: str  # Source (paper, previous experiment, etc.)


class ExperimentConfigManager:
    """Manages experiment configurations and baselines"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent / "config"
        self.config_file = self.config_dir / "experiments.yaml"
        self.baselines_file = self.config_dir / "baselines.yaml"

    def load_experiments(self) -> Dict[str, ExperimentConfig]:
        """Load all experiment configurations"""
        if not self.config_file.exists():
            return {}

        with open(self.config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        experiments = {}
        for name, exp_data in data.get("experiments", {}).items():
            experiments[name] = ExperimentConfig(
                name=name,
                description=exp_data.get("description", ""),
                methodology=Methodology(exp_data.get("methodology", "cot")),
                models=exp_data.get("models", []),
                baseline=exp_data.get("baseline"),
                dataset_name=exp_data.get("dataset_name", "financereasoning"),
                dataset_split=exp_data.get("dataset_split"),
                examples_limit=exp_data.get("examples_limit"),
                concurrency=exp_data.get("concurrency", 3),
                max_cost_usd=exp_data.get("max_cost_usd"),
                max_tokens=exp_data.get("max_tokens"),
                output_dir=exp_data.get("output_dir"),
                save_intermediate_results=exp_data.get("save_intermediate_results", True),
            )

        return experiments

    def save_experiment(self, experiment: ExperimentConfig):
        """Save new experiment configuration"""
        data = self.load_experiments_file()

        # Convert ExperimentConfig to dict
        exp_dict = {
            "description": experiment.description,
            "methodology": experiment.methodology.value,
            "models": experiment.models,
            "baseline": experiment.baseline,
            "dataset_name": experiment.dataset_name,
            "dataset_split": experiment.dataset_split,
            "examples_limit": experiment.examples_limit,
            "concurrency": experiment.concurrency,
            "max_cost_usd": experiment.max_cost_usd,
            "max_tokens": experiment.max_tokens,
            "output_dir": experiment.output_dir,
            "save_intermediate_results": experiment.save_intermediate_results,
        }

        if "experiments" not in data:
            data["experiments"] = {}

        data["experiments"][experiment.name] = exp_dict

        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def load_baselines(self) -> Dict[str, BaselineConfig]:
        """Load baseline configurations"""
        if not self.baselines_file.exists():
            return {}

        with open(self.baselines_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        baselines = {}
        for name, base_data in data.get("baselines", {}).items():
            baselines[name] = BaselineConfig(
                name=name,
                results_file=base_data.get("results_file"),
                source=base_data.get("source", ""),
            )

        return baselines

    def save_baseline(self, baseline: BaselineConfig):
        """Save baseline configuration"""
        data = self.load_baselines_file()

        if "baselines" not in data:
            data["baselines"] = {}

        # Convert BaselineConfig to dict
        base_dict = {
            "results_file": baseline.results_file,
            "source": baseline.source,
        }

        data["baselines"][baseline.name] = base_dict

        with open(self.baselines_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def get_available_models(self) -> List[str]:
        """Get list of available model IDs from config"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

        from config import ConfigManager
        config_manager = ConfigManager()
        models = config_manager.get_models_for_evaluation()
        return [model.id for model in models]


if __name__ == "__main__":
    import sys

    # Create default experiment configuration file
    manager = ExperimentConfigManager()

    # Check if config exists
    if not manager.config_file.exists():
        print(f"[INIT] Creating default experiment config at: {manager.config_file}")

        # Default configuration
        default_config = {
            "experiments": {
                "finance_reasoning_paper_reproduction": {
                    "description": "Reproduce FinanceReasoning paper methodology (COT baseline)",
                    "methodology": "finance_reasoning_compliant",
                    "models": ["claude-sonnet-4.5", "claude-opus-4.5"],
                    "baseline": "gpt_5_paper",
                },
                "methodology_comparison_cot_vs_pot": {
                    "description": "Compare COT vs POT methodologies",
                    "methodology": "cot",
                    "models": ["claude-sonnet-4.5", "gemini-2.5-pro"],
                },
            },
            "baselines": {
                "gpt_5_paper": {
                    "description": "GPT-5 results from FinanceReasoning paper",
                    "results_file": "data/baselines/gpt_5_paper/results.json",
                    "source": "FinanceReasoning Paper (2025)",
                },
            },
        }

        with open(manager.config_file, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print("[OK] Default experiment configuration created")
        print(f"[INFO] Edit: {manager.config_file}")
        print("\nAvailable Methodologies:")
        for method in Methodology:
            print(f"  - {method.value}: {method.name}")
        print("\nAvailable Models:")
        for model_id in manager.get_available_models():
            print(f"  - {model_id}")
