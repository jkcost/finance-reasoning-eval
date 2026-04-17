"""
Config Module - API Key Management and Configuration

Supports multiple API key sources:
1. Environment variables (.env)
2. Config file (config/local_secrets.yaml - gitignored)

Models configuration with provider-specific settings.
"""

import os
import yaml
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Individual model configuration"""

    id: str
    name: str
    provider: str
    api_key_env_var: Optional[str] = None
    model_id: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    base_url: Optional[str] = None
    cost_per_million_tokens: Optional[float] = None


@dataclass
class ProviderConfig:
    """Provider-specific configuration"""

    name: str
    api_key_env_var: str
    base_url: Optional[str] = None
    default_max_requests_per_minute: Optional[int] = None
    default_max_tokens_per_minute: Optional[int] = None


@dataclass
class EvaluationConfig:
    """Evaluation run configuration"""

    models: List[ModelConfig]
    output_dir: str = "results"
    max_cost_usd: Optional[float] = 100.0
    max_tokens: Optional[int] = None
    stop_on_budget_exceed: bool = False
    concurrency_per_provider: int = 3
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    cache_enabled: bool = True
    cache_ttl_hours: int = 24


# Default provider configurations
DEFAULT_PROVIDERS = {
    "openai": ProviderConfig(
        name="OpenAI",
        api_key_env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_max_requests_per_minute=150,
        default_max_tokens_per_minute=150000,
    ),
    "anthropic": ProviderConfig(
        name="Anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        default_max_requests_per_minute=50,
        default_max_tokens_per_minute=100000,
    ),
    "google": ProviderConfig(
        name="Google",
        api_key_env_var="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com",
        default_max_requests_per_minute=60,
        default_max_tokens_per_minute=100000,
    ),
    "ollama": ProviderConfig(
        name="Ollama (Local)",
        api_key_env_var="",  # No API key needed
        base_url="http://localhost:11434",  # Override with OLLAMA_BASE_URL env var
        default_max_requests_per_minute=10,
        default_max_tokens_per_minute=50000,
    ),
}

# Default models from paper and available (verified via API)
DEFAULT_MODELS = [
    # OpenAI models (paper top performers) - API key not verified
    ModelConfig(
        id="gpt-5-2025-08-07",
        name="GPT-5 (Aug 2025)",
        provider="openai",
        api_key_env_var="OPENAI_API_KEY",
        model_id="gpt-5-2025-08-07",
        max_tokens=128000,
        cost_per_million_tokens=15.0,  # GPT-5 pricing
    ),
    ModelConfig(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        api_key_env_var="OPENAI_API_KEY",
        model_id="gpt-4o-mini",
        max_tokens=128000,
        cost_per_million_tokens=0.15,  # GPT-4o-mini pricing
    ),
    ModelConfig(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        api_key_env_var="OPENAI_API_KEY",
        model_id="gpt-4o",
        max_tokens=128000,
        cost_per_million_tokens=2.50,  # GPT-4o pricing
    ),
    # Anthropic models (verified available via API)
    ModelConfig(
        id="claude-sonnet-4.5",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        model_id="claude-sonnet-4-5-20250929",
        max_tokens=64000,  # Max output tokens for Sonnet 4.5
        cost_per_million_tokens=3.0,
        temperature=1.0,
    ),
    ModelConfig(
        id="claude-opus-4.5",
        name="Claude Opus 4.5",
        provider="anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        model_id="claude-opus-4-5-20251101",
        max_tokens=64000,  # Max output tokens for Opus 4.5
        cost_per_million_tokens=25.0,  # Opus 4.5 pricing
        temperature=1.0,
    ),
    ModelConfig(
        id="claude-haiku-4.5",
        name="Claude Haiku 4.5",
        provider="anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        model_id="claude-haiku-4-5-20251001",
        max_tokens=64000,  # Max output tokens for Haiku 4.5
        cost_per_million_tokens=5.0,  # Haiku 4.5 pricing
        temperature=1.0,
    ),
    ModelConfig(
        id="claude-opus-4.1",
        name="Claude Opus 4.1 (Legacy)",
        provider="anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        model_id="claude-opus-4-1-20250805",
        max_tokens=32000,  # Max output tokens for Opus 4.1
        cost_per_million_tokens=15.0,  # Opus 4.1 pricing
        temperature=1.0,
    ),
    ModelConfig(
        id="claude-sonnet-4",
        name="Claude Sonnet 4 (Legacy)",
        provider="anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        model_id="claude-sonnet-4-20250514",
        max_tokens=64000,  # Max output tokens for Sonnet 4
        cost_per_million_tokens=3.0,  # Sonnet 4 pricing
        temperature=1.0,
    ),
    ModelConfig(
        id="claude-3-haiku",
        name="Claude Haiku 3 (Legacy)",
        provider="anthropic",
        api_key_env_var="ANTHROPIC_API_KEY",
        model_id="claude-3-haiku-20240307",
        max_tokens=4096,  # Max output tokens for Haiku 3
        cost_per_million_tokens=0.25,  # Haiku 3 pricing
        temperature=1.0,
    ),
    # Google Gemini models (verified available via API)
    ModelConfig(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="google",
        api_key_env_var="GOOGLE_API_KEY",
        model_id="gemini-2.5-pro",
        max_tokens=1000000,
        cost_per_million_tokens=2.5,  # Gemini 2.5 Pro pricing
        temperature=1.0,
    ),
    ModelConfig(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="google",
        api_key_env_var="GOOGLE_API_KEY",
        model_id="gemini-2.0-flash",
        max_tokens=1000000,
        cost_per_million_tokens=0.075,  # Gemini 2.0 Flash pricing
        temperature=1.0,
    ),
    ModelConfig(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="google",
        api_key_env_var="GOOGLE_API_KEY",
        model_id="gemini-2.5-flash",
        max_tokens=1000000,
        cost_per_million_tokens=0.075,  # Gemini 2.5 Flash pricing
        temperature=1.0,
    ),
    # Ollama local models (no API key required, requires Ollama running locally)
    # Install: https://ollama.ai — then: ollama pull <model_id>
    ModelConfig(
        id="ollama-llama3.1-8b",
        name="Llama 3.1 8B (Local)",
        provider="ollama",
        api_key_env_var=None,
        model_id="llama3.1:8b",
        max_tokens=4096,
        temperature=0.0,
        cost_per_million_tokens=0.0,
    ),
    ModelConfig(
        id="ollama-llama3.1-70b",
        name="Llama 3.1 70B (Local)",
        provider="ollama",
        api_key_env_var=None,
        model_id="llama3.1:70b",
        max_tokens=4096,
        temperature=0.0,
        cost_per_million_tokens=0.0,
    ),
    ModelConfig(
        id="ollama-qwen2.5-7b",
        name="Qwen 2.5 7B (Local)",
        provider="ollama",
        api_key_env_var=None,
        model_id="qwen2.5:7b",
        max_tokens=4096,
        temperature=0.0,
        cost_per_million_tokens=0.0,
    ),
    ModelConfig(
        id="ollama-qwen2.5-72b",
        name="Qwen 2.5 72B (Local)",
        provider="ollama",
        api_key_env_var=None,
        model_id="qwen2.5:72b",
        max_tokens=4096,
        temperature=0.0,
        cost_per_million_tokens=0.0,
    ),
    ModelConfig(
        id="ollama-deepseek-r1-8b",
        name="DeepSeek R1 8B (Local)",
        provider="ollama",
        api_key_env_var=None,
        model_id="deepseek-r1:8b",
        max_tokens=4096,
        temperature=0.0,
        cost_per_million_tokens=0.0,
    ),
]

# Local models subset for quick reference
LOCAL_MODELS = [m for m in DEFAULT_MODELS if m.provider == "ollama"]


class ConfigManager:
    """Manages configuration from multiple sources with precedence"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = (
            config_path
            or Path(inspect.getfile(self.__class__)).parent
            / "config"
            / "local_secrets.yaml"
        )
        # config.py is in data/financereasoning/evaluation/
        # project_root is two levels up: evaluation -> financereasoning (contains raw/FinanceReasoning/)
        self.project_root = Path(inspect.getfile(self.__class__)).parent.parent

    def load_env_vars(self) -> Dict[str, str]:
        """Load API keys from environment variables"""
        env_vars = {}

        # OpenAI
        if openai_key := os.environ.get("OPENAI_API_KEY"):
            env_vars["OPENAI_API_KEY"] = openai_key
            print(f"[OK] OpenAI API key found (length: {len(openai_key)})")
        else:
            print("[WARN] OPENAI_API_KEY not found in environment")

        # Anthropic
        if anthropic_key := os.environ.get("ANTHROPIC_API_KEY"):
            env_vars["ANTHROPIC_API_KEY"] = anthropic_key
            print(f"[OK] Anthropic API key found (length: {len(anthropic_key)})")
        else:
            print("[WARN] ANTHROPIC_API_KEY not found in environment")

        # Google
        if google_key := os.environ.get("GOOGLE_API_KEY"):
            env_vars["GOOGLE_API_KEY"] = google_key
            print(f"[OK] Google API key found (length: {len(google_key)})")
        else:
            print("[WARN] GOOGLE_API_KEY not found in environment")

        return env_vars

    def load_config_file(self) -> Dict[str, Any]:
        """Load configuration from YAML config file"""
        if not self.config_path.exists():
            print(f"[INFO] Config file not found: {self.config_path}")
            return {}

        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                print(f"[OK] Config loaded from {self.config_path}")
                return config
            except yaml.YAMLError as e:
                print(f"[ERROR] Error loading config: {e}")
                return {}

    def load_all(self) -> Dict[str, Any]:
        """Load config from all sources with precedence: env vars > config file"""

        # 1. Load environment variables (highest precedence)
        env_vars = self.load_env_vars()

        # 2. Load config file (lower precedence)
        config_file = self.load_config_file()

        # Merge: env vars override config file values
        # Config structure:
        # models:
        #   - id: model_id
        #   overrides:
        #     provider: provider_name
        #     max_tokens: N
        #     temperature: 0.7
        # evaluation:
        #   max_cost_usd: 50.0
        #   concurrency: 2

        merged_config = {**config_file, **env_vars}

        return merged_config

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Get configuration for a specific model ID"""

        # Load all configuration
        all_config = self.load_all()

        # Find model in default models
        model = next((m for m in DEFAULT_MODELS if m.id == model_id), None)

        if model is None:
            raise ValueError(f"Unknown model ID: {model_id}")

        # Apply config overrides
        overrides = all_config.get("models", {}).get(model_id, {})

        # Override provider settings
        if "provider" in overrides:
            provider = overrides["provider"]
            if provider in DEFAULT_PROVIDERS:
                provider_config = DEFAULT_PROVIDERS[provider]
                model.provider = provider
                if "api_key" in overrides:
                    model.api_key_env_var = overrides["api_key"]
                if "base_url" in overrides:
                    provider_config.base_url = overrides["base_url"]
                if "max_tokens" in overrides:
                    model.max_tokens = overrides["max_tokens"]
                if "temperature" in overrides:
                    model.temperature = overrides["temperature"]

        # Override model settings
        if "cost_per_million_tokens" in overrides:
            model.cost_per_million_tokens = overrides["cost_per_million_tokens"]
        if "model_id" in overrides:
            model.model_id = overrides["model_id"]

        return model

    def get_models_for_evaluation(self) -> List[ModelConfig]:
        """Get all models configured for evaluation"""

        all_config = self.load_all()

        # Get evaluation settings
        evaluation_config = all_config.get("evaluation", {})

        # Use default models
        models_to_eval = DEFAULT_MODELS.copy()

        # Models are at top level of YAML, not under 'evaluation'
        model_overrides = all_config.get("models", {})

        for model_id, overrides in model_overrides.items():
            model = next((m for m in models_to_eval if m.id == model_id), None)

            if model is None:
                print(f"[WARN] Unknown model ID in config: {model_id}")
                continue

            # Apply overrides
            if "provider" in overrides:
                provider = overrides["provider"]
                if provider in DEFAULT_PROVIDERS:
                    provider_config = DEFAULT_PROVIDERS[provider]
                    model.provider = provider

            if "model_id" in overrides:
                model.model_id = overrides["model_id"]
            if "max_tokens" in overrides:
                model.max_tokens = overrides["max_tokens"]
            if "temperature" in overrides:
                model.temperature = overrides["temperature"]

        # Filter models: if model_overrides is empty, return all DEFAULT_MODELS
        # otherwise, return only models specified in overrides
        if model_overrides:
            filtered = [m for m in models_to_eval if m.id in model_overrides]
            return filtered
        else:
            return models_to_eval

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration"""

        all_config = self.load_all()
        eval_config_dict = all_config.get("evaluation", {})

        # Debug: print config structure
        print(f"[DEBUG] all_config keys: {list(all_config.keys())}")
        print(f"[DEBUG] eval_config_dict keys: {list(eval_config_dict.keys())}")
        print(
            f"[DEBUG] eval_config_dict['models'] type: {type(eval_config_dict.get('models', {}))}"
        )
        print(
            f"[DEBUG] eval_config_dict['models']: {eval_config_dict.get('models', {})}"
        )

        # Build evaluation config with defaults
        return EvaluationConfig(
            models=self.get_models_for_evaluation(),
            output_dir=eval_config_dict.get("output_dir", "results"),
            max_cost_usd=eval_config_dict.get("max_cost_usd", 100.0),
            max_tokens=eval_config_dict.get("max_tokens"),
            stop_on_budget_exceed=eval_config_dict.get("stop_on_budget_exceed", False),
            concurrency_per_provider=eval_config_dict.get("concurrency", 3),
            retry_attempts=eval_config_dict.get("retry_attempts", 3),
            retry_delay_seconds=eval_config_dict.get("retry_delay_seconds", 1.0),
            cache_enabled=eval_config_dict.get("cache_enabled", True),
            cache_ttl_hours=eval_config_dict.get("cache_ttl_hours", 24),
        )

    def list_available_models(self) -> None:
        """List all available models with their configuration"""

        all_config = self.load_all()
        model_overrides = all_config.get("models", {})

        print("\n" + "=" * 60)
        print("Available Models:")
        print("=" * 60 + "\n")

        for model in DEFAULT_MODELS:
            overrides = model_overrides.get(model.id, {})

            provider_display = model.provider.upper()
            model_display = model.name

            status = "[OK]" if model.api_key_env_var in os.environ else "[NO KEY]"
            api_status = f"API: {status}"

            overrides_display = []
            if "provider" in overrides:
                overrides_display.append(f"provider={overrides['provider']}")
            if "model_id" in overrides:
                overrides_display.append(f"model_id={overrides['model_id']}")
            if "max_tokens" in overrides:
                overrides_display.append(f"max_tokens={overrides['max_tokens']}")
            if "temperature" in overrides:
                overrides_display.append(f"temperature={overrides['temperature']}")

            overrides_str = (
                f", {', '.join(overrides_display)}" if overrides_display else ""
            )

            print(f"  {model.id:10} {model_display:40}")
            print(f"  {status} {api_status:10}")
            print(f"  Provider: {provider_display:20}")
            if overrides_display:
                print(f"  Overrides: {overrides_str}")

        print("\n" + "=" * 60)
        print(f"Environment variables checked: {len(self.load_env_vars())} keys found")
        print(f"Config file: {'found' if self.config_path.exists() else 'not found'}")
        print("\n" + "=" * 60 + "\n")


def create_example_config():
    """Create example config file for users to customize"""

    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / "local_secrets.yaml"

    example_config = f"""# FinanceReasoning Evaluation Configuration
# This file is gitignored - never commit API keys!

# Global evaluation settings
evaluation:
  # Output directory for results
  output_dir: results

  # Cost and token limits
  max_cost_usd: 50.0  # Stop evaluation when total cost exceeds this
  max_tokens: null  # Total token limit (null = no limit)
  stop_on_budget_exceed: false  # Don't stop, just warn
  concurrency: 3  # Concurrent requests per provider
  retry_attempts: 3  # Retry failed requests
  retry_delay_seconds: 1.0  # Delay between retries

# Model-specific overrides
# These override default model settings
# Add your API keys as environment variables instead!
models:
  # Override OpenAI models
  gpt-5-2025-08-07:
    provider: openai
    model_id: gpt-5-2025-08-07
    max_tokens: 128000
    # api_key: ${OPENAI_API_KEY}  # Use env var instead!

  gpt-4o-mini:
    provider: openai
    model_id: gpt-4o-mini
    max_tokens: 128000
    temperature: 0.7
    # api_key: ${OPENAI_API_KEY}  # Use env var instead!

  # Override Anthropic models
  claude-sonnet-4.5:
    provider: anthropic
    model_id: claude-sonnet-4.5-20250514
    max_tokens: 200000
    temperature: 1.0
    # api_key: ${ANTHROPIC_API_KEY}  # Use env var instead!

  claude-sonnet-4.1:
    provider: anthropic
    model_id: claude-sonnet-4.1-20250201
    max_tokens: 200000
    temperature: 1.0
    # api_key: ${ANTHROPIC_API_KEY}  # Use env var instead!
"""

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(example_config)

    print(f"\n[OK] Example config created: {config_path}")
    print(f"  Edit to add your API keys or model preferences")
    print(f"  [WARN] CRITICAL: Do not commit this file with actual API keys!")
    print(f"  Add to .gitignore: echo 'config/local_secrets.yaml' >> .gitignore")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        create_example_config()
    else:
        manager = ConfigManager()
        manager.list_available_models()
