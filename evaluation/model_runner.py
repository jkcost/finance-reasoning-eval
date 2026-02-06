"""
ModelRunner - Execute LLM queries with async concurrency and rate limiting

Supports:
- Multiple provider APIs (OpenAI, Anthropic, Google, etc.)
- Per-provider concurrency limits (semaphores)
- Rate limiting (token buckets, leaky buckets)
- Retry with exponential backoff
- Cost tracking (prompt + completion tokens)
- Budget guards (max_cost, max_tokens)
"""

import asyncio
import json
import time
import random
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import httpx

# Import our modules
from config import ConfigManager, ModelConfig, EvaluationConfig
from dataset_loader import Example


@dataclass
class UsageStats:
    """Token usage and cost tracking"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class ProviderSemaphore:
    """Per-provider concurrency control"""

    semaphore: asyncio.Semaphore
    tokens_per_second: float
    last_reset: datetime

    async def __aenter__(self):
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()
        return False


@dataclass
class LLMResponse:
    """Parsed LLM response with usage info"""

    model_id: str
    example_id: str
    parse_status: str = "success"  # "success", "partial", "failed"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    response_time_seconds: float = 0.0
    raw_response: str = ""
    parsed_data: Optional[Dict[str, Any]] = None
    final_answer: Any = None  # Extracted final answer from parsed response


# Base Provider Adapter
class BaseProvider:
    """Base class for all LLM providers"""

    def __init__(self, model: ModelConfig, concurrency_limit: int = 3):
        self.model = model
        self.client = None
        now = datetime.now()
        # Semaphore for concurrency control (default 3 concurrent requests)
        self.semaphore = ProviderSemaphore(
            semaphore=asyncio.Semaphore(concurrency_limit),
            tokens_per_second=100.0,  # Rate limit: 100 tokens/second per model
            last_reset=now,
        )
        self.last_reset = now
        # Track total tokens used across all requests
        self.total_tokens_used = 0

    async def _request_with_retry(self, method: str, **kwargs) -> Dict[str, Any]:
        """Execute request with retry logic"""

        config = self.model
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                response = await method(**kwargs)
                # Note: Semaphore is released by caller
                return {
                    "status": "success",
                    "response": response,
                    "attempt": attempt,
                }
            except Exception as e:
                # Release semaphore on error so other requests can proceed
                self.semaphore.release()

                if attempt < max_attempts:
                    # Exponential backoff
                    delay = (2 ** (attempt - 1)) * 1.0 + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)

                    print(
                        f"  [WARN] Retry {attempt}/{max_attempts} for {config.name} after {delay:.1f}s: {e}"
                    )
                else:
                    # Final attempt failed
                    return {
                        "status": "failed",
                        "error": str(e),
                        "attempt": attempt,
                    }

    def _update_usage(self, response: LLMResponse, prompt_tokens: int):
        """Update usage statistics"""

        response.prompt_tokens = prompt_tokens
        response.completion_tokens = response.get("completion_tokens", 0)

        # Total tokens used
        response.total_tokens = prompt_tokens + response.completion_tokens

        # Calculate cost
        cost_per_million = self.model.cost_per_million_tokens or 15.0
        response.cost_usd = (response.total_tokens / 1_000_000) * cost_per_million

    async def call_model(
        self,
        example: Example,
        prompt: str,
    ) -> LLMResponse:
        """Execute single LLM query"""

        model = self.model

        # Estimate prompt tokens (rough estimate: 1.3 chars per token)
        prompt_tokens_est = len(prompt) * 1.3

        # Note: We don't enforce per-model token limits here.
        # Use max_cost_usd and max_tokens in evaluation_config for budget control.

        # Acquire semaphore (concurrency limit)
        async with self.semaphore:
            # Make provider-specific call
            start_time = time.time()
            result = await self._request_with_retry(self._make_request, prompt=prompt)

        if result["status"] != "success":
            return LLMResponse(
                model_id=model.id,
                example_id=example.id,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                response_time_seconds=0.0,
                parse_status="failed",
                raw_response=str(result.get("error", "Unknown error")),
                parsed_data=None,
            )

        # Parse response and extract usage
        response_data = result.get("response", {})
        raw_response = self._extract_response_text(response_data)

        # Extract token usage (provider-specific)
        usage = self._extract_usage(response_data)

        # Track total tokens used
        total_tokens_used = prompt_tokens_est + usage.get("completion_tokens", 0)
        self.total_tokens_used += total_tokens_used

        response_obj = LLMResponse(
            model_id=model.id,
            example_id=example.id,
            prompt_tokens=usage.get("prompt_tokens", prompt_tokens_est),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=total_tokens_used,
            cost_usd=self._calculate_cost(total_tokens_used),
            response_time_seconds=time.time() - start_time,
            parse_status="success",
            raw_response=raw_response,
            parsed_data=response_data,
        )

        return response_obj

    def _make_request(self, prompt: str) -> Dict[str, Any]:
        """Make actual HTTP request - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _make_request")

    def _extract_response_text(self, response_data: Dict[str, Any]) -> str:
        """Extract response text - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _extract_response_text")

    def _extract_usage(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract token usage - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _extract_usage")

    def _calculate_cost(self, total_tokens: int) -> float:
        """Calculate cost based on tokens used"""
        cost_per_million = self.model.cost_per_million_tokens or 15.0
        return (total_tokens / 1_000_000) * cost_per_million


# OpenAI Provider
class OpenAIProvider(BaseProvider):
    """OpenAI GPT models"""

    def __init__(self, model: ModelConfig, concurrency_limit: int = 3):
        super().__init__(model, concurrency_limit=concurrency_limit)

        # Initialize httpx client
        self.client = httpx.AsyncClient(
            base_url=model.base_url or "https://api.openai.com/v1",
            headers={
                "Authorization": f"Bearer {os.environ.get(model.api_key_env_var, '')}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def _make_request(self, prompt: str) -> Dict[str, Any]:
        """Make OpenAI API request"""

        start_time = time.time()

        # Use max_completion_tokens for newer models (o1, gpt-4o, etc.)
        # max_tokens is deprecated for these models
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": self.model.temperature or 0.7,
                "max_completion_tokens": min(self.model.max_tokens or 4096, 16384),
            },
        )

        print(f"[DEBUG OpenAI HTTP] status: {response.status_code}")
        print(f"[DEBUG OpenAI HTTP] response type: {type(response)}")
        print(f"[DEBUG OpenAI HTTP] headers keys: {list(response.headers.keys())}")

        # Parse response JSON
        response_data = response.json()

        print(f"[DEBUG OpenAI HTTP] response_data keys: {list(response_data.keys())}")
        if "error" in response_data:
            print(f"[DEBUG OpenAI HTTP] error: {response_data['error']}")

        # Extract usage from response body (OpenAI returns usage in response, not headers)
        usage_info = response_data.get("usage", {})
        prompt_tokens = usage_info.get("prompt_tokens", 0)
        completion_tokens = usage_info.get("completion_tokens", 0)

        print(f"[DEBUG OpenAI HTTP] usage: {usage_info}")

        return {
            "response": response_data,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "start_time": start_time,
        }

    def _extract_response_text(self, response_data: Dict[str, Any]) -> str:
        """Extract final answer from OpenAI response"""

        # The actual API response is nested under 'response' key
        # response_data = { 'response': {...}, 'usage': {...}, 'start_time': ... }
        api_response = response_data.get("response", {})

        choices = api_response.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")

        return content.strip()

    def _extract_usage(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage from OpenAI response"""

        # Usage is passed separately in response_data from _make_request
        # response_data = { 'response': {...}, 'usage': {...}, 'start_time': ... }
        return response_data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})


# Anthropic Provider
class AnthropicProvider(BaseProvider):
    """Anthropic Claude models"""

    def __init__(self, model: ModelConfig, concurrency_limit: int = 3):
        super().__init__(model, concurrency_limit=concurrency_limit)

        # Initialize httpx client
        self.client = httpx.AsyncClient(
            base_url=model.base_url or "https://api.anthropic.com",
            headers={
                "x-api-key": os.environ.get(model.api_key_env_var, ""),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def _make_request(self, prompt: str) -> Dict[str, Any]:
        """Make Anthropic API request"""

        start_time = time.time()

        response = await self.client.post(
            "/v1/messages",
            json={
                "model": self.model.model_id,
                "max_tokens": self.model.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": self.model.temperature or 1.0,
            },
        )

        print(f"[DEBUG Anthropic HTTP] status: {response.status_code}")
        print(f"[DEBUG Anthropic HTTP] response type: {type(response)}")

        # Parse response JSON
        response_data = response.json()

        if response.status_code != 200:
            print(f"[DEBUG Anthropic HTTP] error: {response_data}")

        # Anthropic returns usage in response body
        usage_info = response_data.get("usage", {})
        prompt_tokens = usage_info.get("input_tokens", 0)
        completion_tokens = usage_info.get("output_tokens", 0)

        return {
            "response": response_data,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "start_time": start_time,
        }

    def _extract_response_text(self, response_data: Dict[str, Any]) -> str:
        """Extract final answer from Anthropic response"""

        # The actual API response is nested under 'response' key
        # response_data = { 'response': {...}, 'usage': {...}, 'start_time': ... }
        api_response = response_data.get("response", {})

        content_block = api_response.get("content", [{}])

        text = ""
        if content_block and len(content_block) > 0:
            text = content_block[0].get("text", "")

        return text.strip()

    def _extract_usage(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage from Anthropic response"""

        # Usage is passed separately in response_data from _make_request
        # response_data = { 'response': {...}, 'usage': {...}, 'start_time': ... }
        return response_data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})


# Google Provider
class GoogleProvider(BaseProvider):
    """Google Gemini models"""

    def __init__(self, model: ModelConfig, concurrency_limit: int = 3):
        super().__init__(model, concurrency_limit=concurrency_limit)

        # Initialize httpx client
        # For Google API, we use the model ID in the URL path
        self.client = httpx.AsyncClient(
            base_url=model.base_url or "https://generativelanguage.googleapis.com",
            headers={
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def _make_request(self, prompt: str) -> Dict[str, Any]:
        """Make Google Gemini API request"""

        start_time = time.time()

        # Gemini API uses model_id in the URL path: /v1beta/models/{model_id}:generateContent
        # Also need to pass API key as query parameter
        api_key = os.environ.get(self.model.api_key_env_var, "")
        endpoint = f"/v1beta/models/{self.model.model_id}:generateContent?key={api_key}"

        response = await self.client.post(
            endpoint,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": self.model.max_tokens,
                    "temperature": self.model.temperature or 1.0,
                },
            },
        )

        print(f"[DEBUG Google HTTP] status: {response.status_code}")
        print(f"[DEBUG Google HTTP] response type: {type(response)}")

        # Parse response JSON
        response_data = response.json()

        if response.status_code != 200:
            print(f"[DEBUG Google HTTP] error: {response_data}")

        # Gemini returns usage in usageMetadata
        usage_metadata = response_data.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)

        return {
            "response": response_data,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "start_time": start_time,
        }

    def _extract_response_text(self, response_data: Dict[str, Any]) -> str:
        """Extract final answer from Google Gemini response"""

        # The actual API response is nested under 'response' key
        # response_data = { 'response': {...}, 'usage': {...}, 'start_time': ... }
        api_response = response_data.get("response", {})

        candidates = api_response.get("candidates", [])
        if not candidates:
            return ""

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""

        text = parts[0].get("text", "")

        return text.strip()

    def _extract_usage(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage from Google Gemini response"""

        # Usage is passed separately in response_data from _make_request
        # response_data = { 'response': {...}, 'usage': {...}, 'start_time': ... }
        return response_data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})


# Model Runner Orchestrator
class ModelRunner:
    """Orchestrates LLM model evaluation with concurrency control"""

    def __init__(
        self,
        config_manager: ConfigManager,
        evaluation_config: EvaluationConfig,
    ):
        self.config_manager = config_manager
        self.evaluation_config = evaluation_config

        # Use models from evaluation_config, not from config_manager
        model_configs = evaluation_config.models
        if not model_configs:
            raise ValueError("No models configured for evaluation")

        # Initialize providers
        self.providers = {}
        for model_config in model_configs:
            provider = model_config.provider

            if provider == "openai":
                self.providers[model_config.id] = OpenAIProvider(
                    model_config,
                    concurrency_limit=evaluation_config.concurrency_per_provider,
                )
            elif provider == "anthropic":
                self.providers[model_config.id] = AnthropicProvider(
                    model_config,
                    concurrency_limit=evaluation_config.concurrency_per_provider,
                )
            elif provider == "google":
                self.providers[model_config.id] = GoogleProvider(
                    model_config,
                    concurrency_limit=evaluation_config.concurrency_per_provider,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

    async def evaluate_example(
        self,
        example: Example,
        template_type: str = "finance_reasoning_compliant",
    ) -> Dict[str, LLMResponse]:
        """Evaluate single example against configured models"""

        from prompt_builder import PromptBuilder

        # Check if RAG is needed
        is_rag_template = template_type in ["cot_rag", "pot_rag"]
        functions_text = None

        if is_rag_template:
            # Initialize RAG enhancer
            try:
                from rag_enhancer import RAGEnhancer

                # Get path to function retriever
                eval_dir = Path(__file__).parent
                rag_enhancer = RAGEnhancer(eval_dir / "function_retriever.py")

                # Retrieve relevant functions
                query = f"{example.question} {example.context[:200]}"
                functions_text, functions_dict = rag_enhancer.retrieve_and_format(
                    query=query,
                    question=example.question,
                    context=example.context,
                    top_k=5,
                    template_type=template_type,
                )

                if functions_text:
                    print(
                        f"  [RAG] Retrieved {len(functions_dict) if functions_dict else 0} relevant functions"
                    )
                else:
                    print(
                        f"  [RAG] No functions retrieved, falling back to non-RAG mode"
                    )

            except Exception as e:
                print(f"  [WARN] RAG retrieval failed: {e}")
                print(f"  [INFO] Continuing without RAG enhancement")
                functions_text = None

        # Build prompt
        builder = PromptBuilder(template_type)
        prompt = builder.build_prompt(
            question=example.question,
            context=example.context,
            python_solution=example.python_solution,
            functions=functions_text,
        )

        responses = {}

        # Run evaluation for each model
        for model_config in self.config_manager.get_models_for_evaluation():
            provider = self.providers.get(model_config.id)
            if not provider:
                print(f"  [WARN] Skipping {model_config.id}: provider not available")
                continue

            print(f"  → Querying {model_config.name} ({model_config.provider})...")

            try:
                response = await provider.call_model(example, prompt)
                responses[model_config.id] = response

                # Check budget limits
                total_cost = sum(r.cost_usd for r in responses.values())
                if (
                    self.evaluation_config.max_cost_usd
                    and total_cost > self.evaluation_config.max_cost_usd
                ):
                    print(
                        f"    💰 Budget exceeded: ${total_cost:.2f} > ${self.evaluation_config.max_cost_usd}"
                    )
                    if self.evaluation_config.stop_on_budget_exceed:
                        break

                total_tokens = sum(r.total_tokens for r in responses.values())
                if (
                    self.evaluation_config.max_tokens
                    and total_tokens > self.evaluation_config.max_tokens
                ):
                    print(
                        f"    💰 Token limit exceeded: {total_tokens} > {self.evaluation_config.max_tokens}"
                    )
                    if self.evaluation_config.stop_on_budget_exceed:
                        break

            except Exception as e:
                print(f"  ❌ Error with {model_config.name}: {e}")
                responses[model_config.id] = LLMResponse(
                    model_id=model_config.id,
                    example_id=example.id,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    response_time_seconds=0.0,
                    parse_status="failed",
                    raw_response=str(e),
                    parsed_data=None,
                )

        return responses


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ModelRunner - LLM Evaluation Orchestrator")
    print("=" * 60)

    # Test initialization
    try:
        from config import EvaluationConfig

        config_manager = ConfigManager()
        eval_config = EvaluationConfig()

        runner = ModelRunner(config_manager, eval_config)

        # Test with example data
        test_example = Example(
            id="test-001",
            question="Test question",
            context="Test context",
            ground_truth_final=42,
            python_solution="answer = 42",
        )

        print("✓ Initialization successful")
        print(f"  Models configured: {len(runner.providers)}")

    except Exception as e:
        print(f"✗ Initialization error: {e}")
        sys.exit(1)
