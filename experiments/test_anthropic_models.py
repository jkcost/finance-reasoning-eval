"""
Check which Anthropic models are available with the current API key
"""

import httpx
import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Load .env
env_file = Path(__file__).parent / ".env"
load_dotenv(env_file)

api_key = os.environ.get("ANTHROPIC_API_KEY")
print(f"API Key: {api_key[:20]}...{api_key[-20:]}")
print(f"API Key Length: {len(api_key)}\n")

# List of common Anthropic models to test
models_to_test = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
    "claude-sonnet-4.5-20250514",
    "claude-3-5-sonnet",
    "claude-3-sonnet-20240229",
    "claude-opus-4-20250514",
    "claude-opus-3-20250514",
    "claude-3-opus-20240229",
]


async def test_model(client, model_id):
    """Test if a model is available"""

    response = await client.post(
        "/v1/messages",
        json={
            "model": model_id,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    return {
        "model": model_id,
        "status": response.status_code,
        "success": response.status_code == 200,
    }


async def main():
    client = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )

    print("Testing Anthropic models availability...\n")
    print("=" * 80)

    tasks = [test_model(client, model) for model in models_to_test]
    results = await asyncio.gather(*tasks)

    available_models = []
    for result in results:
        status = "[OK]" if result["success"] else "[FAIL]"
        print(f"{status} {result['model']:40} Status: {result['status']}")
        if result["success"]:
            available_models.append(result["model"])

    print("=" * 80)
    print(f"\n[OK] Available models ({len(available_models)}):")
    for model in available_models:
        print(f"  - {model}")

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
