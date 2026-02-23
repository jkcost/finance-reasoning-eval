"""
Test Google Gemini API
"""

import httpx
import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Load .env
env_file = Path(__file__).parent / ".env"
load_dotenv(env_file)

api_key = os.environ.get("GOOGLE_API_KEY")
print(f"Google API Key: {api_key[:20]}...{api_key[-20:] if api_key else 'None'}")
print(f"Google API Key Length: {len(api_key) if api_key else 0}\n")

if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in environment!")
    exit(1)

# Test Gemini models
models_to_test = [
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]


async def test_model(client, model_id, api_key):
    """Test if a model is available"""

    # For Gemini API, we use the generateContent endpoint with API key in query parameter
    response = await client.post(
        f"/v1beta/models/{model_id}:generateContent?key={api_key}",
        json={
            "contents": [{"parts": [{"text": "Say 'Hello, World!'"}]}],
            "generationConfig": {
                "maxOutputTokens": 10,
            },
        },
    )

    return {
        "model": model_id,
        "status": response.status_code,
        "success": response.status_code == 200,
    }


async def main():
    client = httpx.AsyncClient(
        base_url=f"https://generativelanguage.googleapis.com",
        headers={
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )

    print("Testing Google Gemini models availability...\n")
    print("=" * 80)

    tasks = [test_model(client, model, api_key) for model in models_to_test]
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
