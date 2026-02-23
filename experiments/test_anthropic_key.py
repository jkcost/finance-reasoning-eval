"""
Direct test of Anthropic API key
"""

import httpx
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env
env_file = Path(__file__).parent / ".env"
load_dotenv(env_file)

api_key = os.environ.get("ANTHROPIC_API_KEY")
print(f"API Key: {api_key[:20]}...{api_key[-20:]}")
print(f"API Key Length: {len(api_key)}")


# Make a simple request
async def test_api():
    client = httpx.AsyncClient(
        base_url="https://api.anthropic.com",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )

    response = await client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say 'Hello, World!'"}],
        },
    )

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nSuccess! Response:")
        content = data.get("content", [{}])[0].get("text", "")
        print(f"Content: {content}")
    else:
        print(f"\nError Response:")
        print(response.text)

    await client.aclose()


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_api())
