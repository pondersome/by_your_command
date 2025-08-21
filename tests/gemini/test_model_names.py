#!/usr/bin/env python3
"""
Test different Gemini model names to find which one works
"""

import asyncio
import os
from google import genai

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1beta'}
)

# Try different model names from the documentation
models_to_try = [
    "gemini-2.0-flash-live-001",
    "models/gemini-2.0-flash-live-001",  # With 'models/' prefix
    "gemini-live-2.5-flash-preview",
    "gemini-2.5-flash-preview-native-audio-dialog",
]

config = {"response_modalities": ["TEXT"]}

async def test_model(model_name):
    print(f"\nTrying model: {model_name}")
    try:
        # Use wait_for to add timeout
        connect_coro = client.aio.live.connect(model=model_name, config=config).__aenter__()
        session = await asyncio.wait_for(connect_coro, timeout=5.0)
        print(f"  ✅ Connected successfully!")
        await session.__aexit__(None, None, None)
        return True
    except asyncio.TimeoutError:
        print(f"  ⏰ Connection timeout (5s)")
        return False
    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}: {str(e)[:100]}")
        return False

async def main():
    print("Testing Gemini Live model names...")
    print(f"API key ending: ...{api_key[-4:]}")
    
    for model in models_to_try:
        success = await test_model(model)
        if success:
            print(f"\n🎉 Working model found: {model}")
            break
    else:
        print("\n❌ No working models found")

if __name__ == "__main__":
    asyncio.run(main())