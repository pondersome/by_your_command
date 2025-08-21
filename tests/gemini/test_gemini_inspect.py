#!/usr/bin/env python3
"""
Inspect Gemini session methods to understand the API
"""

import asyncio
import os
import inspect
from google import genai

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1beta'}
)

model = "gemini-2.0-flash-live-001"
config = {"response_modalities": ["TEXT"]}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        print("Session methods:")
        for name in dir(session):
            if not name.startswith('_'):
                attr = getattr(session, name)
                if callable(attr):
                    try:
                        sig = inspect.signature(attr)
                        print(f"  {name}{sig}")
                    except:
                        print(f"  {name}()")
        
        print("\nTrying send_client_content with different approaches:")
        
        # Try with a types object
        from google.genai import types
        
        try:
            print("1. With LiveClientContent...")
            content = types.LiveClientContent(parts=[types.Part(text="Hello")])
            await session.send_client_content(content)
            print("   ✅ Success with LiveClientContent")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Get one response to see what comes back
        print("\nReceiving response...")
        async for response in session.receive():
            print(f"Response type: {type(response).__name__}")
            if hasattr(response, 'text') and response.text:
                print(f"Text: {response.text}")
                break

if __name__ == "__main__":
    asyncio.run(main())