#!/usr/bin/env python3
"""
Minimal Gemini Live API Test
Simplest possible test to verify the API works
"""

import asyncio
import os
from google import genai

# Get API key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

print(f"Using API key ending in: ...{api_key[-4:]}")

client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1beta'}
)

# Test with the model we're using
model = "gemini-2.0-flash-live-001"

config = {
    "response_modalities": ["TEXT"],  # Start with text only
    "system_instruction": "You are a helpful assistant. Say hello.",
}

async def main():
    print(f"Connecting to model: {model}")
    
    try:
        print("Creating session...")
        async with client.aio.live.connect(model=model, config=config) as session:
            print("✅ Connected!")
            
            # Send a simple text message
            print("Sending text: 'Hello, can you hear me?'")
            # Try just passing the string directly
            await session.send_client_content("Hello, can you hear me?")
            
            print("Waiting for responses...")
            response_count = 0
            
            # Try to get just a few responses
            async for response in session.receive():
                response_count += 1
                print(f"Response {response_count}: {type(response).__name__}")
                
                if hasattr(response, 'text') and response.text:
                    print(f"  Text: {response.text}")
                
                if hasattr(response, 'data') and response.data:
                    print(f"  Data: {len(response.data)} bytes")
                
                # Show all non-private attributes
                attrs = [a for a in dir(response) if not a.startswith('_')]
                non_none = [a for a in attrs if getattr(response, a, None) is not None]
                if non_none:
                    print(f"  Non-None attrs: {non_none}")
                
                # Stop after a few responses
                if response_count >= 5:
                    print("Stopping after 5 responses")
                    break
                    
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())