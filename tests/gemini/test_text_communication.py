#!/usr/bin/env python3
"""
Working Gemini Live API Test
Now that we know the model connects, test actual communication
"""

import asyncio
import os
from google import genai
from google.genai import types

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1beta'}
)

# Use a model we know connects
model = "models/gemini-2.0-flash-live-001"

config = {
    "response_modalities": ["TEXT"],
    "system_instruction": "You are a helpful assistant. Please respond briefly.",
}

async def main():
    print(f"🚀 Connecting to {model}...")
    
    async with client.aio.live.connect(model=model, config=config) as session:
        print("✅ Connected!")
        
        # Send text using the documented approach
        print("\n📤 Sending text message...")
        
        # send_client_content takes keyword arguments directly
        await session.send_client_content(
            turns=types.Content(
                parts=[types.Part(text="Hello! Can you hear me? Please respond with 'Yes, I can hear you'.")]
            ),
            turn_complete=True
        )
        print("✅ Message sent!")
        
        print("\n⏳ Waiting for responses...")
        response_count = 0
        full_text = ""
        
        # Use a timeout approach
        start_time = asyncio.get_event_loop().time()
        timeout = 10  # 10 seconds
        
        try:
            async for response in session.receive():
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    print(f"⏰ Timeout after {timeout}s")
                    break
                
                response_count += 1
                print(f"\n📥 Response {response_count}:")
                print(f"  Type: {type(response).__name__}")
                
                # Check for text
                if hasattr(response, 'text') and response.text:
                    print(f"  Text: '{response.text}'")
                    full_text += response.text
                
                # Check for server_content
                if hasattr(response, 'server_content') and response.server_content:
                    print(f"  Has server_content")
                    if hasattr(response.server_content, 'model_turn'):
                        print(f"    Has model_turn")
                
                # Show non-None attributes
                attrs = [a for a in dir(response) if not a.startswith('_')]
                non_none = [a for a in attrs if getattr(response, a, None) is not None]
                if non_none:
                    print(f"  Non-None attributes: {non_none}")
                
                # Stop if we got a complete response
                if full_text and (elapsed > 2 or response_count > 10):
                    print("\n✅ Got response, stopping")
                    break
                    
        except StopAsyncIteration:
            print("📭 Response stream ended")
        
        print(f"\n📊 Summary:")
        print(f"  Responses: {response_count}")
        print(f"  Full text: '{full_text}'")
        
        if full_text:
            print("\n🎉 SUCCESS! Gemini responded!")
        else:
            print("\n❌ No text response received")

if __name__ == "__main__":
    asyncio.run(main())