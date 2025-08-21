#!/usr/bin/env python3
"""
Test creating receive generator AFTER sending input.
"""

import asyncio
import os
from google import genai
from google.genai import types

async def main():
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        return
    
    # Configure client
    client = genai.Client(api_key=api_key)
    model = "models/gemini-2.0-flash-live-001"
    
    # Simple config
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": "You are a helpful assistant. Respond briefly to any input."
    }
    
    print("=" * 60)
    print("Testing Generator Creation AFTER Sending")
    print("=" * 60)
    
    # Connect to Gemini
    print(f"\n1. Connecting to {model}...")
    async with client.aio.live.connect(model=model, config=config) as session:
        print("✅ Connected")
        
        # Send text FIRST
        print("\n2. Sending text input FIRST...")
        await session.send_client_content(
            turns=types.Content(
                parts=[types.Part(text="Hello, please respond with a short greeting")]
            ),
            turn_complete=True
        )
        print("✅ Text sent")
        
        # NOW create the receive generator
        print("\n3. Creating receive generator AFTER sending...")
        receive_gen = session.receive()
        print(f"✅ Generator created: {type(receive_gen)}")
        
        # Try to receive responses
        print("\n4. Attempting to receive responses...")
        print("-" * 40)
        
        responses_received = 0
        audio_chunks = 0
        
        try:
            async for response in receive_gen:
                responses_received += 1
                
                if hasattr(response, 'data') and response.data:
                    audio_chunks += 1
                    print(f"Response #{responses_received}: Audio ({len(response.data)} bytes)")
                elif hasattr(response, 'text') and response.text:
                    print(f"Response #{responses_received}: Text: '{response.text}'")
                elif hasattr(response, 'server_content') and response.server_content:
                    sc = response.server_content
                    if hasattr(sc, 'turn_complete') and sc.turn_complete:
                        print(f"Response #{responses_received}: Turn complete")
                        break
                    elif hasattr(sc, 'generation_complete') and sc.generation_complete:
                        print(f"Response #{responses_received}: Generation complete")
                else:
                    print(f"Response #{responses_received}: {type(response).__name__}")
                
                # Safety limit
                if responses_received > 100:
                    print("Stopping after 100 responses")
                    break
                    
        except StopAsyncIteration:
            print(f"\n📭 Generator exhausted")
        
        print("-" * 40)
        print(f"\nSummary:")
        print(f"  Total responses: {responses_received}")
        print(f"  Audio chunks: {audio_chunks}")
        
        if responses_received > 0:
            print("\n✅ SUCCESS! Got responses when generator created AFTER sending")
            print("\n🔑 Key finding: The receive generator should be created")
            print("   AFTER sending input, not before!")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Test complete!")