#!/usr/bin/env python3
"""
Test to investigate why receive generator returns no responses.
This mimics what the agent does but in isolation.
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
    print("Testing Receive Generator Issue")
    print("=" * 60)
    
    # Connect to Gemini
    print(f"\n1. Connecting to {model}...")
    async with client.aio.live.connect(model=model, config=config) as session:
        print("✅ Connected")
        
        # Wait a bit to ensure session is ready (mimicking agent behavior)
        print("\n2. Waiting 0.5s for session to be ready...")
        await asyncio.sleep(0.5)
        
        # Create the receive generator BEFORE sending audio
        print("\n3. Creating receive generator...")
        receive_gen = session.receive()
        print(f"✅ Generator created: {type(receive_gen)}")
        
        # Now send some audio
        print("\n4. Sending test audio (saying 'hello')...")
        # Create simple test audio: 1 second of silence (to simulate speech)
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)
        audio_bytes = bytes([0, 0] * num_samples)  # Silent PCM16
        
        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )
        print("✅ Audio sent")
        
        # Now try to receive responses
        print("\n5. Attempting to receive responses...")
        print("-" * 40)
        
        responses_received = 0
        timeout_count = 0
        max_timeouts = 20  # 10 seconds total
        
        try:
            while timeout_count < max_timeouts:
                try:
                    # Try to get next response with timeout
                    response = await asyncio.wait_for(
                        receive_gen.__anext__(),
                        timeout=0.5
                    )
                    
                    responses_received += 1
                    print(f"Response #{responses_received}: {type(response).__name__}")
                    
                    # Check what we got
                    if hasattr(response, 'data') and response.data:
                        print(f"  - Audio data: {len(response.data)} bytes")
                    if hasattr(response, 'text') and response.text:
                        print(f"  - Text: {response.text}")
                    if hasattr(response, 'server_content'):
                        print(f"  - Has server_content")
                        
                    timeout_count = 0  # Reset on successful response
                    
                except asyncio.TimeoutError:
                    timeout_count += 1
                    if timeout_count % 4 == 0:  # Log every 2 seconds
                        print(f"No responses for {timeout_count * 0.5}s...")
                        
                except StopAsyncIteration:
                    print(f"\n📭 Generator exhausted after {responses_received} responses")
                    break
                    
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print("-" * 40)
        print(f"\nSummary:")
        print(f"  Responses received: {responses_received}")
        
        if responses_received == 0:
            print("\n🔍 Debugging: Why no responses?")
            print("  1. Session might not be ready despite connection")
            print("  2. Audio might be too quiet/empty to trigger response")
            print("  3. Receive generator might need different timing")
            
            # Try sending text instead
            print("\n6. Trying with text input instead...")
            await session.send_client_content(
                turns=types.Content(
                    parts=[types.Part(text="Hello, can you hear me?")]
                ),
                turn_complete=True
            )
            print("✅ Text sent")
            
            # Try to receive again
            print("\n7. Attempting to receive after text...")
            try:
                response = await asyncio.wait_for(
                    receive_gen.__anext__(),
                    timeout=5.0
                )
                print(f"✅ Got response: {type(response).__name__}")
            except asyncio.TimeoutError:
                print("❌ Still no response after text")
            except StopAsyncIteration:
                print("❌ Generator already exhausted")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Test complete!")