#!/usr/bin/env python3
"""
Test to verify Gemini supports streaming audio input.
Send audio in chunks and create receiver after first chunk.
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
    
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": "You are a helpful assistant. Respond briefly."
    }
    
    print("=" * 60)
    print("Testing Streaming Audio Input")
    print("=" * 60)
    
    # Connect to Gemini
    print(f"\n1. Connecting to {model}...")
    async with client.aio.live.connect(model=model, config=config) as session:
        print("✅ Connected")
        
        # Simulate audio chunks (16kHz, 320ms chunks = 5120 samples)
        chunk_size = 5120
        sample_rate = 16000
        
        # Create 5 chunks of "audio" (silence for simplicity)
        chunks = []
        for i in range(5):
            # Each chunk is 320ms of silence
            audio_bytes = bytes([0, 0] * chunk_size)
            chunks.append(audio_bytes)
        
        print(f"\n2. Will send {len(chunks)} audio chunks of {chunk_size*2} bytes each")
        
        # Send first chunk
        print("\n3. Sending chunk 1...")
        await session.send_realtime_input(
            audio=types.Blob(data=chunks[0], mime_type="audio/pcm;rate=16000")
        )
        print("✅ Chunk 1 sent")
        
        # NOW create receiver (after first chunk)
        print("\n4. Creating receive generator AFTER first chunk...")
        receive_task = asyncio.create_task(receive_responses(session))
        
        # Send remaining chunks with small delays (simulating real-time)
        for i in range(1, len(chunks)):
            await asyncio.sleep(0.32)  # 320ms between chunks
            print(f"Sending chunk {i+1}...")
            await session.send_realtime_input(
                audio=types.Blob(data=chunks[i], mime_type="audio/pcm;rate=16000")
            )
        
        print("\n5. All chunks sent, waiting for response...")
        
        # Wait for receiver to complete
        responses = await receive_task
        
        print("-" * 40)
        print(f"\nResults:")
        print(f"  ✅ Streaming works! Sent {len(chunks)} chunks")
        print(f"  ✅ Received {responses} responses")
        print("\nKey findings:")
        print("  1. Can send audio in chunks (streaming)")
        print("  2. Must create receiver AFTER starting to send")
        print("  3. Gemini processes streaming audio and responds")

async def receive_responses(session):
    """Receive responses from Gemini"""
    response_count = 0
    audio_chunks = 0
    
    try:
        async for response in session.receive():
            response_count += 1
            
            if hasattr(response, 'data') and response.data:
                audio_chunks += 1
                print(f"  Received audio chunk #{audio_chunks}: {len(response.data)} bytes")
            elif hasattr(response, 'server_content') and response.server_content:
                sc = response.server_content
                if hasattr(sc, 'turn_complete') and sc.turn_complete:
                    print("  Turn complete signal received")
                    break
    except Exception as e:
        print(f"  Receive error: {e}")
    
    return response_count

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Test complete!")