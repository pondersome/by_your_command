#!/usr/bin/env python3
"""
Test Gemini image input and get COMPLETE audio response
"""

import asyncio
import os
import wave
from pathlib import Path
from google import genai
from google.genai import types

api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1beta'}
)

model = "models/gemini-2.0-flash-live-001"

async def main():
    print("🖼️ Getting COMPLETE audio description of image")
    print("="*50)
    
    image_path = "/home/karim/ros2_ws/src/by_your_command/tests/media/whatisthis1.png"
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    print(f"📂 Loaded image: {len(image_bytes)} bytes")
    
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": "Describe images clearly in spoken form."
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        print("✅ Connected")
        
        # Send image
        content = types.Content(
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                types.Part(text="Please describe what you see in this image in detail.")
            ]
        )
        await session.send_client_content(turns=content, turn_complete=True)
        print("📤 Image sent, waiting for COMPLETE audio response...")
        
        # Prepare output
        output_file = "../output/image_description_complete.wav"
        wf = wave.open(output_file, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        
        audio_bytes_total = 0
        chunk_count = 0
        silence_count = 0
        max_silence = 5  # Allow up to 5 responses without audio
        
        print("\nReceiving audio chunks:")
        print("-" * 40)
        
        timeout = 30  # Give it 30 seconds max
        start_time = asyncio.get_event_loop().time()
        
        try:
            async for response in session.receive():
                elapsed = asyncio.get_event_loop().time() - start_time
                
                # Check for audio data
                if hasattr(response, 'data') and response.data:
                    chunk_count += 1
                    chunk_size = len(response.data)
                    audio_bytes_total += chunk_size
                    wf.writeframes(response.data)
                    
                    # Show progress
                    duration_so_far = audio_bytes_total / (24000 * 2)
                    print(f"  Chunk {chunk_count}: {chunk_size} bytes (total: {duration_so_far:.2f}s)")
                    silence_count = 0  # Reset silence counter
                else:
                    # No audio in this response
                    silence_count += 1
                    if silence_count >= max_silence and audio_bytes_total > 0:
                        print(f"\n✅ No more audio after {silence_count} silent responses")
                        break
                
                # Timeout check
                if elapsed > timeout:
                    print(f"\n⏰ Timeout after {timeout}s")
                    break
                    
        except StopAsyncIteration:
            print("\n📭 Stream ended")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        wf.close()
        
        # Summary
        print("-" * 40)
        if audio_bytes_total > 0:
            final_duration = audio_bytes_total / (24000 * 2)
            print(f"\n✅ SUCCESS!")
            print(f"📊 Total audio: {audio_bytes_total} bytes")
            print(f"⏱️ Duration: {final_duration:.2f} seconds")
            print(f"📁 Saved to: {output_file}")
            print(f"🎯 Chunks received: {chunk_count}")
            
            # Verify the file
            import subprocess
            result = subprocess.run(['file', output_file], capture_output=True, text=True)
            print(f"📋 File check: {result.stdout.strip()}")
            
            # Check if duration is reasonable
            if final_duration < 2.0:
                print("\n⚠️ WARNING: Audio seems too short for a description!")
                print("   This might be incomplete.")
            elif final_duration > 10.0:
                print("\n✅ Good length for a detailed description")
            else:
                print("\n✅ Reasonable length for a description")
        else:
            print("\n❌ No audio received!")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Done!")