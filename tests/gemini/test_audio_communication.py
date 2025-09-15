#!/usr/bin/env python3
"""
Clean test of audio-to-audio with Gemini Live
This time we'll actually save the output properly
"""

import asyncio
import io
import os
from pathlib import Path
import wave
from google import genai
from google.genai import types
import soundfile as sf
import librosa

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1beta'}
)

model = "models/gemini-2.0-flash-live-001"

async def main():
    print("🎤 Audio-to-Audio Test with Gemini Live")
    print("="*50)
    
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": "You are a helpful voice assistant. Listen to what I say and respond naturally with speech.",
    }
    
    # Find an audio file to use as input
    audio_dir = Path("/tmp/prompt_voice/assistant_output")
    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        print("❌ No audio files found")
        return
    
    input_file = str(wav_files[0])
    print(f"📂 Input file: {input_file}")
    
    # Load and convert to 16kHz PCM16
    print("🔄 Converting audio to 16kHz PCM16...")
    buffer = io.BytesIO()
    y, sr = librosa.load(input_file, sr=16000)
    sf.write(buffer, y, sr, format='RAW', subtype='PCM_16')
    buffer.seek(0)
    audio_bytes = buffer.read()
    duration_in = len(audio_bytes) / (16000 * 2)  # 16kHz, 16-bit
    print(f"📊 Input: {len(audio_bytes)} bytes ({duration_in:.2f}s)")
    
    async with client.aio.live.connect(model=model, config=config) as session:
        print("✅ Connected to Gemini Live")
        
        # Send the audio
        print("📤 Sending audio...")
        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )
        print("✅ Audio sent successfully")
        
        # Prepare output file
        output_file = "../output/audio_to_audio_verified.wav"
        wf = wave.open(output_file, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)  # Gemini outputs at 24kHz
        
        print("⏳ Waiting for audio response...")
        print("-" * 40)
        
        audio_bytes_received = 0
        chunk_count = 0
        text_responses = []
        
        # Set timeout
        timeout = 20  # 20 seconds should be plenty
        start_time = asyncio.get_event_loop().time()
        
        try:
            async for response in session.receive():
                elapsed = asyncio.get_event_loop().time() - start_time
                
                # Check for audio data
                if hasattr(response, 'data') and response.data:
                    chunk_count += 1
                    chunk_size = len(response.data)
                    audio_bytes_received += chunk_size
                    wf.writeframes(response.data)
                    print(f"  Chunk {chunk_count}: {chunk_size} bytes")
                
                # Also capture any text
                if hasattr(response, 'text') and response.text:
                    text_responses.append(response.text)
                    print(f"  Text: '{response.text}'")
                
                # Stop conditions
                if elapsed > timeout:
                    print(f"\n⏰ Timeout after {timeout}s")
                    break
                
                # Stop after receiving substantial audio (at least 1 second)
                if audio_bytes_received > 48000:  # 1 second at 24kHz stereo
                    print(f"\n✅ Received enough audio, stopping")
                    break
                    
        except StopAsyncIteration:
            print("\n📭 Response stream ended naturally")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        wf.close()
        
        # Summary
        print("-" * 40)
        if audio_bytes_received > 0:
            duration_out = audio_bytes_received / (24000 * 2)  # 24kHz, 16-bit
            print(f"✅ SUCCESS!")
            print(f"📥 Received: {audio_bytes_received} bytes ({duration_out:.2f}s)")
            print(f"📁 Saved to: {output_file}")
            print(f"🎯 Chunks: {chunk_count}")
            
            # Verify the file
            import subprocess
            result = subprocess.run(['file', output_file], capture_output=True, text=True)
            print(f"📋 File info: {result.stdout.strip()}")
            
            size = os.path.getsize(output_file)
            print(f"💾 File size: {size} bytes")
            
            if text_responses:
                print(f"💬 Also got text: {' '.join(text_responses)}")
        else:
            print("❌ FAILED - No audio received")
            print(f"⏱️ Waited {elapsed:.1f}s")
            if text_responses:
                print(f"💬 Got text instead: {' '.join(text_responses)}")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Test complete!")