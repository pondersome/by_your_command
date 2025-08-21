#!/usr/bin/env python3
"""
Test what signals Gemini sends when audio response is complete
"""

import asyncio
import os
from google import genai
from google.genai import types

api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1beta'}
)

model = "models/gemini-2.0-flash-live-001"

async def main():
    print("🔍 Testing for audio completion signals")
    print("="*50)
    
    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": "Give a very brief response."
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        print("✅ Connected")
        
        # Send a simple request for a short response
        await session.send_client_content(
            turns=types.Content(
                parts=[types.Part(text="Say 'Hello, testing complete' and nothing more.")]
            ),
            turn_complete=True
        )
        print("📤 Sent request for SHORT audio response")
        print("\nMonitoring response stream:")
        print("-" * 50)
        
        response_count = 0
        audio_chunks = 0
        last_10_responses = []
        
        try:
            async for response in session.receive():
                response_count += 1
                
                # Track last 10 responses for pattern analysis
                response_info = {
                    'num': response_count,
                    'has_data': False,
                    'has_text': False,
                    'attributes': []
                }
                
                # Check what's in the response
                if hasattr(response, 'data') and response.data:
                    audio_chunks += 1
                    response_info['has_data'] = True
                    print(f"  Response {response_count}: AUDIO ({len(response.data)} bytes)")
                else:
                    # Non-audio response - let's see what it is
                    print(f"  Response {response_count}: NO AUDIO")
                    
                    # Check all attributes
                    for attr in dir(response):
                        if not attr.startswith('_'):
                            value = getattr(response, attr, None)
                            if value is not None and not callable(value):
                                response_info['attributes'].append(attr)
                                
                                # Print specific interesting attributes
                                if attr in ['text', 'turn_complete', 'server_content', 'usage_metadata']:
                                    print(f"    - {attr}: {value}")
                                elif attr == 'server_content' and value:
                                    # Check server_content details
                                    if hasattr(value, 'turn_complete'):
                                        print(f"    - server_content.turn_complete: {value.turn_complete}")
                                    if hasattr(value, 'model_turn'):
                                        if value.model_turn:
                                            print(f"    - server_content.model_turn exists")
                                            if hasattr(value.model_turn, 'parts'):
                                                print(f"      Parts: {len(value.model_turn.parts)}")
                
                if hasattr(response, 'text') and response.text:
                    response_info['has_text'] = True
                    print(f"    Text: '{response.text}'")
                
                # Store for pattern analysis
                last_10_responses.append(response_info)
                if len(last_10_responses) > 10:
                    last_10_responses.pop(0)
                
                # Stop if we see a clear pattern of no more audio
                if audio_chunks > 0:  # Got some audio
                    # Count consecutive non-audio responses
                    consecutive_no_audio = 0
                    for r in reversed(last_10_responses):
                        if not r['has_data']:
                            consecutive_no_audio += 1
                        else:
                            break
                    
                    if consecutive_no_audio >= 5:
                        print(f"\n🔔 Detected {consecutive_no_audio} consecutive non-audio responses")
                        print("   This might indicate audio is complete")
                        break
                
                # Safety limit
                if response_count > 200:
                    print("\n⏰ Stopping after 200 responses")
                    break
                    
        except StopAsyncIteration:
            print("\n📭 Stream ended with StopAsyncIteration")
        except Exception as e:
            print(f"\n❌ Stream ended with error: {e}")
        
        print("-" * 50)
        print(f"\n📊 Summary:")
        print(f"  Total responses: {response_count}")
        print(f"  Audio chunks: {audio_chunks}")
        print(f"  Non-audio responses: {response_count - audio_chunks}")
        
        # Analyze the pattern at the end
        print(f"\n🔍 Last 5 responses pattern:")
        for r in last_10_responses[-5:]:
            desc = f"  Response {r['num']}: "
            if r['has_data']:
                desc += "AUDIO"
            elif r['has_text']:
                desc += "TEXT"
            else:
                desc += "METADATA"
            if r['attributes']:
                interesting = [a for a in r['attributes'] if a not in ['model_config', 'model_fields', 'model_fields_set', 'construct', 'copy', 'dict', 'json']]
                if interesting:
                    desc += f" ({', '.join(interesting[:3])})"
            print(desc)

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Done!")