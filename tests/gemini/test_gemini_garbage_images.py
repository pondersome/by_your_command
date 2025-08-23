#!/usr/bin/env python3
"""
Test Gemini Live API image validation behavior with garbage data
This script intentionally sends invalid image data to understand:
1. What validation Gemini performs
2. What error messages indicate failed image processing  
3. Whether valid images are actually being processed

Author: Karim Virani
Date: August 2025
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

from google import genai
from google.genai import types

# Test result tracking
test_results = []

async def test_image_data(
    client: genai.Client,
    model: str,
    test_name: str,
    image_data: bytes,
    mime_type: str,
    description: str
) -> Dict[str, Any]:
    """
    Test sending image data to Gemini and capture response
    
    Returns dict with test results including any errors or responses
    """
    result = {
        'test_name': test_name,
        'description': description,
        'data_size': len(image_data),
        'mime_type': mime_type,
        'success': False,
        'error': None,
        'response_text': None,
        'response_metadata': [],
        'mentions_visual': False,
        'generic_response': False
    }
    
    print(f"\n{'='*60}")
    print(f"🧪 Test: {test_name}")
    print(f"📝 Description: {description}")
    print(f"📊 Data size: {len(image_data)} bytes, MIME: {mime_type}")
    
    try:
        # Create a new session for each test (clean state)
        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": "You are a helpful assistant that describes images."
        }
        
        async with client.aio.live.connect(model=model, config=config) as session:
            print("✅ Session connected")
            
            # Send the image with a standard prompt
            try:
                content = types.Content(
                    parts=[
                        types.Part.from_bytes(
                            data=image_data,
                            mime_type=mime_type
                        ),
                        types.Part(text="What do you see in this image? Be specific about objects, colors, and details.")
                    ]
                )
                
                # Try to send the content
                send_result = await session.send_client_content(
                    turns=content,
                    turn_complete=True
                )
                print(f"📤 Send result: {send_result}")
                
            except Exception as send_error:
                result['error'] = f"Send error: {str(send_error)}"
                print(f"❌ Failed to send: {send_error}")
                return result
            
            # Collect responses
            print("📥 Collecting responses...")
            response_count = 0
            full_text = ""
            
            try:
                async for response in session.receive():
                    response_count += 1
                    
                    # Collect text responses
                    if hasattr(response, 'text') and response.text:
                        full_text += response.text
                        print(f"   Response {response_count}: {response.text[:100]}...")
                    
                    # Check for metadata
                    if hasattr(response, 'server_content') and response.server_content:
                        sc = response.server_content
                        metadata = {}
                        
                        if hasattr(sc, 'turn_complete'):
                            metadata['turn_complete'] = sc.turn_complete
                        if hasattr(sc, 'interrupted'):
                            metadata['interrupted'] = sc.interrupted
                        if hasattr(sc, 'model_turn') and sc.model_turn:
                            metadata['has_model_turn'] = True
                            
                        if metadata:
                            result['response_metadata'].append(metadata)
                        
                        # Stop on turn complete
                        if hasattr(sc, 'turn_complete') and sc.turn_complete:
                            print("   ✅ Turn complete signal received")
                            break
                    
                    # Safety limit
                    if response_count > 50:
                        print("   ⏰ Stopping after 50 responses")
                        break
                        
            except Exception as receive_error:
                result['error'] = f"Receive error: {str(receive_error)}"
                print(f"❌ Error receiving responses: {receive_error}")
            
            result['response_text'] = full_text
            result['success'] = bool(full_text) and not result['error']
            
            # Analyze response content
            if full_text:
                lower_text = full_text.lower()
                
                # Check if response mentions visual content
                visual_keywords = ['see', 'image', 'photo', 'picture', 'shows', 'appears', 
                                  'visible', 'color', 'object', 'background', 'foreground',
                                  'left', 'right', 'center', 'top', 'bottom']
                result['mentions_visual'] = any(keyword in lower_text for keyword in visual_keywords)
                
                # Check for generic/evasive responses
                generic_phrases = ["i cannot see", "i don't see", "no image", "can't process",
                                 "unable to view", "don't have access", "cannot access"]
                result['generic_response'] = any(phrase in lower_text for phrase in generic_phrases)
                
                print(f"\n📊 Response Analysis:")
                print(f"   Mentions visual content: {result['mentions_visual']}")
                print(f"   Generic/evasive response: {result['generic_response']}")
                
    except Exception as e:
        result['error'] = f"Session error: {str(e)}"
        print(f"❌ Session error: {e}")
        traceback.print_exc()
    
    return result


async def main():
    print("🔬 Gemini Live API Image Validation Test")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        return
    
    # Initialize client
    client = genai.Client(
        api_key=api_key,
        http_options={'api_version': 'v1beta'}
    )
    
    model = "models/gemini-2.0-flash-live-001"
    print(f"📡 Using model: {model}")
    
    # Test data directory
    media_dir = Path(__file__).parent.parent / "media"
    
    # Prepare test cases
    test_cases = []
    
    # 1. Valid JPEG image
    jpg_path = media_dir / "tablethings.jpg"
    if jpg_path.exists():
        with open(jpg_path, 'rb') as f:
            jpg_data = f.read()
        test_cases.append({
            'name': 'valid_jpeg',
            'data': jpg_data,
            'mime_type': 'image/jpeg',
            'description': 'Valid JPEG image (tablethings.jpg)'
        })
    
    # 2. Valid PNG image
    png_path = media_dir / "whatisthis1.png"
    if png_path.exists():
        with open(png_path, 'rb') as f:
            png_data = f.read()
        test_cases.append({
            'name': 'valid_png',
            'data': png_data,
            'mime_type': 'image/png',
            'description': 'Valid PNG image (whatisthis1.png)'
        })
    
    # 3. Random garbage bytes
    test_cases.append({
        'name': 'random_bytes',
        'data': os.urandom(1024),
        'mime_type': 'image/jpeg',
        'description': 'Random 1KB garbage with image/jpeg MIME'
    })
    
    # 4. Text as image
    test_cases.append({
        'name': 'text_as_image',
        'data': b"This is definitely not an image, just plain text!",
        'mime_type': 'image/png',
        'description': 'Plain text with image/png MIME type'
    })
    
    # 5. Wrong MIME type (if we have valid image)
    if jpg_path.exists():
        test_cases.append({
            'name': 'wrong_mime',
            'data': jpg_data,
            'mime_type': 'text/plain',
            'description': 'Valid JPEG with text/plain MIME type'
        })
    
    # 6. Truncated image (corrupted)
    if jpg_path.exists():
        test_cases.append({
            'name': 'truncated_image',
            'data': jpg_data[:100],  # Only first 100 bytes
            'mime_type': 'image/jpeg',
            'description': 'Truncated JPEG (first 100 bytes only)'
        })
    
    # 7. Empty data
    test_cases.append({
        'name': 'empty_data',
        'data': b'',
        'mime_type': 'image/jpeg',
        'description': 'Empty data with image/jpeg MIME'
    })
    
    # 8. Large garbage (1MB)
    test_cases.append({
        'name': 'large_garbage',
        'data': os.urandom(1024 * 1024),  # 1MB
        'mime_type': 'image/jpeg',
        'description': '1MB of random data with image/jpeg MIME'
    })
    
    # 9. Invalid JPEG header
    test_cases.append({
        'name': 'bad_jpeg_header',
        'data': b'\xFF\xD9' + os.urandom(1022),  # JPEG end marker at start
        'mime_type': 'image/jpeg',
        'description': 'Data with invalid JPEG header'
    })
    
    # 10. Python bytecode as image
    pyc_path = Path(__file__).parent / "__pycache__"
    pyc_files = list(pyc_path.glob("*.pyc")) if pyc_path.exists() else []
    if pyc_files:
        with open(pyc_files[0], 'rb') as f:
            pyc_data = f.read()[:1024]  # First 1KB
        test_cases.append({
            'name': 'python_bytecode',
            'data': pyc_data,
            'mime_type': 'image/png',
            'description': 'Python bytecode with image/png MIME'
        })
    
    # Run all tests
    print(f"\n🧪 Running {len(test_cases)} test cases...")
    
    for test_case in test_cases:
        result = await test_image_data(
            client=client,
            model=model,
            test_name=test_case['name'],
            image_data=test_case['data'],
            mime_type=test_case['mime_type'],
            description=test_case['description']
        )
        test_results.append(result)
        
        # Brief pause between tests
        await asyncio.sleep(2)
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    # Group results
    valid_images = [r for r in test_results if 'valid' in r['test_name']]
    garbage_data = [r for r in test_results if 'valid' not in r['test_name']]
    
    print("\n✅ VALID IMAGES:")
    for result in valid_images:
        status = "✓" if result['success'] and result['mentions_visual'] else "✗"
        print(f"  {status} {result['test_name']}:")
        print(f"     Success: {result['success']}")
        print(f"     Mentions visual: {result['mentions_visual']}")
        print(f"     Generic response: {result['generic_response']}")
        if result['error']:
            print(f"     Error: {result['error']}")
        if result['response_text']:
            preview = result['response_text'][:150].replace('\n', ' ')
            print(f"     Response: {preview}...")
    
    print("\n🗑️ GARBAGE DATA:")
    for result in garbage_data:
        status = "!" if result['error'] else ("?" if result['generic_response'] else "✓")
        print(f"  {status} {result['test_name']}:")
        print(f"     Success: {result['success']}")
        print(f"     Error: {result['error'] if result['error'] else 'None'}")
        print(f"     Generic response: {result['generic_response']}")
        if result['response_text'] and not result['generic_response']:
            preview = result['response_text'][:100].replace('\n', ' ')
            print(f"     Response: {preview}...")
    
    # Key findings
    print("\n🔍 KEY FINDINGS:")
    
    # Check if valid images work
    valid_working = [r for r in valid_images if r['success'] and r['mentions_visual']]
    if valid_working:
        print(f"✅ {len(valid_working)}/{len(valid_images)} valid images got visual responses")
    else:
        print(f"❌ No valid images got proper visual responses!")
    
    # Check if garbage causes errors
    garbage_errors = [r for r in garbage_data if r['error']]
    if garbage_errors:
        print(f"✅ {len(garbage_errors)}/{len(garbage_data)} garbage inputs caused errors")
        # Show unique error types
        error_types = set(r['error'].split(':')[0] for r in garbage_errors if r['error'])
        print(f"   Error types: {', '.join(error_types)}")
    else:
        print(f"⚠️ No garbage data caused errors - might indicate no validation!")
    
    # Check for generic responses
    garbage_generic = [r for r in garbage_data if r['generic_response']]
    if garbage_generic:
        print(f"ℹ️ {len(garbage_generic)}/{len(garbage_data)} garbage inputs got generic responses")
    
    # Save detailed results
    output_path = Path(__file__).parent / "garbage_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {output_path}")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("🎯 VERDICT:")
    
    if valid_working and garbage_errors:
        print("✅ Gemini DOES validate images and process them correctly!")
        print("   → Valid images work, garbage causes errors")
    elif valid_working and not garbage_errors:
        print("⚠️ Gemini processes valid images but doesn't validate format")
        print("   → Might accept anything that looks like binary data")
    elif not valid_working and garbage_errors:
        print("❌ Gemini validates format but isn't processing images properly")
        print("   → Something wrong with how we send valid images")
    else:
        print("❌ Gemini is NOT processing images at all!")
        print("   → Both valid and garbage get similar treatment")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Test complete!")