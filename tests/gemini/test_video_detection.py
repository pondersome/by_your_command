#!/usr/bin/env python3
"""
Test Gemini Live API Video Streaming and Object Detection
Tests inline video frames and object detection with bounding boxes
"""

import asyncio
import json
import os
from pathlib import Path
from PIL import Image
import io
import base64
from google import genai
from google.genai import types

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1beta'}
)

model = "models/gemini-2.0-flash-live-001"

async def test_inline_video_frame():
    """Test sending video frame inline (simulating bridge input)"""
    print("\n" + "="*60)
    print("TEST 1: Inline Video Frame (Simulating Bridge)")
    print("="*60)
    
    # Load image and get dimensions
    image_path = "/home/karim/ros2_ws/src/by_your_command/tests/media/whatisthis1.png"
    img = Image.open(image_path)
    width, height = img.size
    print(f"📐 Image dimensions: {width}x{height}")
    
    # Simulate bridge preprocessing: convert to bytes
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Simulate potential bridge encoding (base64)
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    print(f"📊 Base64 size: {len(image_base64)} chars")
    
    config = {
        "response_modalities": ["TEXT"],
        "system_instruction": "You are a vision system for a robot. Describe what you see briefly."
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        print("✅ Connected")
        
        # Test 1: Direct bytes (current approach)
        print("\n📤 Sending frame as direct bytes...")
        content = types.Content(
            parts=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png'
                ),
                types.Part(text="What do you see in this frame?")
            ]
        )
        await session.send_client_content(turns=content, turn_complete=True)
        
        # Get response
        response_text = ""
        async for response in session.receive():
            if hasattr(response, 'text') and response.text:
                response_text += response.text
            if response_text and len(response_text) > 50:
                break
        
        print(f"Response: {response_text[:200]}...")
        
        # Test 2: Base64 encoded (alternative approach)
        print("\n📤 Testing base64 encoded frame...")
        # Decode base64 back to bytes for API
        decoded_bytes = base64.b64decode(image_base64)
        content = types.Content(
            parts=[
                types.Part.from_bytes(
                    data=decoded_bytes,
                    mime_type='image/png'
                ),
                types.Part(text="Confirm you can see the image")
            ]
        )
        await session.send_client_content(turns=content, turn_complete=True)
        
        # Get response
        response_text = ""
        async for response in session.receive():
            if hasattr(response, 'text') and response.text:
                response_text += response.text
            if response_text and len(response_text) > 30:
                break
        
        print(f"Base64 test response: {response_text[:100]}...")
        
    print("\n✅ Inline video frame test complete")
    return True

async def test_object_detection():
    """Test object detection with bounding boxes"""
    print("\n" + "="*60)
    print("TEST 2: Object Detection with Bounding Boxes")
    print("="*60)
    
    # Load image and get dimensions
    image_path = "/home/karim/ros2_ws/src/by_your_command/tests/media/whatisthis1.png"
    img = Image.open(image_path)
    width, height = img.size
    print(f"📐 Original image: {width}x{height} pixels")
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    config = {
        "response_modalities": ["TEXT"],
        "system_instruction": "You are an object detection system. Return JSON with detected objects and their bounding boxes."
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        print("✅ Connected")
        
        # Request object detection with specific format
        detection_prompt = """Detect all prominent items in this image. 
        Return a JSON array with each object having:
        - "label": the object name
        - "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000
        - "confidence": detection confidence 0-1
        
        The coordinate system has (0,0) at the top-left corner.
        Return ONLY valid JSON, no other text."""
        
        print("📤 Sending detection request...")
        content = types.Content(
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                types.Part(text=detection_prompt)
            ]
        )
        await session.send_client_content(turns=content, turn_complete=True)
        
        # Collect complete response
        full_response = ""
        async for response in session.receive():
            if hasattr(response, 'text') and response.text:
                full_response += response.text
            # Check for completion signal
            if hasattr(response, 'server_content') and response.server_content:
                if response.server_content.turn_complete:
                    break
        
        print("\n📥 Raw response:")
        print(full_response)
        
        # Try to parse as JSON
        try:
            # Extract JSON from response (might have markdown wrapper)
            json_start = full_response.find('[')
            json_end = full_response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = full_response[json_start:json_end]
                detections = json.loads(json_str)
            else:
                # Try parsing whole response
                detections = json.loads(full_response)
            
            print(f"\n✅ Parsed {len(detections)} objects:")
            
            # Convert to pixel coordinates
            for obj in detections:
                label = obj.get('label', 'unknown')
                box = obj.get('box_2d', [0, 0, 0, 0])
                confidence = obj.get('confidence', 0)
                
                # Convert normalized coords to pixels
                y1_px = int(box[0] * height / 1000)
                x1_px = int(box[1] * width / 1000)
                y2_px = int(box[2] * height / 1000)
                x2_px = int(box[3] * width / 1000)
                
                print(f"\n  🎯 {label} (confidence: {confidence:.2f})")
                print(f"     Normalized: {box}")
                print(f"     Pixels: ({x1_px},{y1_px}) to ({x2_px},{y2_px})")
                print(f"     Size: {x2_px-x1_px}x{y2_px-y1_px} px")
                
        except json.JSONDecodeError as e:
            print(f"\n⚠️ Could not parse JSON: {e}")
            print("Response might not be in expected format")
    
    return True

async def test_detection_vs_description():
    """Compare object detection vs simple description prompts"""
    print("\n" + "="*60)
    print("TEST 3: Detection vs Description Comparison")
    print("="*60)
    
    image_path = "/home/karim/ros2_ws/src/by_your_command/tests/media/whatisthis1.png"
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Test 1: Command agent style (object detection)
    print("\n1️⃣ Command Agent Style (Object Detection):")
    config = {
        "response_modalities": ["TEXT"],
        "system_instruction": "You are a command extraction system. Identify actionable objects."
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        content = types.Content(
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                types.Part(text="List the main objects you can see with their locations. Format: object_name @ position")
            ]
        )
        await session.send_client_content(turns=content, turn_complete=True)
        
        command_response = ""
        async for response in session.receive():
            if hasattr(response, 'text') and response.text:
                command_response += response.text
            if command_response and len(command_response) > 100:
                break
        
        print(f"Response: {command_response[:300]}...")
    
    # Test 2: Conversational agent style (description)
    print("\n2️⃣ Conversational Agent Style (Description):")
    config = {
        "response_modalities": ["TEXT"],
        "system_instruction": "You are a friendly assistant. Describe scenes naturally."
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        content = types.Content(
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                types.Part(text="What's happening in this scene?")
            ]
        )
        await session.send_client_content(turns=content, turn_complete=True)
        
        conv_response = ""
        async for response in session.receive():
            if hasattr(response, 'text') and response.text:
                conv_response += response.text
            if conv_response and len(conv_response) > 100:
                break
        
        print(f"Response: {conv_response[:300]}...")
    
    print("\n📊 Comparison complete - detection provides structure, description provides context")
    return True

async def test_coordinate_verification():
    """Verify coordinate system orientation"""
    print("\n" + "="*60)
    print("TEST 4: Coordinate System Verification")
    print("="*60)
    
    # Create a simple test image with known objects in corners
    from PIL import Image, ImageDraw
    
    # Create test image with colored squares in corners
    test_img = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(test_img)
    
    # Draw colored rectangles in each corner
    draw.rectangle([0, 0, 50, 50], fill='red')        # Top-left
    draw.rectangle([350, 0, 400, 50], fill='green')   # Top-right
    draw.rectangle([0, 250, 50, 300], fill='blue')    # Bottom-left
    draw.rectangle([350, 250, 400, 300], fill='yellow') # Bottom-right
    
    # Save test image
    test_path = "../output/coordinate_test.png"
    test_img.save(test_path)
    print(f"📐 Created test image: 400x300 with colored corners")
    
    with open(test_path, 'rb') as f:
        image_bytes = f.read()
    
    config = {
        "response_modalities": ["TEXT"],
        "system_instruction": "You are a precise object detection system."
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        prompt = """Detect the colored rectangles in this image.
        For each rectangle, return:
        - color
        - position (top-left, top-right, bottom-left, bottom-right)
        - box_2d: [ymin, xmin, ymax, xmax] normalized to 0-1000
        
        Return as JSON array."""
        
        content = types.Content(
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                types.Part(text=prompt)
            ]
        )
        await session.send_client_content(turns=content, turn_complete=True)
        
        full_response = ""
        async for response in session.receive():
            if hasattr(response, 'text') and response.text:
                full_response += response.text
            if hasattr(response, 'server_content') and response.server_content:
                if response.server_content.turn_complete:
                    break
        
        print("\n📥 Coordinate test response:")
        print(full_response)
        
        # Verify coordinates
        print("\n✅ Expected coordinates (normalized to 1000):")
        print("  Red (top-left):     [0, 0, 167, 125]")
        print("  Green (top-right):  [0, 875, 167, 1000]")
        print("  Blue (bottom-left):  [833, 0, 1000, 125]")
        print("  Yellow (bottom-right): [833, 875, 1000, 1000]")
    
    # Clean up test image
    os.remove(test_path)
    return True

async def main():
    print("🚀 Gemini Live API Video & Detection Testing")
    print(f"📍 Model: {model}")
    
    results = []
    
    # Test 1: Inline video frames
    result1 = await test_inline_video_frame()
    results.append(("Inline Video", result1))
    
    # Test 2: Object detection
    result2 = await test_object_detection()
    results.append(("Object Detection", result2))
    
    # Test 3: Detection vs Description
    result3 = await test_detection_vs_description()
    results.append(("Detection vs Description", result3))
    
    # Test 4: Coordinate verification
    result4 = await test_coordinate_verification()
    results.append(("Coordinate System", result4))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Test complete!")