#!/usr/bin/env python3
"""
Object detection test that SAVES results to JSON file
"""

import asyncio
import json
import re
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1beta'}
)

model = "models/gemini-2.0-flash-live-001"

def parse_detection_response(response_text):
    """Robust parser for detection responses"""
    # Try standard JSON first
    try:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Fallback to regex parsing
    detections = []
    object_pattern = r'\{\s*"label"\s*:\s*"([^"]+)"[^}]*"box_2d"\s*:\s*\[([^\]]+)\][^}]*"confidence"\s*:\s*([\d.]+)'
    matches = re.finditer(object_pattern, response_text)
    
    for match in matches:
        label = match.group(1)
        box_str = match.group(2)
        confidence = float(match.group(3))
        
        coords = re.findall(r'\d+', box_str)
        if len(coords) >= 4:
            box = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
            detections.append({
                "label": label,
                "box_2d": box,
                "confidence": confidence
            })
    
    return detections

async def main():
    print("🎯 Object Detection with JSON Output")
    print("="*50)
    
    # Input image
    image_path = "/home/karim/ros2_ws/src/by_your_command/tests/media/whatisthis1.png"
    img = Image.open(image_path)
    width, height = img.size
    print(f"📐 Input image: {width}x{height} pixels")
    print(f"📂 File: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    config = {
        "response_modalities": ["TEXT"],
        "system_instruction": "You are a precise object detection system. Always provide confidence scores between 0.0 and 1.0."
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        print("\n📤 Requesting object detection...")
        
        # Improved prompt for better confidence scores
        prompt = """Detect all objects in this image and return a JSON array.
        For each object provide:
        - "label": the object name
        - "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000
        - "confidence": your confidence score from 0.0 to 1.0
        
        Example format:
        [{"label": "person", "box_2d": [100, 200, 500, 600], "confidence": 0.95}]
        
        Return ONLY the JSON array, no other text."""
        
        content = types.Content(
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                types.Part(text=prompt)
            ]
        )
        await session.send_client_content(turns=content, turn_complete=True)
        
        # Collect complete response
        full_response = ""
        async for response in session.receive():
            if hasattr(response, 'text') and response.text:
                full_response += response.text
            if hasattr(response, 'server_content') and response.server_content:
                if response.server_content.turn_complete:
                    break
        
        print("\n📥 Raw response received")
        
    # Parse detections
    detections = parse_detection_response(full_response)
    
    # Prepare output data
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "source_image": {
            "path": str(image_path),
            "width": width,
            "height": height
        },
        "model": model,
        "raw_response": full_response,
        "detections": []
    }
    
    # Process each detection
    print(f"\n✅ Detected {len(detections)} objects:")
    for i, obj in enumerate(detections, 1):
        label = obj['label']
        box = obj['box_2d']
        conf = obj.get('confidence', 0.0)
        
        # Convert to pixels
        x1_px = int(box[1] * width / 1000)
        y1_px = int(box[0] * height / 1000)
        x2_px = int(box[3] * width / 1000)
        y2_px = int(box[2] * height / 1000)
        
        # Create comprehensive detection entry
        detection_entry = {
            "id": i,
            "label": label,
            "confidence": conf,
            "bbox_normalized": {
                "format": "ymin_xmin_ymax_xmax",
                "scale": "0-1000",
                "coords": box
            },
            "bbox_pixels": {
                "x1": x1_px,
                "y1": y1_px,
                "x2": x2_px,
                "y2": y2_px,
                "width": x2_px - x1_px,
                "height": y2_px - y1_px,
                "center_x": (x1_px + x2_px) // 2,
                "center_y": (y1_px + y2_px) // 2
            }
        }
        
        output_data["detections"].append(detection_entry)
        
        print(f"\n  {i}. {label} (confidence: {conf:.2f})")
        print(f"     Normalized: {box}")
        print(f"     Pixels: ({x1_px},{y1_px}) to ({x2_px},{y2_px})")
        print(f"     Size: {x2_px-x1_px}x{y2_px-y1_px} px")
        print(f"     Center: ({detection_entry['bbox_pixels']['center_x']}, {detection_entry['bbox_pixels']['center_y']})")
    
    # Save to JSON file
    output_file = "../output/object_detection_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"📊 File size: {os.path.getsize(output_file)} bytes")
    
    # Also save a simplified version for easier use
    simple_output = {
        "image": str(image_path),
        "objects": [
            {
                "label": d["label"],
                "confidence": d["confidence"],
                "box": d["bbox_pixels"]
            }
            for d in output_data["detections"]
        ]
    }
    
    simple_file = "../output/detection_simple.json"
    with open(simple_file, 'w') as f:
        json.dump(simple_output, f, indent=2)
    
    print(f"💾 Simplified results saved to: {simple_file}")
    
    # Print summary
    print(f"\n📋 Summary:")
    print(f"  - Detected {len(detections)} objects")
    print(f"  - Full results: {output_file}")
    print(f"  - Simple format: {simple_file}")
    print(f"  - Timestamp: {output_data['timestamp']}")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Done!")