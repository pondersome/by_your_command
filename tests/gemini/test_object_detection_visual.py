#!/usr/bin/env python3
"""
Visual Object Detection Test
Detects objects and overlays bounding boxes on the original image
"""

import asyncio
import json
import re
import os
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
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
    detections = []
    
    # Try standard JSON first
    try:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return parsed
    except:
        pass
    
    # Fallback to regex parsing for malformed JSON
    # More flexible pattern to catch various formats
    patterns = [
        # Standard format
        r'\{\s*"label"\s*:\s*"([^"]+)"[^}]*"box_2d"\s*:\s*\[([^\]]+)\][^}]*"confidence"\s*:\s*([\d.]+)[^}]*\}',
        # Format with missing confidence
        r'\{\s*"label"\s*:\s*"([^"]+)"[^}]*"box_2d"\s*:\s*\[([^\]]+)\][^}]*\}',
        # Format with extra fields
        r'"label"\s*:\s*"([^"]+)"[^,]*,[^}]*"box_2d"\s*:\s*\[([^\]]+)\]',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                label = match.group(1) if match.lastindex >= 1 else "unknown"
                box_str = match.group(2) if match.lastindex >= 2 else "0,0,0,0"
                confidence = float(match.group(3)) if match.lastindex >= 3 else 0.0
                
                # Parse box coordinates
                coords = re.findall(r'-?\d+', box_str)
                if len(coords) >= 4:
                    box = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                    detections.append({
                        "label": label,
                        "box_2d": box,
                        "confidence": confidence
                    })
            except:
                continue
    
    return detections

def generate_colors(n):
    """Generate n distinct colors for bounding boxes"""
    colors = []
    for i in range(n):
        hue = i * 360 / n
        # Convert HSV to RGB (simplified)
        r = int(255 * abs((hue / 60) % 2 - 1))
        g = int(255 * (1 - abs((hue / 120) % 2 - 1)))
        b = int(255 * (1 - abs(((hue - 240) / 120) % 2 - 1)))
        colors.append((r, g, b))
    
    # Shuffle for variety
    random.shuffle(colors)
    return colors

def draw_detection_overlay(image_path, detections, output_path):
    """Draw bounding boxes and labels on the image"""
    # Load image
    img = Image.open(image_path)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Generate colors
    colors = generate_colors(len(detections))
    
    # Statistics
    stats = {
        "high_conf": 0,  # > 0.8
        "medium_conf": 0,  # 0.5 - 0.8
        "low_conf": 0,  # < 0.5
        "no_conf": 0  # 0.0
    }
    
    print(f"\n📊 Drawing {len(detections)} detections on image...")
    
    for i, det in enumerate(detections):
        label = det.get('label', 'unknown')
        box = det.get('box_2d', [0, 0, 0, 0])
        conf = det.get('confidence', 0.0)
        
        # Convert normalized coords to pixels
        y1 = int(box[0] * height / 1000)
        x1 = int(box[1] * width / 1000)
        y2 = int(box[2] * height / 1000)
        x2 = int(box[3] * width / 1000)
        
        # Ensure coordinates are valid
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)
        
        # Get color based on confidence
        if conf > 0.8:
            color = colors[i % len(colors)]
            line_width = 3
            stats["high_conf"] += 1
        elif conf > 0.5:
            color = tuple(int(c * 0.8) for c in colors[i % len(colors)])  # Slightly darker
            line_width = 2
            stats["medium_conf"] += 1
        elif conf > 0:
            color = tuple(int(c * 0.6) for c in colors[i % len(colors)])  # Even darker
            line_width = 2
            stats["low_conf"] += 1
        else:
            color = (128, 128, 128)  # Gray for no confidence
            line_width = 1
            stats["no_conf"] += 1
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Prepare label text
        if conf > 0:
            label_text = f"{label} ({conf:.2f})"
        else:
            label_text = f"{label} (?)"
        
        # Calculate text size
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        label_y = y1 - text_height - 4 if y1 > text_height + 4 else y2 + 2
        draw.rectangle(
            [x1, label_y, x1 + text_width + 4, label_y + text_height + 2],
            fill=color,
            outline=color
        )
        
        # Draw label text
        text_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
        draw.text((x1 + 2, label_y), label_text, fill=text_color, font=font)
        
        # Print detection info
        print(f"  {i+1}. {label}:")
        print(f"     Confidence: {conf:.3f}")
        print(f"     Box: ({x1},{y1}) to ({x2},{y2})")
        print(f"     Size: {x2-x1}x{y2-y1} px")
    
    # Add summary statistics to image
    stats_text = f"Detections: {len(detections)} | High: {stats['high_conf']} | Med: {stats['medium_conf']} | Low: {stats['low_conf']} | None: {stats['no_conf']}"
    
    # Draw stats bar at bottom
    stats_height = 25
    overlay = Image.new('RGBA', (width, stats_height), (0, 0, 0, 180))
    stats_draw = ImageDraw.Draw(overlay)
    stats_draw.text((5, 5), stats_text, fill=(255, 255, 255), font=small_font)
    
    # Composite stats overlay
    img_with_stats = img.copy()
    img_with_stats.paste(overlay, (0, height - stats_height), overlay)
    
    # Save image with overlay
    img_with_stats.save(output_path)
    print(f"\n💾 Saved annotated image to: {output_path}")
    
    return stats

async def main():
    print("🎯 Visual Object Detection Test")
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
        "system_instruction": "You are a comprehensive object detection system. Detect ALL objects, including background elements."
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        print("\n📤 Requesting comprehensive object detection...")
        
        # Request detection with emphasis on finding everything
        prompt = """Detect ALL objects and regions in this image, including:
        - Main subjects (people, animals)
        - Objects they're interacting with
        - Background elements (sky, sand, water, etc.)
        - Any other notable features
        
        For each detection provide:
        - "label": descriptive name
        - "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000
        - "confidence": your confidence from 0.0 to 1.0
        
        Include even low-confidence detections.
        Return as JSON array only."""
        
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
        
        print("\n📥 Response received")
    
    # Parse detections
    detections = parse_detection_response(full_response)
    
    if not detections:
        print("❌ No objects detected")
        # Save raw response for debugging
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        debug_file = str(output_dir / "detection_debug.txt")
        with open(debug_file, 'w') as f:
            f.write(full_response)
        print(f"📝 Raw response saved to: {debug_file}")
        return
    
    # Sort by confidence for better visualization
    detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    print(f"\n✅ Detected {len(detections)} objects")
    
    # Create annotated image
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_image = str(output_dir / "detection_annotated.png")
    stats = draw_detection_overlay(image_path, detections, output_image)
    
    # Save detection data
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "source_image": str(image_path),
        "image_size": {"width": width, "height": height},
        "total_detections": len(detections),
        "statistics": stats,
        "detections": detections
    }
    
    output_json = str(output_dir / "detection_visual_results.json")
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Detection data saved to: {output_json}")
    
    # Print summary
    print(f"\n📋 Detection Summary:")
    print(f"  Total objects: {len(detections)}")
    print(f"  High confidence (>0.8): {stats['high_conf']}")
    print(f"  Medium confidence (0.5-0.8): {stats['medium_conf']}")
    print(f"  Low confidence (<0.5): {stats['low_conf']}")
    print(f"  No confidence: {stats['no_conf']}")
    
    # List all detected labels
    print(f"\n🏷️ Detected labels:")
    for det in detections:
        print(f"  - {det['label']} ({det['confidence']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Visual detection test complete!")