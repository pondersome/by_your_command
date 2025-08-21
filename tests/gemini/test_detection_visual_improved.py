#!/usr/bin/env python3
"""
Improved Visual Object Detection Test
- Better drawing order (by area, larger first)
- Option to test different models
- Transparency for overlapping boxes
"""

import asyncio
import json
import re
import os
import sys
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

# Model selection - can be overridden by command line
DEFAULT_MODEL = "models/gemini-2.0-flash-live-001"
PREVIEW_MODEL = "gemini-live-2.5-flash-preview"  # Might have better detection

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
    
    # Fallback parsing (same as before)
    patterns = [
        r'\{\s*"label"\s*:\s*"([^"]+)"[^}]*"box_2d"\s*:\s*\[([^\]]+)\][^}]*"confidence"\s*:\s*([\d.]+)[^}]*\}',
        r'\{\s*"label"\s*:\s*"([^"]+)"[^}]*"box_2d"\s*:\s*\[([^\]]+)\][^}]*\}',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                label = match.group(1) if match.lastindex >= 1 else "unknown"
                box_str = match.group(2) if match.lastindex >= 2 else "0,0,0,0"
                confidence = float(match.group(3)) if match.lastindex >= 3 else 0.0
                
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

def calculate_box_area(box, width, height):
    """Calculate pixel area of a bounding box"""
    y1 = int(box[0] * height / 1000)
    x1 = int(box[1] * width / 1000)
    y2 = int(box[2] * height / 1000)
    x2 = int(box[3] * width / 1000)
    return abs((x2 - x1) * (y2 - y1))

def generate_colors(n):
    """Generate distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        hue = (i * 137.5) % 360  # Golden angle for better distribution
        # High saturation and value for vivid colors
        # Convert HSV to RGB
        c = hue / 60
        x = 255 * (1 - abs(c % 2 - 1))
        if c < 1:
            r, g, b = 255, int(x), 0
        elif c < 2:
            r, g, b = int(x), 255, 0
        elif c < 3:
            r, g, b = 0, 255, int(x)
        elif c < 4:
            r, g, b = 0, int(x), 255
        elif c < 5:
            r, g, b = int(x), 0, 255
        else:
            r, g, b = 255, 0, int(x)
        colors.append((r, g, b))
    return colors

def draw_detection_overlay(image_path, detections, output_path, model_name):
    """Draw bounding boxes with improved visibility"""
    img = Image.open(image_path)
    width, height = img.size
    
    # Create overlay for transparency
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # Main image for solid elements
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Sort detections by area (largest first) so smaller objects appear on top
    detections_with_area = []
    for det in detections:
        area = calculate_box_area(det['box_2d'], width, height)
        detections_with_area.append((det, area))
    
    detections_with_area.sort(key=lambda x: x[1], reverse=True)
    
    # Generate colors
    colors = generate_colors(len(detections))
    
    # Statistics
    stats = {
        "excellent": 0,  # > 0.95
        "good": 0,      # 0.90 - 0.95
        "fair": 0,      # 0.80 - 0.90
        "poor": 0,      # < 0.80
    }
    
    print(f"\n📊 Drawing {len(detections)} detections (ordered by area, largest first)...")
    
    for i, (det, area) in enumerate(detections_with_area):
        label = det.get('label', 'unknown')
        box = det.get('box_2d', [0, 0, 0, 0])
        conf = det.get('confidence', 0.0)
        
        # Convert to pixels
        y1 = int(box[0] * height / 1000)
        x1 = int(box[1] * width / 1000)
        y2 = int(box[2] * height / 1000)
        x2 = int(box[3] * width / 1000)
        
        # Ensure valid coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)
        
        # Color and style based on confidence
        base_color = colors[i % len(colors)]
        
        if conf > 0.95:
            color = base_color
            line_width = 3
            alpha = 255
            stats["excellent"] += 1
            quality = "★★★"
        elif conf > 0.90:
            color = base_color
            line_width = 2
            alpha = 220
            stats["good"] += 1
            quality = "★★"
        elif conf > 0.80:
            color = tuple(int(c * 0.8) for c in base_color)
            line_width = 2
            alpha = 180
            stats["fair"] += 1
            quality = "★"
        else:
            color = tuple(int(c * 0.6) for c in base_color)
            line_width = 1
            alpha = 140
            stats["poor"] += 1
            quality = "○"
        
        # Draw semi-transparent fill for large boxes (likely background)
        if area > (width * height * 0.3):  # More than 30% of image
            fill_color = (*color, 30)  # Very transparent
            draw_overlay.rectangle([x1, y1, x2, y2], fill=fill_color)
        
        # Draw box outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Add corner markers for better visibility
        corner_length = min(20, (x2-x1)//4, (y2-y1)//4)
        if corner_length > 5:
            # Top-left corner
            draw.line([(x1, y1), (x1 + corner_length, y1)], fill=color, width=line_width+1)
            draw.line([(x1, y1), (x1, y1 + corner_length)], fill=color, width=line_width+1)
            # Top-right corner
            draw.line([(x2 - corner_length, y1), (x2, y1)], fill=color, width=line_width+1)
            draw.line([(x2, y1), (x2, y1 + corner_length)], fill=color, width=line_width+1)
            # Bottom-left corner
            draw.line([(x1, y2 - corner_length), (x1, y2)], fill=color, width=line_width+1)
            draw.line([(x1, y2), (x1 + corner_length, y2)], fill=color, width=line_width+1)
            # Bottom-right corner
            draw.line([(x2 - corner_length, y2), (x2, y2)], fill=color, width=line_width+1)
            draw.line([(x2, y2 - corner_length), (x2, y2)], fill=color, width=line_width+1)
        
        # Label with quality indicator
        label_text = f"{quality} {label} ({conf:.2f})"
        
        # Calculate text size
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position label (prefer above box, but adjust if needed)
        if y1 > text_height + 4:
            label_y = y1 - text_height - 4
        else:
            label_y = y2 + 2
        
        # Ensure label is within image bounds
        if label_y + text_height > height:
            label_y = y1 + 2  # Place inside box at top
        
        # Draw label background with transparency
        label_bg = Image.new('RGBA', (text_width + 6, text_height + 4), (*color, alpha))
        img.paste(label_bg, (x1, label_y), label_bg)
        
        # Draw text
        text_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
        draw.text((x1 + 3, label_y + 2), label_text, fill=text_color, font=font)
        
        # Print info
        print(f"  {i+1}. {label} ({area:,} px²):")
        print(f"     Confidence: {conf:.3f} {quality}")
        print(f"     Box: ({x1},{y1}) to ({x2},{y2})")
    
    # Composite overlay onto image
    img = Image.alpha_composite(img.convert('RGBA'), overlay)
    
    # Add info bar
    info_text = f"Model: {model_name.split('/')[-1]} | Total: {len(detections)} | "
    info_text += f"Excellent(>95%): {stats['excellent']} | Good(90-95%): {stats['good']} | "
    info_text += f"Fair(80-90%): {stats['fair']} | Poor(<80%): {stats['poor']}"
    
    # Create info bar
    info_height = 25
    info_bar = Image.new('RGBA', (width, info_height), (0, 0, 0, 200))
    info_draw = ImageDraw.Draw(info_bar)
    info_draw.text((5, 5), info_text, fill=(255, 255, 255), font=small_font)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_bbox = info_draw.textbbox((0, 0), timestamp, font=small_font)
    ts_width = ts_bbox[2] - ts_bbox[0]
    info_draw.text((width - ts_width - 5, 5), timestamp, fill=(200, 200, 200), font=small_font)
    
    # Composite info bar
    final_img = Image.new('RGBA', (width, height + info_height))
    final_img.paste(img, (0, 0))
    final_img.paste(info_bar, (0, height), info_bar)
    
    # Save
    final_img.save(output_path, 'PNG')
    print(f"\n💾 Saved annotated image to: {output_path}")
    
    return stats

async def test_model(model_name, image_path, image_bytes):
    """Test a specific model"""
    print(f"\n🔬 Testing model: {model_name}")
    print("-" * 50)
    
    config = {
        "response_modalities": ["TEXT"],
        "system_instruction": "You are a precise object detection system. Detect all objects and regions."
    }
    
    try:
        async with client.aio.live.connect(model=model_name, config=config) as session:
            prompt = """Detect ALL objects and regions in this image.
            For each provide:
            - "label": accurate descriptive name
            - "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000
            - "confidence": your true confidence from 0.0 to 1.0
            
            Be accurate with your confidence scores.
            Return as JSON array only."""
            
            content = types.Content(
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                    types.Part(text=prompt)
                ]
            )
            await session.send_client_content(turns=content, turn_complete=True)
            
            # Collect response
            full_response = ""
            async for response in session.receive():
                if hasattr(response, 'text') and response.text:
                    full_response += response.text
                if hasattr(response, 'server_content') and response.server_content:
                    if response.server_content.turn_complete:
                        break
            
            return parse_detection_response(full_response), full_response
            
    except Exception as e:
        print(f"❌ Error with model {model_name}: {e}")
        return [], str(e)

async def main():
    print("🎯 Improved Visual Object Detection Test")
    print("="*50)
    
    # Check for model argument
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        models_to_test = [PREVIEW_MODEL, DEFAULT_MODEL]
        print("Testing both preview and standard models")
    else:
        models_to_test = [DEFAULT_MODEL]
        print("Testing standard model (use --preview to test both)")
    
    # Input image
    image_path = "/home/karim/ros2_ws/src/by_your_command/tests/media/whatisthis1.png"
    img = Image.open(image_path)
    width, height = img.size
    print(f"📐 Input image: {width}x{height} pixels")
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Test each model
    for model_name in models_to_test:
        detections, raw_response = await test_model(model_name, image_path, image_bytes)
        
        if not detections:
            print(f"❌ No detections from {model_name}")
            continue
        
        print(f"✅ Got {len(detections)} detections")
        
        # Create output filename based on model
        model_suffix = model_name.replace('/', '_').replace('-', '_')
        output_image = str(output_dir / f"detection_{model_suffix}.png")
        
        # Draw annotations
        stats = draw_detection_overlay(image_path, detections, output_image, model_name)
        
        # Save JSON data
        output_data = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "image": str(image_path),
            "statistics": stats,
            "detections": detections
        }
        
        output_json = str(output_dir / f"detection_{model_suffix}.json")
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 JSON saved to: {output_json}")
        
        # Print summary
        print(f"\n📈 {model_name} Summary:")
        print(f"  Excellent (>95%): {stats['excellent']}")
        print(f"  Good (90-95%): {stats['good']}")
        print(f"  Fair (80-90%): {stats['fair']}")
        print(f"  Poor (<80%): {stats['poor']}")

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Test complete!")