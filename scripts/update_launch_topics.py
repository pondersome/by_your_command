#!/usr/bin/env python3
"""
Script to update all launch files with new topic names
"""

import os
from pathlib import Path

def update_launch_file(filepath):
    """Update topic names in a launch file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace topic names in parameters
    replacements = [
        ("'topic': 'audio_out'", "'topic': 'response_voice'"),
        ("'input_topic': 'audio_out'", "'input_topic': 'response_voice'"),
        ("'command_transcript_topic': 'command_transcript'", "'command_transcript_topic': 'response_cmd'"),
        
        # Replace in string messages
        ("/command_transcript", "/response_cmd"),
        ("command_transcript", "response_cmd"),  # Without slash
        
        # Update directory names (keep for clarity even though they're just directory names)
        ("'/tmp/voice_chunks/", "'/tmp/prompt_voice/"),  # Update directory names for consistency
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Only write if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    launch_dir = Path(__file__).parent.parent / "bringup"
    
    print("Updating launch files with new topic names...")
    print("-" * 50)
    
    updated_files = []
    for launch_file in launch_dir.glob("*.launch.py"):
        if update_launch_file(launch_file):
            updated_files.append(launch_file.name)
            print(f"✅ Updated {launch_file.name}")
        else:
            print(f"⏭️  No changes needed in {launch_file.name}")
    
    print("\n" + "=" * 50)
    if updated_files:
        print(f"✅ Updated {len(updated_files)} launch files:")
        for f in updated_files:
            print(f"  - {f}")
    else:
        print("No launch files needed updating")

if __name__ == "__main__":
    main()