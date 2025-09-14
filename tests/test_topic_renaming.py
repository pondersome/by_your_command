#!/usr/bin/env python3
"""
Test script to verify topic renaming changes
"""

import yaml
import sys
from pathlib import Path

def test_config_files():
    """Check that all config files use new topic names"""
    config_dir = Path(__file__).parent.parent / "config"
    
    old_topics = ["voice_chunks", "text_input", "audio_out", "llm_transcript", "command_transcript"]
    new_topics = ["prompt_voice", "prompt_text", "response_voice", "response_text", "response_cmd", "prompt_transcript"]
    
    errors = []
    
    # Check config files
    for config_file in config_dir.glob("*.yaml"):
        with open(config_file, 'r') as f:
            content = f.read()
            
        # Check for old topic names (except in comments)
        for old_topic in old_topics:
            # Skip if it's in a comment
            for line in content.split('\n'):
                if old_topic in line and not line.strip().startswith('#'):
                    # Check if it's a field name (ends with _topic:)
                    if f"{old_topic}_topic:" in line:
                        continue  # This is OK, it's a field name
                    # Check if it's in quotes (actual topic name)
                    if f'"{old_topic}"' in line or f"'{old_topic}'" in line:
                        errors.append(f"{config_file.name}: Found old topic '{old_topic}' in line: {line.strip()}")
    
    # Check for new topics presence
    for config_file in config_dir.glob("*.yaml"):
        if "bridge" not in config_file.name and "gemini" not in config_file.name and "oai" not in config_file.name:
            continue
            
        with open(config_file, 'r') as f:
            content = f.read()
            
        has_new_topics = any(new_topic in content for new_topic in new_topics)
        if not has_new_topics and ("oai" in config_file.name or "gemini" in config_file.name):
            print(f"Warning: {config_file.name} doesn't seem to have any new topic names")
    
    return errors

def test_agent_code():
    """Check that agent code uses new topic names"""
    agents_dir = Path(__file__).parent.parent / "agents"
    
    errors = []
    
    # Check OpenAI agent
    oai_agent = agents_dir / "oai_realtime" / "oai_realtime_agent.py"
    if oai_agent.exists():
        with open(oai_agent, 'r') as f:
            content = f.read()
        
        # Check for new published_topics structure
        if "'response_voice'" not in content:
            errors.append("oai_realtime_agent.py: Missing 'response_voice' in published_topics")
        if "'response_text'" not in content:
            errors.append("oai_realtime_agent.py: Missing 'response_text' in published_topics")
        if "'prompt_transcript'" not in content:
            errors.append("oai_realtime_agent.py: Missing 'prompt_transcript' in published_topics")
    
    # Check Gemini agent
    gemini_agent = agents_dir / "gemini_live" / "gemini_live_agent.py"
    if gemini_agent.exists():
        with open(gemini_agent, 'r') as f:
            content = f.read()
        
        if "'response_voice'" not in content:
            errors.append("gemini_live_agent.py: Missing 'response_voice' in published_topics")
        if "'response_cmd'" not in content:
            errors.append("gemini_live_agent.py: Missing 'response_cmd' in published_topics")
    
    # Check Gemini receive coordinator
    gemini_coordinator = agents_dir / "gemini_live" / "receive_coordinator.py"
    if gemini_coordinator.exists():
        with open(gemini_coordinator, 'r') as f:
            content = f.read()
        
        if "'response_cmd'" not in content:
            errors.append("receive_coordinator.py: Missing 'response_cmd' topic")
        if "'prompt_transcript'" not in content:
            errors.append("receive_coordinator.py: Missing 'prompt_transcript' topic")
    
    return errors

def main():
    print("Testing topic renaming changes...")
    print("-" * 50)
    
    # Test config files
    print("\nChecking configuration files...")
    config_errors = test_config_files()
    if config_errors:
        print("❌ Config file errors:")
        for error in config_errors:
            print(f"  - {error}")
    else:
        print("✅ All config files updated correctly")
    
    # Test agent code
    print("\nChecking agent code...")
    code_errors = test_agent_code()
    if code_errors:
        print("❌ Agent code errors:")
        for error in code_errors:
            print(f"  - {error}")
    else:
        print("✅ All agent code updated correctly")
    
    # Summary
    print("\n" + "=" * 50)
    total_errors = len(config_errors) + len(code_errors)
    if total_errors == 0:
        print("✅ SUCCESS: All topic renaming changes verified!")
        return 0
    else:
        print(f"❌ FAILED: Found {total_errors} issues that need fixing")
        return 1

if __name__ == "__main__":
    sys.exit(main())