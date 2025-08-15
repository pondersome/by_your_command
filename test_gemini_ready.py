#!/usr/bin/env python3
"""Test if Gemini agent is ready to run"""

import os
import sys
import yaml

def test_gemini_setup():
    """Check if all components are ready"""
    print("🔍 Checking Gemini Live Agent Setup...")
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✅ GEMINI_API_KEY is set ({len(api_key)} chars)")
    else:
        print("❌ GEMINI_API_KEY not set")
        return False
        
    # Check config files exist
    configs = [
        'config/gemini_conversation_agent.yaml',
        'config/gemini_command_agent.yaml',
        'config/prompts.yaml'
    ]
    
    for config in configs:
        path = f'/home/karim/ros2_ws/src/by_your_command/{config}'
        if os.path.exists(path):
            print(f"✅ {config} exists")
        else:
            print(f"❌ {config} missing")
            return False
            
    # Check prompts exist
    with open('/home/karim/ros2_ws/src/by_your_command/config/prompts.yaml', 'r') as f:
        prompts_data = yaml.safe_load(f)
        
    required_prompts = [
        'barney_conversational_gemini',
        'barney_command_extractor_gemini'
    ]
    
    for prompt_id in required_prompts:
        if 'prompts' in prompts_data and prompt_id in prompts_data['prompts']:
            print(f"✅ Prompt '{prompt_id}' found")
        else:
            print(f"❌ Prompt '{prompt_id}' missing")
            return False
            
    # Check Python packages
    try:
        import pipecat
        print(f"✅ Pipecat {pipecat.__version__} installed")
    except ImportError:
        print("❌ Pipecat not installed")
        return False
        
    try:
        from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
        print("✅ GeminiMultimodalLiveLLMService available")
    except ImportError as e:
        print(f"❌ GeminiMultimodalLiveLLMService import failed: {e}")
        return False
        
    print("\n🎉 All checks passed! Ready to launch:")
    print("   ros2 launch by_your_command gemini_single.launch.py")
    print("   ros2 launch by_your_command gemini_dual_agent.launch.py")
    return True

if __name__ == '__main__':
    success = test_gemini_setup()
    sys.exit(0 if success else 1)