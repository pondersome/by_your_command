#!/usr/bin/env python3
"""
Test Named Prompt System Integration

Tests the named prompt system with various contexts and configurations
to ensure proper prompt selection and integration with SessionManager.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import sys
import os

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from prompt_loader import PromptLoader


def test_prompt_selection():
    """Test basic prompt selection functionality"""
    print("=== Testing Basic Prompt Selection ===")
    
    loader = PromptLoader()
    
    print(f"Available prompts: {loader.list_prompts()}")
    print(f"Selection rules: {loader.selection_rules}")
    print()
    
    # Test 1: Default selection (no context)
    print("Test 1: Default selection (no context)")
    default_prompt = loader.select_prompt()
    print(f"Selected prompt preview: {default_prompt[:150]}...")
    print()
    
    # Test 2: Adult user context (should get Barney prompt)
    print("Test 2: Adult user context")
    adult_context = {
        'user_age': 25,
        'environment': 'normal',
        'robot_name': 'Barney'
    }
    adult_prompt = loader.select_prompt(adult_context)
    print(f"Context: {adult_context}")
    print(f"Selected prompt preview: {adult_prompt[:150]}...")
    print()
    
    # Test 3: Child user context (should get friendly assistant)
    print("Test 3: Child user context")
    child_context = {
        'user_age': 8,
        'environment': 'normal'
    }
    child_prompt = loader.select_prompt(child_context)
    print(f"Context: {child_context}")
    print(f"Selected prompt preview: {child_prompt[:150]}...")
    print()
    
    # Test 4: Crowded environment context
    print("Test 4: Crowded environment context")
    crowded_context = {
        'user_age': 25,
        'environment': 'crowded'
    }
    crowded_prompt = loader.select_prompt(crowded_context)
    print(f"Context: {crowded_context}")
    print(f"Selected prompt preview: {crowded_prompt[:150]}...")
    print()


def test_user_prefixes():
    """Test user prompt prefix functionality"""
    print("=== Testing User Prompt Prefixes ===")
    
    loader = PromptLoader()
    
    # Test context reminder prefix
    context = {
        'last_topic': 'robot movements',
        'last_command': 'look left and move forward'
    }
    
    prefix = loader.get_user_prefix('context_reminder', context)
    print("Context reminder prefix:")
    print(prefix)
    print()
    
    # Test full prompt building
    base_prompt = loader.select_prompt({'user_age': 25})
    full_prompt = loader.build_full_prompt(
        base_prompt=base_prompt,
        context=context,
        user_prefix_id='context_reminder'
    )
    
    print("Full prompt with prefix (first 300 chars):")
    print(full_prompt[:300] + "...")
    print()


def test_ab_testing():
    """Test A/B testing functionality"""
    print("=== Testing A/B Testing (Disabled by Default) ===")
    
    loader = PromptLoader()
    
    # Check if A/B test is configured
    ab_config = loader.selection_rules.get('ab_tests', {}).get('visual_description_test', {})
    print(f"A/B test configuration: {ab_config}")
    
    if ab_config.get('enabled', False):
        print("A/B testing is enabled - testing variant selection...")
        # Run multiple selections to see distribution
        selections = {}
        for i in range(10):
            prompt = loader.select_prompt({'user_age': 25})
            # Identify which prompt this is
            for prompt_id, prompt_info in loader.prompts.items():
                if prompt_info.system_prompt == prompt:
                    selections[prompt_id] = selections.get(prompt_id, 0) + 1
                    break
        print(f"Selection distribution over 10 trials: {selections}")
    else:
        print("A/B testing is disabled (as expected)")
    print()


def test_prompt_info():
    """Test prompt metadata retrieval"""
    print("=== Testing Prompt Metadata ===")
    
    loader = PromptLoader()
    
    for prompt_id in loader.list_prompts():
        info = loader.get_prompt_info(prompt_id)
        print(f"Prompt: {prompt_id}")
        print(f"  Name: {info.name}")
        print(f"  Description: {info.description}")
        print(f"  Version: {info.version}")
        print(f"  Tested with: {info.tested_with}")
        print(f"  Prompt length: {len(info.system_prompt)} characters")
        print()
    
    metadata = loader.get_metadata()
    print(f"Prompt file metadata: {metadata}")
    print()


def test_prompt_selection_scenarios():
    """Test specific scenarios for Barney robot"""
    print("=== Testing Barney Robot Scenarios ===")
    
    loader = PromptLoader()
    
    scenarios = [
        {
            'name': 'Normal adult interaction',
            'context': {'user_age': 30, 'environment': 'normal', 'robot_name': 'Barney'}
        },
        {
            'name': 'Child interaction',
            'context': {'user_age': 7, 'environment': 'normal', 'robot_name': 'Barney'}
        },
        {
            'name': 'Crowded environment',
            'context': {'user_age': 25, 'environment': 'crowded', 'robot_name': 'Barney'}
        },
        {
            'name': 'Elderly user',
            'context': {'user_age': 75, 'environment': 'quiet', 'robot_name': 'Barney'}
        }
    ]
    
    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"Context: {scenario['context']}")
        
        prompt = loader.select_prompt(scenario['context'])
        
        # Identify which prompt was selected
        selected_prompt_id = 'unknown'
        for prompt_id, prompt_info in loader.prompts.items():
            if prompt_info.system_prompt == prompt:
                selected_prompt_id = prompt_id
                break
                
        print(f"Selected prompt: {selected_prompt_id}")
        
        # Show key characteristics of the prompt
        if 'barney' in selected_prompt_id.lower():
            print("  → Barney command/visual mode with preset arm positions")
        elif 'friendly' in selected_prompt_id.lower():
            print("  → Simple conversational assistant mode")
        else:
            print("  → Other prompt type")
            
        print(f"  Prompt preview: {prompt[:100]}...")
        print()


if __name__ == "__main__":
    print("🤖 Testing Named Prompt System for OpenAI Realtime Agent")
    print("=" * 60)
    print()
    
    try:
        test_prompt_selection()
        test_user_prefixes()
        test_ab_testing()
        test_prompt_info()
        test_prompt_selection_scenarios()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()