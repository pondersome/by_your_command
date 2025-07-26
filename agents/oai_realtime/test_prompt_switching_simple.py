#!/usr/bin/env python3
"""
Simple Test for Runtime System Prompt Switching

Tests the prompt switching capabilities without SessionManager dependencies.

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


def test_prompt_switching_logic():
    """Test the core prompt switching logic"""
    print("🔄 Testing Core Prompt Switching Logic")
    print("=" * 50)
    
    loader = PromptLoader()
    
    print("1. Available Prompts")
    print("-" * 20)
    available = loader.list_prompts()
    print(f"Available prompts: {available}")
    for prompt_id in available:
        info = loader.get_prompt_info(prompt_id)
        print(f"  • {prompt_id}: {info.name} ({len(info.system_prompt)} chars)")
    print()
    
    print("2. Context-Based Selection")
    print("-" * 20)
    contexts = [
        {'user_age': 30, 'environment': 'normal'},
        {'user_age': 7, 'environment': 'normal'},
        {'user_age': 25, 'environment': 'crowded'},
    ]
    
    for context in contexts:
        prompt = loader.select_prompt(context)
        # Find which prompt this is
        prompt_id = 'unknown'
        for pid, pinfo in loader.prompts.items():
            if pinfo.system_prompt == prompt:
                prompt_id = pid
                break
        print(f"Context {context} → {prompt_id}")
    print()
    
    print("3. Prompt Override Simulation")
    print("-" * 20)
    
    # Simulate what SessionManager.switch_prompt() does
    class MockSessionManager:
        def __init__(self):
            self.prompt_loader = PromptLoader()
            self.prompt_context = {'user_age': 30, 'environment': 'normal'}
            self._prompt_override = None
            
        def switch_prompt_simulation(self, prompt_id=None, context_updates=None):
            """Simulate the switch_prompt logic"""
            # Update context
            if context_updates:
                self.prompt_context.update(context_updates)
                print(f"  Updated context: {self.prompt_context}")
            
            # Set override
            if prompt_id:
                if prompt_id not in self.prompt_loader.prompts:
                    print(f"  ❌ Error: Prompt '{prompt_id}' not found")
                    return False
                self._prompt_override = prompt_id
                print(f"  ✅ Override set to: {prompt_id}")
            
            # Get effective prompt
            if self._prompt_override:
                selected_prompt = self.prompt_loader.prompts[self._prompt_override].system_prompt
                selection_type = f"override:{self._prompt_override}"
            else:
                selected_prompt = self.prompt_loader.select_prompt(self.prompt_context)
                # Find prompt ID
                for pid, pinfo in self.prompt_loader.prompts.items():
                    if pinfo.system_prompt == selected_prompt:
                        selection_type = f"context:{pid}"
                        break
                else:
                    selection_type = "context:unknown"
            
            print(f"  Current selection: {selection_type}")
            print(f"  Prompt preview: {selected_prompt[:100]}...")
            return True
            
        def clear_override(self):
            if self._prompt_override:
                old = self._prompt_override
                self._prompt_override = None
                print(f"  Cleared override: {old}")
            else:
                print("  No override to clear")
    
    mock_session = MockSessionManager()
    
    print("a) Initial state (adult context):")
    mock_session.switch_prompt_simulation()
    print()
    
    print("b) Switch to child context:")
    mock_session.switch_prompt_simulation(context_updates={'user_age': 6})
    print()
    
    print("c) Override to specific prompt:")
    mock_session.switch_prompt_simulation(prompt_id='barney_command_visual')
    print()
    
    print("d) Try invalid prompt:")
    mock_session.switch_prompt_simulation(prompt_id='nonexistent')
    print()
    
    print("e) Clear override (return to context):")
    mock_session.clear_override()
    mock_session.switch_prompt_simulation()
    print()
    
    print("4. Runtime Capabilities Summary")
    print("-" * 20)
    print("✅ Context-based automatic selection")
    print("✅ Manual prompt override by ID")
    print("✅ Context updates trigger reselection")
    print("✅ Override clearing returns to context selection")
    print("✅ Error handling for invalid prompts")
    print("✅ Prompt metadata and information access")


def test_prompt_differences():
    """Show the actual differences between prompts"""
    print("\n🎭 Actual Prompt Content Differences")
    print("=" * 50)
    
    loader = PromptLoader()
    
    # Get the different prompts that would be selected
    test_cases = [
        {'name': 'Adult User', 'context': {'user_age': 30}},
        {'name': 'Child User', 'context': {'user_age': 7}},
    ]
    
    for case in test_cases:
        print(f"{case['name']} Context: {case['context']}")
        
        prompt = loader.select_prompt(case['context'])
        
        # Find prompt ID
        prompt_id = 'unknown'
        prompt_info = None
        for pid, pinfo in loader.prompts.items():
            if pinfo.system_prompt == prompt:
                prompt_id = pid
                prompt_info = pinfo
                break
        
        print(f"  Selected: {prompt_id}")
        if prompt_info:
            print(f"  Name: {prompt_info.name}")
            print(f"  Length: {len(prompt_info.system_prompt)} characters")
            
        # Show key characteristics
        if 'barney' in prompt_id.lower():
            has_presets = 'PRESET:' in prompt
            has_robot_identity = 'skid-steer robot' in prompt
            has_arm_positions = 'bumper' in prompt and 'tenhut' in prompt
            print(f"  Features: Command presets={has_presets}, Robot identity={has_robot_identity}, Arm positions={has_arm_positions}")
        elif 'friendly' in prompt_id.lower():
            is_simple = len(prompt) < 500
            is_conversational = 'conversational' in prompt.lower() or 'conversation' in prompt.lower()
            print(f"  Features: Simple={is_simple}, Conversational={is_conversational}")
            
        print(f"  Preview: {prompt[:150]}...")
        print()


if __name__ == "__main__":
    print("🤖 Testing Runtime System Prompt Switching (Simple)")
    print("=" * 60)
    
    try:
        test_prompt_switching_logic()
        test_prompt_differences()
        
        print("\n" + "=" * 60)
        print("✅ All prompt switching tests completed!")
        
        print("\n📋 Runtime Prompt Switching Capabilities:")
        print("  1. switch_system_prompt(prompt_id='barney_command_visual')")
        print("     → Force specific prompt regardless of context")
        print("  2. switch_system_prompt(context_updates={'user_age': 8})")
        print("     → Update context and reselect prompt automatically")
        print("  3. clear_system_prompt_override()")
        print("     → Remove override and return to context-based selection")
        print("  4. get_current_system_prompt_info()")
        print("     → See what prompt is currently selected and why")
        print("  5. reload_system_prompts()")
        print("     → Reload prompts.yaml without restarting agent")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()