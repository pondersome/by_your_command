#!/usr/bin/env python3
"""
Test Runtime System Prompt Switching

Demonstrates the ability to change system prompts during agent runtime,
both with specific prompt overrides and context-based selection.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import sys
import os
import asyncio
import logging

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from oai_session_manager import OpenAISessionManager as SessionManager


async def test_runtime_prompt_switching():
    """Test runtime prompt switching capabilities"""
    print("🔄 Testing Runtime System Prompt Switching")
    print("=" * 50)
    
    # Configure logging to see the switch messages
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    # Create session manager with adult user context
    config = {
        'user_age': 30,
        'environment': 'normal', 
        'robot_name': 'Barney',
        'agent_id': 'openai_realtime',
        'openai_api_key': 'test_key',
        'voice': 'alloy',
        'max_context_tokens': 2000,
        'max_context_age': 3600
    }
    
    session_manager = SessionManager(config)
    
    print("1. Initial State")
    print("-" * 20)
    initial_info = session_manager.get_effective_prompt_selection()
    print(f"Selection type: {initial_info['selection_type']}")
    print(f"Prompt ID: {initial_info.get('prompt_id', 'unknown')}")
    print(f"Context: {initial_info['context']}")
    print()
    
    print("2. Available Prompts")
    print("-" * 20)
    available_prompts = session_manager.list_available_prompts()
    print(f"Available prompts: {available_prompts}")
    print()
    
    print("3. Switch to Child Mode (Context Update)")
    print("-" * 20)
    success = await session_manager.switch_prompt(context_updates={'user_age': 7})
    if success:
        child_info = session_manager.get_effective_prompt_selection()
        print(f"✅ Switch successful")
        print(f"New prompt ID: {child_info.get('prompt_id', 'unknown')}")
        print(f"Selection type: {child_info['selection_type']}")
    else:
        print("❌ Switch failed")
    print()
    
    print("4. Override to Specific Prompt")
    print("-" * 20)
    success = await session_manager.switch_prompt(prompt_id='barney_command_visual')
    if success:
        override_info = session_manager.get_effective_prompt_selection()
        print(f"✅ Override successful")
        print(f"Selection type: {override_info['selection_type']}")
        print(f"Overridden prompt: {override_info.get('prompt_id', 'unknown')}")
    else:
        print("❌ Override failed")
    print()
    
    print("5. Try Invalid Prompt")
    print("-" * 20)
    success = await session_manager.switch_prompt(prompt_id='nonexistent_prompt')
    if not success:
        print("✅ Correctly rejected invalid prompt")
    else:
        print("❌ Should have rejected invalid prompt")
    print()
    
    print("6. Clear Override (Return to Context)")
    print("-" * 20)
    session_manager.clear_prompt_override()
    cleared_info = session_manager.get_effective_prompt_selection()
    print(f"Selection type: {cleared_info['selection_type']}")
    print(f"Context-selected prompt: {cleared_info.get('prompt_id', 'unknown')}")
    print(f"Current context: {cleared_info['context']}")
    print()
    
    print("7. Switch Back to Adult Context")
    print("-" * 20)
    success = await session_manager.switch_prompt(context_updates={'user_age': 35, 'environment': 'quiet'})
    if success:
        adult_info = session_manager.get_effective_prompt_selection()
        print(f"✅ Context updated successfully")
        print(f"New prompt ID: {adult_info.get('prompt_id', 'unknown')}")
        print(f"Updated context: {adult_info['context']}")


async def test_prompt_content_switching():
    """Test that different prompts actually have different content"""
    print("\n🎭 Testing Prompt Content Differences")
    print("=" * 50)
    
    config = {'user_age': 25, 'environment': 'normal', 'openai_api_key': 'test'}
    session_manager = SessionManager(config)
    
    # Test different prompt selections
    contexts = [
        {'user_age': 25, 'environment': 'normal'},  # Should get barney_command_visual
        {'user_age': 5, 'environment': 'normal'},   # Should get friendly_assistant
    ]
    
    for i, context in enumerate(contexts, 1):
        print(f"{i}. Context: {context}")
        
        # Update context and check selection
        session_manager.update_prompt_context(context)
        selection_info = session_manager.get_effective_prompt_selection()
        prompt_id = selection_info.get('prompt_id', 'unknown')
        
        # Get the actual prompt content
        selected_prompt = session_manager.prompt_loader.select_prompt(context)
        
        print(f"   Selected prompt: {prompt_id}")
        print(f"   Content length: {len(selected_prompt)} characters")
        print(f"   Preview: {selected_prompt[:100]}...")
        
        # Show key characteristics
        if 'barney' in prompt_id.lower():
            if 'PRESET:' in selected_prompt:
                print("   → ✅ Contains command preset system")
            if 'skid-steer robot' in selected_prompt:
                print("   → ✅ Contains Barney robot identity")
        elif 'friendly' in prompt_id.lower():
            if 'simple' in selected_prompt.lower() or len(selected_prompt) < 500:
                print("   → ✅ Simple conversational prompt")
        print()


async def demonstrate_session_update():
    """Demonstrate what happens when we have an active session (simulated)"""
    print("\n🔄 Simulating Active Session Prompt Update")
    print("=" * 50)
    
    config = {'user_age': 30, 'environment': 'normal', 'openai_api_key': 'test'}
    session_manager = SessionManager(config)
    
    print("Simulating session states...")
    print(f"Current state: {session_manager.state.value}")
    
    # Show what would happen in different states
    print("\nIf session were ACTIVE:")
    print("  → switch_prompt() would send session.update to OpenAI WebSocket")
    print("  → New instructions would be applied immediately")
    print("  → User would notice personality change in next response")
    
    print("\nIf session is IDLE:")
    print("  → switch_prompt() stores new prompt selection")
    print("  → Change takes effect when next session is created")
    print("  → Seamless transition for user")
    
    # Show the actual message that would be sent
    test_prompt = session_manager.prompt_loader.select_prompt()
    system_prompt = session_manager.context_manager.build_system_prompt(test_prompt, None)
    
    print(f"\nExample session.update message structure:")
    config_msg = {
        "type": "session.update",
        "session": {
            "instructions": f"{system_prompt[:200]}...[truncated]",
            "modalities": ["text", "audio"],
            "voice": "alloy",
            # ... other config
        }
    }
    print(f"Message size: ~{len(str(config_msg))} characters")


if __name__ == "__main__":
    print("🤖 Testing Runtime System Prompt Switching")
    print("=" * 60)
    
    try:
        asyncio.run(test_runtime_prompt_switching())
        asyncio.run(test_prompt_content_switching())
        asyncio.run(demonstrate_session_update())
        
        print("\n" + "=" * 60)
        print("✅ All runtime prompt switching tests completed!")
        print("\n📝 Summary of Capabilities:")
        print("   • Switch prompts by ID override")
        print("   • Switch prompts by context updates") 
        print("   • Apply changes to active sessions immediately")
        print("   • Store changes for next session if no active session")
        print("   • Clear overrides to return to context-based selection")
        print("   • Reload prompts from file without restart")
        print("   • Get current prompt information and available options")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()