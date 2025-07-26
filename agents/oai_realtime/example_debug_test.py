#!/usr/bin/env python3
"""
Example Debug Test Script

Practical example of testing the OpenAI Realtime Agent in standalone mode.
This demonstrates how to create test scripts for the agent.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import asyncio
import logging
import os
import sys

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from debug_interface import create_test_audio_sine_wave


async def simple_agent_test():
    """Simple test of agent with debug interface"""
    
    # Import here to avoid import issues if modules not ready
    try:
        from oai_realtime_agent import OpenAIRealtimeAgent
    except ImportError as e:
        print(f"❌ Cannot import agent: {e}")
        print("This is expected if context/session_manager modules have import issues")
        return
    
    print("🧪 Simple Agent Debug Test")
    print("=" * 30)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create agent config
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY', 'test_key'),
        'model': 'gpt-4o-realtime-preview',
        'voice': 'alloy',
        'user_age': 30,  # Will use Barney prompt
        'bridge_connection': {
            'host': 'localhost',
            'port': 8765,  # Will fail to connect
            'max_reconnect_attempts': 1  # Fail quickly
        }
    }
    
    print(f"API Key: {'✅ Set' if config['openai_api_key'] != 'test_key' else '❌ Not set (will fail)'}")
    
    # Create and initialize agent
    agent = OpenAIRealtimeAgent(config)
    
    try:
        print("Initializing agent...")
        await agent.initialize()
        
        if not agent.is_standalone_mode():
            print("❌ Agent not in standalone mode")
            return
            
        print("✅ Agent in standalone mode with debug interface")
        
        # Start agent task
        print("Starting agent...")
        agent_task = asyncio.create_task(agent.run())
        
        # Wait for startup
        await asyncio.sleep(1)
        
        # Test audio injection
        print("Generating test audio (440Hz sine wave)...")
        test_audio = create_test_audio_sine_wave(440, 1.0)
        
        print("Injecting audio into agent...")
        success = await agent.debug_inject_audio(
            test_audio,
            utterance_id="example_test",
            is_utterance_end=True
        )
        
        print(f"Audio injection: {'✅ Success' if success else '❌ Failed'}")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check stats
        stats = agent.get_debug_stats()
        metrics = agent.get_metrics()
        
        print("\nResults:")
        print(f"  Messages injected: {stats.get('messages_injected', 0)}")
        print(f"  Messages processed: {metrics.get('messages_processed', 0)}")
        print(f"  Messages to OpenAI: {metrics.get('messages_sent_to_openai', 0)}")
        
        # Stop agent
        await agent.stop()
        await agent_task
        
        print("✅ Test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        try:
            await agent.stop()
        except:
            pass


def show_usage_examples():
    """Show practical usage examples"""
    print("\n📖 Usage Examples")
    print("=" * 20)
    
    print("1. Basic Audio Test:")
    print("```python")
    print("from debug_interface import create_test_audio_sine_wave")
    print("from oai_realtime_agent import OpenAIRealtimeAgent")
    print("")
    print("# Setup")
    print("config = {'openai_api_key': 'your_key', 'user_age': 30}")
    print("agent = OpenAIRealtimeAgent(config)")
    print("await agent.initialize()")
    print("")
    print("# Test")
    print("audio = create_test_audio_sine_wave(440, 2.0)")
    print("await agent.debug_inject_audio(audio, is_utterance_end=True)")
    print("```")
    print()
    
    print("2. Prompt Testing:")
    print("```python")
    print("# Test adult vs child prompts")
    print("await agent.switch_system_prompt(context_updates={'user_age': 30})")
    print("await agent.debug_inject_audio(test_audio)  # Uses Barney prompt")
    print("")
    print("await agent.switch_system_prompt(context_updates={'user_age': 7})")
    print("await agent.debug_inject_audio(test_audio)  # Uses friendly prompt")
    print("```")
    print()
    
    print("3. Session Cycle Testing:")
    print("```python")
    print("# Test pause-based session cycling")
    print("await agent.debug_inject_audio(audio1, is_utterance_end=True)")
    print("await asyncio.sleep(12)  # Wait for pause timeout")
    print("await agent.debug_inject_audio(audio2, is_utterance_end=True)")
    print("# Session should cycle between these")
    print("```")
    print()


if __name__ == "__main__":
    print("🚀 OpenAI Realtime Agent - Example Debug Test")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  No OPENAI_API_KEY set - agent may fail to create sessions")
        print("   Set with: export OPENAI_API_KEY='your_key_here'")
        print()
    
    try:
        # Run the test
        asyncio.run(simple_agent_test())
        
        # Show examples
        show_usage_examples()
        
    except KeyboardInterrupt:
        print("\n⌨️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n💡 To run with real OpenAI integration:")
    print("   export OPENAI_API_KEY='your_actual_key'")
    print("   python3 example_debug_test.py")