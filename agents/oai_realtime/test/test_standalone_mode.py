#!/usr/bin/env python3
"""
Test Standalone Mode for OpenAI Realtime Agent

Demonstrates how to use the agent in standalone mode without ROS bridge,
including audio injection and direct testing capabilities.

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

from oai_realtime_agent import OpenAIRealtimeAgent
from debug_interface import create_test_audio_sine_wave, create_test_audio_noise


async def test_standalone_agent():
    """Test agent in standalone mode"""
    print("🤖 Testing OpenAI Realtime Agent - Standalone Mode")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Agent configuration (bridge will fail to connect)
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY', 'test_key_not_real'),
        'model': 'gpt-4o-realtime-preview',
        'voice': 'alloy',
        'user_age': 30,  # Adult user -> Barney prompt
        'environment': 'normal',
        'robot_name': 'Barney',
        'agent_id': 'test_agent_standalone',
        
        # Bridge connection (will fail)
        'bridge_connection': {
            'host': 'localhost',
            'port': 8765,  # No bridge running
            'reconnect_interval': 1.0,  # Fast for testing
            'max_reconnect_attempts': 2  # Quick failure
        },
        
        # Session settings
        'session_pause_timeout': 5.0,
        'session_max_duration': 60.0,
        'max_context_tokens': 1000
    }
    
    print("1. Initializing Agent")
    print("-" * 30)
    print("Configuration:")
    print(f"   API Key: {'✅ Set' if config['openai_api_key'] != 'test_key_not_real' else '❌ Test key (no real OpenAI calls)'}")
    print(f"   User Age: {config['user_age']} (should select Barney prompt)")
    print(f"   Bridge: {config['bridge_connection']['host']}:{config['bridge_connection']['port']} (should fail)")
    print()
    
    agent = OpenAIRealtimeAgent(config)
    
    try:
        await agent.initialize()
        
        print("2. Agent Status")
        print("-" * 30)
        print(f"   Standalone mode: {agent.is_standalone_mode()}")
        print(f"   Bridge connected: {agent.bridge_interface is not None}")
        print(f"   Debug interface: {agent.debug_interface is not None}")
        
        # Show prompt information
        prompt_info = agent.get_current_system_prompt_info()
        print(f"   Current prompt: {prompt_info.get('prompt_id', 'unknown')}")
        print(f"   Selection type: {prompt_info.get('selection_type', 'unknown')}")
        print()
        
        if not agent.is_standalone_mode():
            print("❌ Agent not in standalone mode - stopping test")
            return
            
        print("3. Testing Debug Interface")
        print("-" * 30)
        
        # Start agent in background
        agent_task = asyncio.create_task(agent.run())
        
        # Give agent time to start
        await asyncio.sleep(1)
        
        # Test 1: Inject sine wave audio
        print("   a) Injecting test sine wave (440Hz, 1 second)...")
        sine_audio = create_test_audio_sine_wave(440, 1.0)
        success = await agent.debug_inject_audio(
            sine_audio, 
            utterance_id="test_sine_440", 
            is_utterance_end=True
        )
        print(f"      Result: {'✅ Success' if success else '❌ Failed'}")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Test 2: Inject different frequency
        print("   b) Injecting test sine wave (880Hz, 0.5 seconds)...")
        sine_audio_high = create_test_audio_sine_wave(880, 0.5)
        success = await agent.debug_inject_audio(
            sine_audio_high,
            utterance_id="test_sine_880",
            is_utterance_end=True
        )
        print(f"      Result: {'✅ Success' if success else '❌ Failed'}")
        
        await asyncio.sleep(2)
        
        # Test 3: Inject noise
        print("   c) Injecting white noise (0.3 seconds)...")
        noise_audio = create_test_audio_noise(0.3)
        success = await agent.debug_inject_audio(
            noise_audio,
            utterance_id="test_noise",
            is_utterance_end=True
        )
        print(f"      Result: {'✅ Success' if success else '❌ Failed'}")
        
        await asyncio.sleep(2)
        
        # Test 4: Inject text
        print("   d) Injecting text message...")
        success = await agent.debug_inject_text("Hello Barney, this is a test message!")
        print(f"      Result: {'✅ Success' if success else '❌ Failed'}")
        
        await asyncio.sleep(1)
        
        print("\n4. Agent Statistics")
        print("-" * 30)
        
        # Agent metrics
        metrics = agent.get_metrics()
        print("   Agent metrics:")
        print(f"      Messages processed: {metrics['messages_processed']}")
        print(f"      Messages to OpenAI: {metrics['messages_sent_to_openai']}")
        print(f"      Messages to ROS: {metrics['messages_sent_to_ros']}")
        print(f"      Runtime: {metrics.get('current_runtime', 0):.1f}s")
        
        # Debug stats
        debug_stats = agent.get_debug_stats()
        print("   Debug interface:")
        for key, value in debug_stats.items():
            print(f"      {key}: {value}")
        
        # Session status
        print("   Session status:")
        session_metrics = metrics.get('session_manager', {})
        print(f"      State: {session_metrics.get('state', 'unknown')}")
        print(f"      Sessions created: {session_metrics.get('sessions_created', 0)}")
        print(f"      Current duration: {session_metrics.get('current_session_duration', 0):.1f}s")
        
        print("\n5. Testing Prompt Switching")
        print("-" * 30)
        
        # Switch to child mode
        print("   a) Switching to child mode context...")
        success = await agent.switch_system_prompt(context_updates={'user_age': 7})
        if success:
            new_prompt_info = agent.get_current_system_prompt_info()
            print(f"      New prompt: {new_prompt_info.get('prompt_id', 'unknown')}")
        
        # Override to specific prompt
        print("   b) Overriding to friendly_assistant...")
        success = await agent.switch_system_prompt(prompt_id='friendly_assistant')
        if success:
            override_info = agent.get_current_system_prompt_info()
            print(f"      Override prompt: {override_info.get('prompt_id', 'unknown')}")
            print(f"      Selection type: {override_info.get('selection_type', 'unknown')}")
        
        # Clear override
        print("   c) Clearing override...")
        agent.clear_system_prompt_override()
        final_info = agent.get_current_system_prompt_info()
        print(f"      Final prompt: {final_info.get('prompt_id', 'unknown')}")
        
        print("\n✅ Standalone mode testing complete!")
        print("\n💡 Key Capabilities Demonstrated:")
        print("   • Agent runs without ROS bridge connection")
        print("   • Debug interface allows direct audio/text injection")
        print("   • Named prompt system works in standalone mode")
        print("   • Session management handles injected messages")
        print("   • Runtime prompt switching functions correctly")
        print("   • Metrics and monitoring work in standalone mode")
        
        # Stop the agent
        await agent.stop()
        await agent_task
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            await agent.stop()
        except:
            pass


async def demo_audio_generation():
    """Demonstrate audio generation for testing"""
    print("\n🎵 Audio Generation Demo")
    print("=" * 30)
    
    # Generate different test audio types
    print("Generating test audio samples:")
    
    sine_440 = create_test_audio_sine_wave(440, 0.5)  # A4 note
    sine_880 = create_test_audio_sine_wave(880, 0.3)  # A5 note  
    noise = create_test_audio_noise(0.2)
    
    print(f"   440Hz sine wave: {len(sine_440)} samples (0.5s)")
    print(f"   880Hz sine wave: {len(sine_880)} samples (0.3s)")
    print(f"   White noise: {len(noise)} samples (0.2s)")
    
    print("\n📊 Sample Analysis:")
    import numpy as np
    
    for name, data in [("440Hz", sine_440), ("880Hz", sine_880), ("Noise", noise)]:
        np_data = np.array(data)
        print(f"   {name}: min={np_data.min()}, max={np_data.max()}, mean={np_data.mean():.1f}")


if __name__ == "__main__":
    print("🚀 OpenAI Realtime Agent - Standalone Mode Test")
    print("=" * 70)
    
    if os.getenv('OPENAI_API_KEY'):
        print("🔑 Real OpenAI API key detected - agent will make actual API calls")
    else:
        print("🔧 No API key - agent will run in test mode (may fail session creation)")
    
    print()
    
    try:
        asyncio.run(test_standalone_agent())
        asyncio.run(demo_audio_generation())
        
    except KeyboardInterrupt:
        print("\n⌨️  Interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()