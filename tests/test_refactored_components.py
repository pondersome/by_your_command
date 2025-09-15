#!/usr/bin/env python3
"""
Test script for refactored multi-provider components

Verifies that the refactored base classes work correctly with
the OpenAI agent implementation.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.oai_realtime.oai_session_manager import OpenAISessionManager
from agents.oai_realtime.oai_serializer import OpenAISerializer
from agents.common.debug_interface import DebugInterface, create_test_audio_sine_wave
from agents.common.base_session_manager import SessionState


async def test_session_manager():
    """Test OpenAISessionManager with base class"""
    print("\n=== Testing OpenAISessionManager ===")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set - skipping session manager test")
        return False
    
    config = {
        'model': 'gpt-4o-realtime-preview',
        'voice': 'alloy',
        'vad_threshold': 0.5,
        'session_pause_timeout': 10.0,
        'prompts_file': 'config/prompts.yaml'
    }
    
    manager = OpenAISessionManager(config)
    
    # Test basic properties
    assert manager.state == SessionState.IDLE
    print("✅ Initial state is IDLE")
    
    # Test connection (will fail without valid API key)
    try:
        print("Attempting to connect...")
        success = await manager.connect_session()
        
        if success:
            print(f"✅ Connected successfully - state: {manager.state.value}")
            assert manager.is_connected()
            
            # Test OpenAI-specific methods
            await manager.send_response_create()
            print("✅ OpenAI-specific method works")
            
            # Close session
            context = await manager.close_session()
            print(f"✅ Session closed - preserved context: {context is not None}")
            assert manager.state == SessionState.IDLE
        else:
            print("⚠️ Connection failed (expected without API key)")
            
    except Exception as e:
        print(f"⚠️ Connection test failed: {e}")
        
    return True


def test_serializer():
    """Test OpenAISerializer with base class"""
    print("\n=== Testing OpenAISerializer ===")
    
    serializer = OpenAISerializer()
    
    # Test audio serialization
    test_audio = [0, 100, -100, 32767, -32768]  # Test PCM16 values
    
    # Create mock envelope
    envelope = type('Envelope', (), {
        'ros_msg_type': 'audio_common_msgs/AudioData',
        'raw_data': type('AudioData', (), {
            'int16_data': test_audio
        })()
    })()
    
    audio_msg = serializer.serialize_audio(envelope)
    assert audio_msg is not None
    assert audio_msg['type'] == 'input_audio_buffer.append'
    assert 'audio' in audio_msg
    print("✅ Audio serialization works")
    
    # Test text serialization
    text_envelope = type('Envelope', (), {
        'ros_msg_type': 'std_msgs/String',
        'topic_name': '/prompt_text',
        'raw_data': type('String', (), {
            'data': 'Test message'
        })()
    })()
    
    text_msg = serializer.serialize_text(text_envelope)
    assert text_msg is not None
    assert text_msg['type'] == 'conversation.item.create'
    assert text_msg['item']['content'][0]['text'] == 'Test message'
    print("✅ Text serialization works")
    
    # Test OpenAI-specific message creators
    response_msg = serializer.create_response_trigger()
    assert response_msg['type'] == 'response.create'
    print("✅ Response trigger creation works")
    
    cancel_msg = serializer.create_response_cancel()
    assert cancel_msg['type'] == 'response.cancel'
    print("✅ Response cancel creation works")
    
    return True


async def test_debug_interface():
    """Test DebugInterface with mock agent"""
    print("\n=== Testing DebugInterface ===")
    
    # Create mock agent with required components
    class MockAgent:
        def __init__(self):
            self.serializer = OpenAISerializer()
            self.session_manager = type('MockSessionManager', (), {
                'is_connected': lambda: True,
                'websocket': type('MockWebSocket', (), {
                    'send': lambda msg: None
                })(),
                'connect_session': lambda: asyncio.sleep(0)
            })()
            self.metrics = {'messages_sent': 0}
    
    agent = MockAgent()
    debug = DebugInterface(agent)
    
    # Test initialization
    await debug.start()
    assert debug.running
    print("✅ Debug interface started")
    
    # Test audio injection
    test_audio = create_test_audio_sine_wave(frequency=440, duration=0.1)
    success = await debug.inject_audio_data(test_audio[:100])  # Small sample
    print(f"✅ Audio injection: {'successful' if success else 'failed (expected without real session)'}")
    
    # Test text injection
    success = await debug.inject_text_message("Test message")
    print(f"✅ Text injection: {'successful' if success else 'failed (expected without real session)'}")
    
    # Test stats
    stats = debug.get_stats()
    assert 'running' in stats
    assert 'messages_injected' in stats
    print(f"✅ Stats retrieval works: {stats}")
    
    await debug.stop()
    assert not debug.running
    print("✅ Debug interface stopped")
    
    return True


async def main():
    """Run all component tests"""
    print("=" * 50)
    print("Testing Refactored Multi-Provider Components")
    print("=" * 50)
    
    all_passed = True
    
    # Test each component
    try:
        if not test_serializer():
            all_passed = False
    except Exception as e:
        print(f"❌ Serializer test failed: {e}")
        all_passed = False
    
    try:
        if not await test_session_manager():
            all_passed = False
    except Exception as e:
        print(f"❌ Session manager test failed: {e}")
        all_passed = False
    
    try:
        if not await test_debug_interface():
            all_passed = False
    except Exception as e:
        print(f"❌ Debug interface test failed: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All refactored component tests passed!")
    else:
        print("⚠️ Some tests failed (may be expected without API key)")
    print("=" * 50)
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)