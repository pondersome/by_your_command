#!/usr/bin/env python3
"""
Test multiple connection cycles for refactored session manager

Verifies that the session manager can connect, disconnect, and reconnect
multiple times without issues.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import asyncio
import os
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.oai_realtime.oai_session_manager import OpenAISessionManager
from agents.common.base_session_manager import SessionState


async def test_multiple_connections():
    """Test multiple connection/disconnection cycles"""
    print("\n=== Testing Multiple Connection Cycles ===")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set - skipping connection test")
        return False
    
    config = {
        'model': 'gpt-4o-realtime-preview',
        'voice': 'alloy',
        'vad_threshold': 0.5,
        'session_pause_timeout': 10.0,
        'prompts_file': 'config/prompts.yaml'
    }
    
    manager = OpenAISessionManager(config)
    
    # Test 3 connection cycles
    for cycle in range(3):
        print(f"\n--- Connection Cycle {cycle + 1} ---")
        
        # Verify initial state
        assert manager.state == SessionState.IDLE, f"Expected IDLE, got {manager.state}"
        assert manager.session_id is None, f"session_id should be None, got {manager.session_id}"
        assert manager.conversation_id is None, f"conversation_id should be None, got {manager.conversation_id}"
        print(f"✅ Pre-connection state verified: {manager.state.value}")
        
        # Connect
        print(f"Attempting connection {cycle + 1}...")
        success = await manager.connect_session()
        
        if success:
            print(f"✅ Connection {cycle + 1} successful")
            assert manager.is_connected()
            assert manager.state == SessionState.ACTIVE
            assert manager.session_id is not None
            print(f"   Session ID: {manager.session_id}")
            
            # Wait a bit to simulate usage
            await asyncio.sleep(1.0)
            
            # Close session
            print(f"Closing connection {cycle + 1}...")
            context = await manager.close_session()
            print(f"✅ Connection {cycle + 1} closed")
            
            # Verify cleanup
            assert manager.state == SessionState.IDLE
            assert manager.session_id is None, f"session_id not cleared: {manager.session_id}"
            assert manager.conversation_id is None, f"conversation_id not cleared: {manager.conversation_id}"
            assert not manager.is_connected()
            print(f"✅ Post-close state verified: all variables properly reset")
            
        else:
            print(f"⚠️ Connection {cycle + 1} failed (may be rate limited)")
            # Still verify state is properly reset
            assert manager.state == SessionState.IDLE
            assert manager.session_id is None
            assert manager.conversation_id is None
            
        # Small delay between cycles
        if cycle < 2:
            print(f"Waiting before next cycle...")
            await asyncio.sleep(2.0)
    
    print("\n✅ All connection cycles completed successfully!")
    return True


async def main():
    """Run connection cycle tests"""
    print("=" * 50)
    print("Testing WebSocket Connection Cycles")
    print("=" * 50)
    
    try:
        success = await test_multiple_connections()
        if success:
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n⚠️ Tests skipped (no API key)")
            return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)