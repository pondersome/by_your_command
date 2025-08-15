#!/usr/bin/env python3
"""
Test WebSocket Connection Retry Logic

Tests the retry behavior when bridge is not available.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import asyncio
import logging
import sys
import os

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from websocket_bridge import WebSocketBridgeInterface


async def test_connection_retry():
    """Test connection retry behavior"""
    print("🔄 Testing WebSocket Connection Retry Logic")
    print("=" * 50)
    
    # Configure logging to see retry attempts
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    # Create bridge interface with short retry settings for testing
    config = {
        'bridge_connection': {
            'host': 'localhost',
            'port': 8765,  # Assuming bridge is not running on this port
            'reconnect_interval': 2.0,  # Short interval for testing
            'max_reconnect_attempts': 3  # Few attempts for testing
        },
        'agent_id': 'test_agent'
    }
    
    print("Configuration:")
    bridge_config = config['bridge_connection']
    print(f"  Host: {bridge_config['host']}")
    print(f"  Port: {bridge_config['port']}")
    print(f"  Retry interval: {bridge_config['reconnect_interval']}s")
    print(f"  Max attempts: {bridge_config['max_reconnect_attempts']}")
    print()
    
    bridge = WebSocketBridgeInterface(config)
    
    print("Testing initial connection with retry...")
    print("(This should fail since no bridge is running)")
    print()
    
    import time
    start_time = time.time()
    
    success = await bridge.connect_with_retry()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print()
    print("Results:")
    print(f"  Success: {success}")
    print(f"  Total duration: {duration:.1f}s")
    print(f"  Connection attempts: {bridge.connection_attempts}")
    print(f"  Currently connected: {bridge.is_connected()}")
    
    # Test metrics
    metrics = bridge.get_metrics()
    print(f"  Metrics: {metrics}")


async def test_single_connection_attempt():
    """Test single connection attempt (old behavior)"""
    print("\n🔄 Testing Single Connection Attempt")
    print("=" * 50)
    
    config = {
        'bridge_connection': {
            'host': 'localhost',
            'port': 8765,
            'reconnect_interval': 5.0,
            'max_reconnect_attempts': 10
        },
        'agent_id': 'test_agent'
    }
    
    bridge = WebSocketBridgeInterface(config)
    
    print("Testing single connection attempt...")
    
    import time
    start_time = time.time()
    
    success = await bridge.connect()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print()
    print("Results:")
    print(f"  Success: {success}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Connection attempts: {bridge.connection_attempts}")
    print("  → This should fail quickly with just one attempt")


if __name__ == "__main__":
    print("🤖 Testing WebSocket Connection Retry Logic")
    print("=" * 60)
    
    try:
        asyncio.run(test_connection_retry())
        asyncio.run(test_single_connection_attempt())
        
        print("\n" + "=" * 60)
        print("✅ Connection retry tests completed!")
        print("\n📋 Expected Behavior:")
        print("  • connect_with_retry() should attempt multiple connections")
        print("  • Each attempt should wait reconnect_interval between tries")
        print("  • connect() should only try once and fail quickly")
        print("  • Both should handle connection failures gracefully")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()