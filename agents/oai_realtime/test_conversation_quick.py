#!/usr/bin/env python3
"""
Quick test for conversation timeout with 5 second timeout
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.oai_realtime.oai_realtime_agent import OpenAIRealtimeAgent

async def test_quick_timeout():
    """Test conversation timeout with 5 second timeout"""
    config = {
        'openai_api_key': 'test-key',  # Won't connect to OpenAI
        'conversation_timeout': 5.0,   # 5 second timeout for testing
        'session_pause_timeout': 10.0,
        'bridge_connection': {
            'host': 'localhost',
            'port': 8765,
            'max_reconnect_attempts': 1  # Fail fast
        }
    }
    
    print("Creating agent with 5 second conversation timeout...")
    agent = OpenAIRealtimeAgent(config)
    
    # Initialize (will run in standalone mode)
    await agent.initialize()
    
    print(f"Initial conversation ID: {agent.conversation_monitor.current_conversation_id}")
    print("\nWaiting 6 seconds for timeout...")
    
    # Wait for timeout
    await asyncio.sleep(6)
    
    # Check if conversation ID changed
    print(f"Current conversation ID: {agent.conversation_monitor.current_conversation_id}")
    print(f"Timeouts detected: {agent.conversation_monitor.timeouts_detected}")
    
    # Cleanup
    await agent.conversation_monitor.stop_monitoring()
    
if __name__ == "__main__":
    asyncio.run(test_quick_timeout())