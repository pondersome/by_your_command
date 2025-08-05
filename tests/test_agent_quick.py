#!/usr/bin/env python3
"""Quick test to see if conversation_id is published"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.oai_realtime.oai_realtime_agent import OpenAIRealtimeAgent

async def test_conversation_publish():
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY', 'test-key'),
        'conversation_timeout': 600.0,
        'session_pause_timeout': 10.0,
        'bridge_connection': {
            'host': 'localhost',
            'port': 8765,
            'max_reconnect_attempts': 2,  # Reduce to fail faster
            'reconnect_interval': 2.0
        }
    }
    
    print("Creating agent...")
    agent = OpenAIRealtimeAgent(config)
    
    # Initialize 
    print("Initializing agent...")
    await agent.initialize()
    
    print(f"Bridge connected: {agent.bridge_interface is not None and agent.bridge_interface.is_connected()}")
    print(f"Conversation ID: {agent.conversation_monitor.current_conversation_id}")
    
    # Give it a moment to process
    await asyncio.sleep(2)
    
    # Stop
    await agent.stop()
    
if __name__ == "__main__":
    # Make sure bridge is running
    print("Make sure ros_ai_bridge is running!")
    asyncio.run(test_conversation_publish())