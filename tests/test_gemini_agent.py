#!/usr/bin/env python3
"""
Test the Gemini Live agent with our fixes
"""

import asyncio
import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gemini_live.gemini_live_agent import GeminiLiveAgent

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s:%(message)s'
    )
    
    # Create minimal config
    config = {
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
        'model': 'models/gemini-2.0-flash-live-001',
        'voice': 'Kore',
        'enable_video': False,
        'session_pause_timeout': 10.0,
        'system_prompt': 'You are a helpful assistant. Be concise.',
        'log_level': logging.INFO
    }
    
    print("=" * 60)
    print("Testing Gemini Live Agent with fixes")
    print("=" * 60)
    
    # Create agent
    agent = GeminiLiveAgent(config)
    
    # Initialize
    print("\n1. Initializing agent...")
    await agent.initialize()
    print("✅ Agent initialized")
    
    # Run for a short time to test session creation
    print("\n2. Running agent (will stop after 10 seconds)...")
    
    # Create a task to run the agent
    run_task = asyncio.create_task(agent.run())
    
    # Wait 10 seconds
    await asyncio.sleep(10)
    
    print("\n3. Checking metrics...")
    metrics = agent.get_metrics()
    print(f"  Sessions created: {metrics.get('sessions_created', 0)}")
    print(f"  Messages processed: {metrics.get('messages_processed', 0)}")
    print(f"  Responses received: {metrics.get('responses_received', 0)}")
    
    # Stop the agent
    print("\n4. Stopping agent...")
    agent.running = False
    run_task.cancel()
    
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    
    await agent.cleanup()
    print("✅ Agent stopped cleanly")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    if metrics.get('errors', 0) > 0:
        print(f"❌ Had {metrics['errors']} errors")
    else:
        print("✅ No errors detected")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())