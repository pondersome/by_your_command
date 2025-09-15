#!/usr/bin/env python3
"""
Test the new hybrid Gemini agent implementation
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
        format='%(name)s: %(message)s'
    )
    
    # Create config
    config = {
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
        'model': 'models/gemini-2.0-flash-live-001',
        'voice': 'Kore',
        'session_pause_timeout': 10.0,
        'conversation_timeout': 600.0,
        'system_prompt': 'You are a helpful assistant. Be concise.',
        'response_voice_topic': 'response_voice',
        'transcript_topic': 'response_text',
        'interruption_signal_topic': 'interruption_signal'
    }
    
    print("=" * 60)
    print("Testing Hybrid Gemini Live Agent")
    print("=" * 60)
    
    # Create agent
    agent = GeminiLiveAgent(config)
    
    # Initialize
    print("\n1. Initializing agent...")
    await agent.initialize()
    print("✅ Agent initialized")
    
    # Check components
    print("\n2. Checking components...")
    print(f"  Bridge: {'Connected' if agent.bridge_interface else 'Not connected (standalone)'}")
    print(f"  Session Manager: {agent.session_manager.state.value}")
    print(f"  Receive Coordinator: {'Ready' if agent.receive_coordinator else 'Not initialized'}")
    
    # Run for a short time
    print("\n3. Running agent for 5 seconds...")
    
    run_task = asyncio.create_task(agent.run())
    await asyncio.sleep(5)
    
    # Get metrics
    print("\n4. Metrics:")
    metrics = agent.get_metrics()
    print(f"  Messages processed: {metrics.get('messages_processed', 0)}")
    print(f"  Sessions created: {metrics.get('sessions_created', 0)}")
    print(f"  Errors: {metrics.get('errors', 0)}")
    
    # Check coordinator metrics
    if 'receive_coordinator' in metrics:
        coord_metrics = metrics['receive_coordinator']
        print(f"\n  Coordinator metrics:")
        print(f"    Audio chunks sent: {coord_metrics.get('audio_chunks_sent', 0)}")
        print(f"    Responses received: {coord_metrics.get('responses_received', 0)}")
        print(f"    Turns completed: {coord_metrics.get('turns_completed', 0)}")
    
    # Stop agent
    print("\n5. Stopping agent...")
    agent.running = False
    run_task.cancel()
    
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    
    await agent.cleanup()
    print("✅ Agent stopped cleanly")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    if metrics.get('errors', 0) == 0:
        print("✅ No errors - architecture is working!")
    else:
        print(f"⚠️ Had {metrics['errors']} errors")
    
    print("\nKey architectural changes:")
    print("  • Clean separation of concerns")
    print("  • Receive coordinator manages generator lifecycle")
    print("  • Bridge interface identical to OpenAI")
    print("  • Simplified session management")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())