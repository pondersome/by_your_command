#!/usr/bin/env python3
"""
Test Conversation Timeout Functionality

Verifies that conversation IDs change after timeout and context is reset.

Author: Karim Virani
Version: 1.0  
Date: August 2025
"""

import asyncio
import logging
import time
from datetime import datetime

from conversation_monitor import ConversationMonitor


async def test_conversation_timeout():
    """Test basic conversation timeout detection"""
    print("Testing Conversation Timeout...")
    
    # Track conversation changes
    changes = []
    
    def on_change(old_id, new_id):
        changes.append((old_id, new_id, time.time()))
        print(f"🔄 Conversation changed: {old_id[-12:]} → {new_id[-12:]}")
    
    # Create monitor with 5 second timeout for testing
    monitor = ConversationMonitor(timeout=5.0, on_conversation_change=on_change)
    
    print(f"Initial conversation ID: {monitor.current_conversation_id}")
    print(f"Timeout set to: {monitor.timeout}s")
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Simulate activity
    print("\n📨 Simulating voice chunks...")
    for i in range(3):
        monitor.record_activity()
        print(f"  Activity recorded at {i+1}s")
        await asyncio.sleep(1)
    
    print("\n⏳ Waiting for timeout...")
    await asyncio.sleep(6)  # Wait for timeout
    
    # Check results
    print(f"\nConversations started: {monitor.conversations_started}")
    print(f"Timeouts detected: {monitor.timeouts_detected}")
    print(f"Changes recorded: {len(changes)}")
    
    if changes:
        print("\nConversation changes:")
        for old, new, timestamp in changes:
            print(f"  {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')} - {old[-12:]} → {new[-12:]}")
    
    # Test external reset
    print("\n🔄 Testing external reset...")
    new_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    result = monitor.handle_external_reset(new_id)
    print(f"External reset result: {result}")
    print(f"Current ID: {monitor.current_conversation_id[-12:]}")
    
    # Get final metrics
    print("\n📊 Final metrics:")
    metrics = monitor.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    # Stop monitoring
    await monitor.stop_monitoring()
    print("\n✅ Test complete!")


async def test_conversation_monitor_with_agent():
    """Test conversation monitor integrated with mock agent behavior"""
    print("\nTesting Conversation Monitor with Agent Integration...\n")
    
    class MockAgent:
        def __init__(self):
            self.conversation_contexts = []
            self.current_context = []
            
        def reset_context(self):
            if self.current_context:
                self.conversation_contexts.append(self.current_context)
                self.current_context = []
            print("  🧹 Agent context reset")
            
        def add_to_context(self, text):
            self.current_context.append(text)
            print(f"  📝 Added to context: {text}")
    
    agent = MockAgent()
    
    def on_change(old_id, new_id):
        print(f"🔄 Conversation changed: {old_id[-12:]} → {new_id[-12:]}")
        agent.reset_context()
    
    # Create monitor with 3 second timeout
    monitor = ConversationMonitor(timeout=3.0, on_conversation_change=on_change)
    await monitor.start_monitoring()
    
    # Simulate conversation
    print("📨 Simulating conversation...")
    
    # First conversation
    monitor.record_activity()
    agent.add_to_context("User: Hello robot")
    await asyncio.sleep(1)
    
    monitor.record_activity()
    agent.add_to_context("Robot: Hello! How can I help?")
    await asyncio.sleep(1)
    
    monitor.record_activity()
    agent.add_to_context("User: What's the weather?")
    
    # Wait for timeout
    print("\n⏳ Waiting for timeout...")
    await asyncio.sleep(4)
    
    # New conversation after timeout
    print("\n📨 New conversation after timeout...")
    monitor.record_activity()
    agent.add_to_context("User: Hey there")
    await asyncio.sleep(1)
    
    monitor.record_activity()
    agent.add_to_context("Robot: Hi! How can I assist?")
    
    # Save final context
    if agent.current_context:
        agent.conversation_contexts.append(agent.current_context)
    
    # Display results
    print(f"\n📊 Total conversations: {len(agent.conversation_contexts)}")
    for i, context in enumerate(agent.conversation_contexts):
        print(f"\nConversation {i+1}:")
        for line in context:
            print(f"  {line}")
    
    await monitor.stop_monitoring()
    print("\n✅ Integration test complete!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    asyncio.run(test_conversation_timeout())
    asyncio.run(test_conversation_monitor_with_agent())