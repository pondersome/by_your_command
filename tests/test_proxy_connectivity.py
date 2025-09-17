#!/usr/bin/env python3
"""
Test script to verify WebSocket proxy connectivity.

This script:
1. Connects to the proxy as a test agent
2. Sends a registration message
3. Publishes a test message
4. Verifies the proxy is working
"""

import asyncio
import json
import websockets
import sys
import logging


async def test_proxy():
    """Test proxy connectivity"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    proxy_url = "ws://localhost:8766"

    try:
        logger.info(f"🔗 Connecting to proxy at {proxy_url}")
        async with websockets.connect(proxy_url) as websocket:
            logger.info("✅ Connected to proxy!")

            # Send registration
            registration = {
                "type": "register",
                "agent_id": "test_agent",
                "subscriptions": ["prompt_text", "response_text"],
                "metadata": {
                    "agent_role": "test",
                    "sibling_agents": []
                }
            }

            logger.info("📤 Sending registration...")
            await websocket.send(json.dumps(registration))

            # Send a test message
            test_message = {
                "type": "outbound_message",
                "topic": "response_text",
                "msg_type": "std_msgs/String",
                "data": {"data": "Test message from test agent"},
                "metadata": {"agent_id": "test_agent", "agent_role": "test"}
            }

            logger.info("📤 Sending test message...")
            await websocket.send(json.dumps(test_message))

            # Listen for any responses
            logger.info("👂 Listening for responses (5 seconds)...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                logger.info(f"📨 Received: {response}")
            except asyncio.TimeoutError:
                logger.info("⏱️ No response received (timeout)")

            logger.info("✅ Test complete!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_proxy())