#!/usr/bin/env python3
"""Test Gemini agent connection to WebSocket bridge"""

import asyncio
import websockets
import json

async def test_bridge_server():
    """Simple WebSocket server to test if agent can connect"""
    async def handle_client(websocket, path):
        print(f"✅ Client connected from {websocket.remote_address}")
        
        # Send handshake
        handshake = {
            "type": "handshake",
            "agent_id": "test_bridge",
            "capabilities": ["audio", "text"]
        }
        await websocket.send(json.dumps(handshake))
        
        # Listen for messages
        try:
            async for message in websocket:
                data = json.loads(message)
                print(f"📨 Received: {data.get('type', 'unknown')}")
                
                if data.get('type') == 'handshake':
                    print(f"🤝 Agent identified as: {data.get('agent_id')}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("❌ Client disconnected")
            
    print("🚀 Starting test WebSocket server on port 8765...")
    async with websockets.serve(handle_client, "localhost", 8765):
        print("📡 Server listening on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(test_bridge_server())