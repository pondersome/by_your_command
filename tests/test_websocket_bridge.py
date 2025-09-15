#!/usr/bin/env python3
"""
Test WebSocket bridge functionality with simple client
"""
import asyncio
import json
import logging
import websockets
from websockets import WebSocketClientProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWebSocketAgent:
    def __init__(self, host: str = "localhost", port: int = 8765, agent_id: str = "test_agent"):
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.websocket: WebSocketClientProtocol = None
        self.message_count = 0
        
    async def connect(self):
        """Connect to bridge WebSocket server"""
        try:
            uri = f"ws://{self.host}:{self.port}"
            logger.info(f"Connecting to {uri}")
            
            self.websocket = await websockets.connect(uri)
            logger.info("WebSocket connection established")
            
            # Register with bridge
            registration = {
                "type": "register",
                "agent_id": self.agent_id,
                "capabilities": ["audio_processing", "test_agent"],
                "subscriptions": [
                    {"topic": "/prompt_voice", "msg_type": "by_your_command/AudioDataUtterance"},
                    {"topic": "/prompt_text", "msg_type": "std_msgs/String"}
                ]
            }
            
            await self.websocket.send(json.dumps(registration))
            logger.info(f"Sent registration for agent: {self.agent_id}")
            
            # Wait for registration response
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("status") == "success":
                logger.info(f"✅ Registration successful! Session ID: {data.get('session_id')}")
                return True
            else:
                logger.error(f"❌ Registration failed: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
            
    async def listen_for_messages(self, duration: int = 30):
        """Listen for messages from bridge for specified duration"""
        if not self.websocket:
            logger.error("Not connected to WebSocket")
            return
            
        logger.info(f"Listening for messages for {duration} seconds...")
        start_time = asyncio.get_event_loop().time()
        
        try:
            while asyncio.get_event_loop().time() - start_time < duration:
                try:
                    # Listen for messages with timeout
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "message":
                        self.message_count += 1
                        envelope = data["envelope"]
                        logger.info(
                            f"📨 Message {self.message_count}: "
                            f"Topic: {envelope['topic_name']} | "
                            f"Type: {envelope['ros_msg_type']} | "
                            f"Timestamp: {envelope['timestamp']}"
                        )
                        
                        # Log some data details
                        msg_data = envelope.get("data", {})
                        if "audio_data" in msg_data:
                            audio_len = len(msg_data["audio_data"])
                            logger.info(f"   📊 Audio data: {audio_len} samples")
                        elif "data" in msg_data:
                            text_data = str(msg_data["data"])[:50]
                            logger.info(f"   📝 Text data: {text_data}...")
                            
                    elif data.get("type") == "heartbeat":
                        # Respond to heartbeat
                        await self.websocket.send(json.dumps({"type": "heartbeat_response"}))
                        logger.debug("💓 Heartbeat exchanged")
                        
                except asyncio.TimeoutError:
                    # No message - continue listening
                    continue
                    
        except Exception as e:
            logger.error(f"Error listening for messages: {e}")
            
        logger.info(f"✅ Listening completed. Received {self.message_count} messages")
        
    async def send_test_message(self):
        """Send a test message back to ROS"""
        if not self.websocket:
            logger.error("Not connected to WebSocket")
            return
            
        test_message = {
            "type": "outbound_message",
            "topic": "/response_text",
            "msg_type": "std_msgs/String",
            "data": {"data": f"Test message from {self.agent_id}"}
        }
        
        await self.websocket.send(json.dumps(test_message))
        logger.info("📤 Sent test message to ROS")
        
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket disconnected")

async def test_websocket_bridge():
    """Test WebSocket bridge functionality"""
    agent = SimpleWebSocketAgent()
    
    # Connect to bridge
    success = await agent.connect()
    if not success:
        logger.error("Failed to connect to bridge")
        return False
        
    # Send a test message
    await agent.send_test_message()
    
    # Listen for messages
    await agent.listen_for_messages(duration=10)
    
    # Disconnect
    await agent.disconnect()
    
    logger.info(f"🏁 Test completed. Total messages received: {agent.message_count}")
    return True

def main():
    """Main entry point"""
    logger.info("🧪 Starting WebSocket bridge test")
    logger.info("Make sure the ROS AI Bridge is running with WebSocket enabled!")
    logger.info("Example: ros2 run by_your_command ros_ai_bridge --ros-args -p websocket_server.enabled:=true")
    
    try:
        result = asyncio.run(test_websocket_bridge())
        if result:
            logger.info("✅ Test passed!")
        else:
            logger.error("❌ Test failed!")
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test error: {e}")

if __name__ == '__main__':
    main()