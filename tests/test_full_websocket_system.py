#!/usr/bin/env python3
"""
End-to-end test of WebSocket-based bridge-agent architecture

Tests the complete system:
1. ROS AI Bridge with WebSocket server
2. OpenAI Realtime Agent with WebSocket client  
3. AudioDataUtterance message handling and transcoding
4. Bidirectional communication

Author: Karim Virani
Version: 1.0  
Date: July 2025
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.insert(0, '/home/karim/ros2_ws/src/by_your_command')

from agents.oai_realtime.websocket_bridge import WebSocketBridgeInterface
from agents.oai_realtime.serializers import OpenAIRealtimeSerializer

class MockAudioDataUtterance:
    """Mock AudioDataUtterance for testing"""
    def __init__(self):
        self.audio_data = [100, 200, 300, 400, 500] * 100  # Mock PCM data
        self.utterance_id = "test_utterance_001"
        self.start_time = time.time()
        self.confidence = 0.95
        self.chunk_sequence = 1
        self.is_utterance_end = False

class EndToEndTester:
    """Complete system tester"""
    
    def __init__(self):
        self.agent_config = {
            'agent_id': 'test_openai_realtime',
            'bridge_connection': {
                'host': 'localhost',
                'port': 8765,
                'reconnect_interval': 2.0,
                'max_reconnect_attempts': 3
            },
            'openai_api_key': 'sk-test-key',
            'model': 'gpt-4o-realtime-preview'
        }
        
        self.bridge_interface = None
        self.serializer = OpenAIRealtimeSerializer()
        
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection to bridge"""
        logger.info("🧪 Testing WebSocket connection...")
        
        try:
            self.bridge_interface = WebSocketBridgeInterface(self.agent_config)
            success = await self.bridge_interface.connect()
            
            if success:
                logger.info("✅ WebSocket connection successful")
                return True
            else:
                logger.error("❌ WebSocket connection failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ WebSocket connection error: {e}")
            return False
            
    async def test_audio_data_utterance_processing(self) -> bool:
        """Test AudioDataUtterance transcoding"""
        logger.info("🧪 Testing AudioDataUtterance processing...")
        
        try:
            # Create mock envelope
            from agents.oai_realtime.websocket_bridge import WebSocketMessageEnvelope
            
            mock_envelope_data = {
                "msg_type": "topic",
                "topic_name": "/prompt_voice",
                "ros_msg_type": "by_your_command/AudioDataUtterance",
                "timestamp": time.time(),
                "metadata": {"source_node": "test"},
                "data": {
                    "audio_data": [100, 200, 300, 400, 500] * 100,
                    "utterance_id": "test_utterance_001", 
                    "start_time": time.time(),
                    "confidence": 0.95,
                    "chunk_sequence": 1,
                    "is_utterance_end": False
                }
            }
            
            envelope = WebSocketMessageEnvelope(mock_envelope_data)
            
            # Test serialization
            api_msg = await self.serializer.safe_serialize(envelope)
            
            if api_msg and api_msg.get("type") == "input_audio_buffer.append":
                logger.info("✅ AudioDataUtterance serialization successful")
                
                # Test metadata extraction
                metadata = self.serializer.get_utterance_metadata()
                if metadata.get("utterance_id") == "test_utterance_001":
                    logger.info("✅ Metadata extraction successful")
                    return True
                else:
                    logger.error(f"❌ Metadata extraction failed: {metadata}")
                    return False
            else:
                logger.error(f"❌ AudioDataUtterance serialization failed: {api_msg}")
                return False
                
        except Exception as e:
            logger.error(f"❌ AudioDataUtterance processing error: {e}")
            return False
            
    async def test_bidirectional_communication(self) -> bool:
        """Test sending message back to bridge"""
        logger.info("🧪 Testing bidirectional communication...")
        
        if not self.bridge_interface or not self.bridge_interface.is_connected():
            logger.error("❌ No bridge connection for bidirectional test")
            return False
            
        try:
            # Send test transcript back to bridge
            transcript_data = {"data": "Test transcript from agent"}
            success = await self.bridge_interface.put_outbound_message(
                "/response_text",
                transcript_data,
                "std_msgs/String"
            )
            
            if success:
                logger.info("✅ Bidirectional communication successful")
                return True
            else:
                logger.error("❌ Failed to send message to bridge")
                return False
                
        except Exception as e:
            logger.error(f"❌ Bidirectional communication error: {e}")
            return False
            
    async def test_connection_resilience(self) -> bool:
        """Test connection metrics and status"""
        logger.info("🧪 Testing connection resilience...")
        
        if not self.bridge_interface:
            logger.error("❌ No bridge interface for resilience test")
            return False
            
        try:
            # Get metrics
            metrics = self.bridge_interface.get_metrics()
            status = self.bridge_interface.get_status_summary()
            
            logger.info(f"📊 Bridge metrics: {metrics}")
            logger.info(f"📊 Bridge status: {status}")
            
            # Check if metrics are reasonable
            if isinstance(metrics, dict) and "connected" in metrics:
                logger.info("✅ Connection resilience test passed")
                return True
            else:
                logger.error(f"❌ Invalid metrics: {metrics}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection resilience test error: {e}")
            return False
            
    async def cleanup(self):
        """Clean up test resources"""
        if self.bridge_interface:
            await self.bridge_interface.close()
            logger.info("🧹 Test cleanup completed")
            
    async def run_full_test(self) -> bool:
        """Run complete end-to-end test suite"""
        logger.info("🚀 Starting end-to-end WebSocket system test")
        logger.info("⚠️  Make sure the ROS AI Bridge is running with WebSocket enabled!")
        logger.info("   Example: ros2 run by_your_command ros_ai_bridge --ros-args -p websocket_server.enabled:=true")
        
        tests = [
            ("WebSocket Connection", self.test_websocket_connection),
            ("AudioDataUtterance Processing", self.test_audio_data_utterance_processing),
            ("Bidirectional Communication", self.test_bidirectional_communication),
            ("Connection Resilience", self.test_connection_resilience)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            try:
                result = await test_func()
                results.append((test_name, result))
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results.append((test_name, False))
                
        # Summary
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        logger.info(f"\n🏁 Test Results Summary:")
        logger.info(f"📊 {passed}/{total} tests passed")
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"   {status}: {test_name}")
            
        # Cleanup
        await self.cleanup()
        
        return passed == total

async def main():
    """Main test entry point"""
    tester = EndToEndTester()
    
    try:
        success = await tester.run_full_test()
        if success:
            logger.info("🎉 All tests passed! WebSocket system is working correctly.")
            return 0
        else:
            logger.error("💥 Some tests failed. Check the logs above.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        await tester.cleanup()
        return 130
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        await tester.cleanup()
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)