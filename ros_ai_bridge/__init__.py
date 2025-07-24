"""
ROS AI Bridge Package

A minimal data transport layer that shuttles messages between ROS2's callback-based 
concurrency model and agents using asyncio-based concurrency.

Author: Karim Virani
Version: 3.0
Date: July 2025
"""

from .ros_ai_bridge import (
    MessageEnvelope,
    MessageQueues, 
    AgentInterface,
    BridgeReconfigurer,
    ROSAIBridge,
    run_bridge_async
)

__version__ = "3.0"
__author__ = "Karim Virani"

__all__ = [
    "MessageEnvelope",
    "MessageQueues",
    "AgentInterface", 
    "BridgeReconfigurer",
    "ROSAIBridge",
    "run_bridge_async"
]