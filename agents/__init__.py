"""
Agents Package

LLM integration agents with asyncio concurrency for the by_your_command package.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

# OpenAI Realtime API agent
from .oai_realtime.oai_realtime_agent import OpenAIRealtimeAgent
from .oai_realtime.serializers import OpenAIRealtimeSerializer

# Google Gemini Live agent (placeholder)
# from .gemini_live import GeminiLiveAgent

# LangGraph orchestration (placeholder)
# from .graph import AgentGraph

__version__ = "1.0"
__author__ = "Karim Virani"

__all__ = [
    "OpenAIRealtimeAgent",
    "OpenAIRealtimeSerializer"
    # "GeminiLiveAgent",
    # "AgentGraph"
]