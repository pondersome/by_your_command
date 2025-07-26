#!/usr/bin/env python3
"""
Standalone Mode Demo for OpenAI Realtime Agent

Simple demonstration of how standalone mode works and how to inject test data.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import asyncio
import logging
import os
import sys

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import debug interface components
from debug_interface import create_test_audio_sine_wave, create_test_audio_noise


def explain_standalone_mode():
    """Explain how standalone mode works"""
    print("🤖 OpenAI Realtime Agent - Standalone Mode")
    print("=" * 50)
    print()
    print("How Standalone Mode Works:")
    print("-" * 25)
    print("1. 🔗 Agent tries to connect to ROS bridge via WebSocket")
    print("2. ❌ Connection fails (no bridge running)")
    print("3. 🔧 Agent enables debug interface for testing")
    print("4. 📨 Debug interface allows direct message injection")
    print("5. 🎯 Messages are processed same as bridge messages")
    print("6. 🤖 Agent can create OpenAI sessions and process responses")
    print()
    print("Key Benefits:")
    print("• Test agent without full ROS system")
    print("• Inject synthetic audio data")
    print("• Test prompt switching")
    print("• Debug session management")
    print("• Validate OpenAI API integration")
    print()


def demo_audio_data_creation():
    """Demonstrate creating test audio data"""
    print("🎵 Creating Test Audio Data")
    print("=" * 30)
    
    # Generate different types of test audio
    print("Available audio generation functions:")
    print()
    
    print("1. create_test_audio_sine_wave(frequency, duration, sample_rate=16000)")
    sine_440 = create_test_audio_sine_wave(440, 1.0)  # 1 second A4 note
    print(f"   Example: 440Hz for 1s = {len(sine_440)} samples")
    print(f"   Data type: List[int] with 16-bit PCM values")
    print(f"   Range: {min(sine_440)} to {max(sine_440)}")
    print()
    
    print("2. create_test_audio_noise(duration, sample_rate=16000)")
    noise = create_test_audio_noise(0.5)  # 0.5 seconds white noise
    print(f"   Example: 0.5s noise = {len(noise)} samples")
    print(f"   Data type: List[int] with random 16-bit PCM values")
    print()
    
    print("3. load_wav_file(file_path, target_sample_rate=16000) [requires scipy]")
    print("   Loads WAV file and converts to 16kHz mono PCM")
    print("   Returns List[int] compatible with debug interface")
    print()


async def demo_debug_injection():
    """Demonstrate debug message injection concept"""
    print("🔧 Debug Message Injection")
    print("=" * 30)
    
    print("How to inject audio data into standalone agent:")
    print()
    
    print("# 1. Create agent in standalone mode")
    print("agent = OpenAIRealtimeAgent(config)")
    print("await agent.initialize()  # Bridge connection fails")
    print("# → agent.debug_interface is created")
    print()
    
    print("# 2. Generate test audio")
    print("audio_data = create_test_audio_sine_wave(440, 2.0)")
    print()
    
    print("# 3. Inject audio into agent")
    print("success = await agent.debug_inject_audio(")
    print("    audio_data,")
    print("    utterance_id='test_speech',")
    print("    confidence=0.95,")
    print("    is_utterance_end=True")
    print(")")
    print()
    
    print("# 4. Agent processes as if from ROS bridge")
    print("# → Creates OpenAI session if needed")
    print("# → Serializes audio to base64 PCM")
    print("# → Sends to OpenAI Realtime API")
    print("# → Processes responses")
    print()
    
    print("Available injection methods:")
    print("• await agent.debug_inject_audio(audio_data, ...)")
    print("• await agent.debug_inject_text(text)")
    print("• agent.get_debug_stats()")
    print("• agent.is_standalone_mode()")
    print()


def demo_testing_scenarios():
    """Show different testing scenarios"""
    print("🧪 Testing Scenarios")
    print("=" * 20)
    
    scenarios = [
        {
            "name": "Basic Audio Processing",
            "description": "Test agent's ability to handle audio input",
            "code": """
# Generate 2 seconds of 440Hz tone
audio = create_test_audio_sine_wave(440, 2.0)
await agent.debug_inject_audio(audio, is_utterance_end=True)
# → Should create session and send to OpenAI
            """
        },
        {
            "name": "Prompt Switching",
            "description": "Test runtime prompt changes",
            "code": """
# Start with adult prompt (Barney)
await agent.switch_system_prompt(context_updates={'user_age': 30})
await agent.debug_inject_audio(test_audio)

# Switch to child prompt (friendly assistant)  
await agent.switch_system_prompt(context_updates={'user_age': 7})
await agent.debug_inject_audio(test_audio)
# → Should use different personality
            """
        },
        {
            "name": "Session Cycling",
            "description": "Test pause-based session management",
            "code": """
# Send audio, then wait for pause
await agent.debug_inject_audio(audio1, is_utterance_end=True)
await asyncio.sleep(12)  # Longer than pause timeout
await agent.debug_inject_audio(audio2, is_utterance_end=True)
# → Should cycle session between messages
            """
        },
        {
            "name": "Stress Testing",
            "description": "Test with rapid message injection",
            "code": """
# Inject multiple messages quickly
for i in range(10):
    audio = create_test_audio_sine_wave(440 + i*100, 0.5)
    await agent.debug_inject_audio(audio, f"test_{i}")
    await asyncio.sleep(0.1)
# → Test queue handling and rate limiting
            """
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Code example:")
        for line in scenario['code'].strip().split('\n'):
            print(f"   {line}")
        print()


def show_architecture():
    """Show the architecture of standalone mode"""
    print("🏗️  Standalone Mode Architecture")
    print("=" * 35)
    print()
    print("Normal Mode (with ROS bridge):")
    print("┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐")
    print("│ ROS Topics      │───►│ Bridge       │───►│ Agent           │")
    print("│ /voice_chunks   │    │ WebSocket    │    │ Session Manager │")
    print("│ /text_input     │◄───│ Server       │◄───│ OpenAI API      │")
    print("└─────────────────┘    └──────────────┘    └─────────────────┘")
    print()
    print("Standalone Mode (debug interface):")
    print("┌─────────────────┐                       ┌─────────────────┐")
    print("│ Test Scripts    │──────────────────────►│ Agent           │")
    print("│ debug_inject_*  │                       │ Debug Interface │")
    print("│ Audio/Text Data │◄──────────────────────│ Session Manager │")
    print("└─────────────────┘                       │ OpenAI API      │")
    print("                                          └─────────────────┘")
    print()
    print("Key Differences:")
    print("• No WebSocket bridge connection")
    print("• Direct message injection via debug methods")
    print("• Same internal processing pipeline")
    print("• Same OpenAI API integration")
    print("• Same session management and prompt system")
    print()


if __name__ == "__main__":
    print("📚 OpenAI Realtime Agent - Standalone Mode Guide")
    print("=" * 60)
    print()
    
    explain_standalone_mode()
    demo_audio_data_creation()
    
    # Run async demo
    asyncio.run(demo_debug_injection())
    
    demo_testing_scenarios()
    show_architecture()
    
    print("🚀 Next Steps:")
    print("=" * 15)
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run: python3 test_standalone_mode.py")
    print("3. Or create custom test scripts using debug interface")
    print("4. Check agent logs for session creation and API calls")
    print()
    print("💡 For real usage:")
    print("• Start ROS bridge: ros2 launch by_your_command oai_realtime.launch.py")
    print("• Agent will connect to bridge instead of using debug interface")
    print("• Audio comes from /voice_chunks topic via VAD processing")