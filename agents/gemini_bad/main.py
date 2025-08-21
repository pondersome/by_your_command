#!/usr/bin/env python3
"""
Gemini Live Agent Main Entry Point

Standalone executable for running the Gemini Live agent with
ROS AI Bridge integration.

Author: Karim Virani  
Version: 1.0
Date: August 2025
"""

import asyncio
import logging
import os
import yaml
from typing import Dict, Any
from datetime import datetime

from agents.gemini_live.gemini_live_agent import GeminiLiveAgent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load agent configuration from file and environment"""
    
    # Default configuration
    config = {
        'gemini_api_key': '',
        'model': 'models/gemini-2.0-flash-exp',  # Gemini Live model
        'voice': 'Kore',  # Gemini voice options: Aoede, Charon, Fenrir, Kore, Puck
        'proactive_audio': False,  # Let model decide when to speak
        'enable_video': False,  # Enable video support (2-minute limit)
        'video_topic': '/camera/image_compressed',
        'video_frame_interval': 1.0,  # Send frame every second
        'session_pause_timeout': 10.0,
        'max_context_tokens': 2000,
        'conversation_timeout': 600.0,  # 10 minutes
        'log_level': logging.INFO,
        'agent_id': 'gemini_live',
        'audio_out_topic': 'audio_out',
        'transcript_topic': 'llm_transcript',
        'command_detected_topic': 'command_detected',
        'interruption_signal_topic': 'interruption_signal',
        'system_prompt': """You are a helpful robotic assistant with vision capabilities. 
You can see through the robot's camera, control robot movements, answer questions, 
and engage in natural conversation. Be concise but friendly.
Respond naturally to the user's speech and provide helpful information or assistance.
If video is enabled, you can describe what you see and help with visual tasks."""
    }
    
    # Load from config file if specified
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Look for agent-specific configuration
                if 'gemini_live_agent' in file_config:
                    agent_config = file_config['gemini_live_agent']
                    print(f"✅ Found gemini_live_agent config with keys: {list(agent_config.keys())}")
                    if 'voice' in agent_config:
                        print(f"✅ Voice setting in YAML: {agent_config['voice']}")
                    if 'enable_video' in agent_config:
                        print(f"📹 Video enabled: {agent_config['enable_video']}")
                    config.update(agent_config)
                else:
                    config.update(file_config)
            print(f"✅ Loaded configuration from {config_path}")
        except Exception as e:
            print(f"⚠️ Error loading config file {config_path}: {e}")
    
    # Override with environment variables
    env_mappings = {
        'GEMINI_API_KEY': 'gemini_api_key',
        'GOOGLE_API_KEY': 'gemini_api_key',  # Alternative env var name
        'GEMINI_MODEL': 'model', 
        'GEMINI_VOICE': 'voice',
        'GEMINI_PROACTIVE_AUDIO': 'proactive_audio',
        'GEMINI_ENABLE_VIDEO': 'enable_video',
        'PAUSE_TIMEOUT': 'session_pause_timeout',
        'VIDEO_TOPIC': 'video_topic',
        'VIDEO_FRAME_INTERVAL': 'video_frame_interval'
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        if 'API_KEY' in env_var:
            print(f"🔍 Checking {env_var}: {'***' if value else 'NOT SET'}")
        else:
            print(f"🔍 Checking {env_var}: {value or 'NOT SET'}")
            
        if value:
            # Convert boolean values
            if config_key in ['proactive_audio', 'enable_video']:
                config[config_key] = value.lower() in ['true', '1', 'yes', 'on']
            # Convert numeric values
            elif config_key in ['session_pause_timeout', 'video_frame_interval']:
                try:
                    config[config_key] = float(value)
                except ValueError:
                    print(f"⚠️ Invalid numeric value for {env_var}: {value}")
            else:
                config[config_key] = value
    
    # Validate API key
    if not config.get('gemini_api_key'):
        print("⚠️ WARNING: No Gemini API key found!")
        print("  Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        print("  Or add 'gemini_api_key' to your config file")
    
    # Show video mode warning if enabled
    if config.get('enable_video'):
        print("📹 VIDEO MODE ENABLED - Session limit is 2 MINUTES!")
        print(f"   Video topic: {config.get('video_topic')}")
        print(f"   Frame interval: {config.get('video_frame_interval')}s")
    else:
        print("🎤 Audio-only mode - Session limit is 15 minutes")
    
    # Show final configuration (without sensitive data)
    print("\n📋 Final configuration:")
    for key, value in config.items():
        if 'key' in key.lower():
            print(f"  {key}: ***")
        else:
            print(f"  {key}: {value}")
    
    return config


async def main():
    """Main entry point for Gemini Live agent"""
    
    print("\n" + "="*60)
    print("🚀 Gemini Live Agent Starting")
    print("="*60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Gemini Live Agent')
    parser.add_argument('--config', type=str, help='Path to config file',
                       default='config/gemini_live_agent.yaml')
    parser.add_argument('--standalone', action='store_true',
                       help='Run in standalone mode without ROS AI Bridge')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--video', action='store_true',
                       help='Enable video support (2-minute session limit)')
    parser.add_argument('--proactive', action='store_true',
                       help='Enable proactive audio (model decides when to speak)')
    parser.add_argument('--pause-timeout', type=float, default=10.0,
                       help='Session pause timeout in seconds')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.pause_timeout:
        config['session_pause_timeout'] = args.pause_timeout
        print(f"⏱️ Pause timeout set to: {args.pause_timeout}s")
    
    if args.debug:
        config['log_level'] = logging.DEBUG
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=config.get('log_level', logging.INFO))
    
    if args.video:
        config['enable_video'] = True
        print("📹 Video mode enabled via command line")
    
    if args.proactive:
        config['proactive_audio'] = True
        print("🎙️ Proactive audio enabled - model will decide when to speak")
    
    # Create agent (it creates its own bridge interface internally)
    agent = GeminiLiveAgent(config)
    
    # Initialize agent
    await agent.initialize()
    print("✅ Agent initialized")
    
    # Run agent
    try:
        print("\n" + "="*60)
        print("🎮 Gemini Live Agent Ready")
        print("="*60)
        
        if config.get('enable_video'):
            print("📹 Video: ENABLED (2-minute session limit)")
        else:
            print("🎤 Video: DISABLED (15-minute session limit)")
            
        if config.get('proactive_audio'):
            print("🎙️ Proactive Audio: ENABLED (model decides when to speak)")
        else:
            print("🔕 Proactive Audio: DISABLED (always responds)")
            
        print(f"🤖 Model: {config.get('model')}")
        print(f"🎵 Voice: {config.get('voice')}")
        print("\nPress Ctrl+C to stop\n")
        
        await agent.run()
        
    except KeyboardInterrupt:
        print("\n\n⛔ Keyboard interrupt received")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Shutting down...")
        await agent.cleanup()
        print("✅ Gemini Live Agent stopped")
        print(f"⏰ Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print final metrics
        metrics = agent.get_metrics()
        print("\n📊 Final Metrics:")
        print(f"  Messages processed: {metrics.get('messages_processed', 0)}")
        print(f"  Audio chunks sent: {metrics.get('audio_chunks_sent', 0)}")
        print(f"  Text messages sent: {metrics.get('text_messages_sent', 0)}")
        print(f"  Video frames sent: {metrics.get('video_frames_sent', 0)}")
        print(f"  Responses received: {metrics.get('responses_received', 0)}")
        print(f"  Sessions created: {metrics.get('sessions_created', 0)}")
        print(f"  Sessions cycled (pause): {metrics.get('sessions_cycled_on_pause', 0)}")
        print(f"  Sessions cycled (limits): {metrics.get('sessions_cycled_on_limits', 0)}")
        print(f"  Interruptions: {metrics.get('interruptions', 0)}")
        print(f"  Errors: {metrics.get('errors', 0)}")


if __name__ == "__main__":
    asyncio.run(main())