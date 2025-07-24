#!/usr/bin/env python3
"""
OpenAI Realtime Agent Main Entry Point

Standalone executable for running the OpenAI Realtime agent with
ROS AI Bridge integration.

Author: Karim Virani  
Version: 1.0
Date: July 2025
"""

import asyncio
import logging
import os
import yaml
from typing import Dict, Any

from agents.oai_realtime_agent import OpenAIRealtimeAgent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load agent configuration from file and environment"""
    
    # Default configuration
    config = {
        'openai_api_key': '',
        'model': 'gpt-4o-realtime-preview',
        'voice': 'alloy',
        'session_pause_timeout': 10.0,
        'session_max_duration': 120.0,
        'session_max_tokens': 50000,
        'session_max_cost': 5.00,
        'max_context_tokens': 2000,
        'max_context_age': 3600,
        'vad_threshold': 0.5,
        'vad_prefix_padding': 300,
        'vad_silence_duration': 200,
        'log_level': logging.INFO,
        'system_prompt': """You are a helpful robotic assistant. You can control robot movements, 
answer questions, and engage in natural conversation. Be concise but friendly.
Respond naturally to the user's speech and provide helpful information or assistance."""
    }
    
    # Load from config file if specified
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if 'openai_realtime_agent' in file_config:
                    config.update(file_config['openai_realtime_agent'])
                else:
                    config.update(file_config)
            print(f"✅ Loaded configuration from {config_path}")
        except Exception as e:
            print(f"⚠️ Error loading config file {config_path}: {e}")
    
    # Override with environment variables
    env_mappings = {
        'OPENAI_API_KEY': 'openai_api_key',
        'OPENAI_MODEL': 'model', 
        'OPENAI_VOICE': 'voice',
        'PAUSE_TIMEOUT': 'session_pause_timeout',
        'MAX_DURATION': 'session_max_duration'
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        print(f"🔍 Checking {env_var}: {'***' if env_var == 'OPENAI_API_KEY' and value else value or 'NOT SET'}")
        if value:
            # Convert numeric values
            if config_key.endswith('_timeout') or config_key.endswith('_duration'):
                try:
                    config[config_key] = float(value)
                except ValueError:
                    print(f"⚠️ Invalid numeric value for {env_var}: {value}")
            else:
                config[config_key] = value
    
    # Validate required configuration
    if not config.get('openai_api_key'):
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or add to config file.")
    
    return config


def setup_logging(level: int = logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Set specific logger levels
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


async def run_standalone_agent(config: Dict[str, Any]):
    """Run agent as standalone process connecting to bridge queues"""
    
    try:
        # Create and initialize agent (connects to existing bridge)
        agent = OpenAIRealtimeAgent(config)
        await agent.initialize()
        
        print("🚀 OpenAI Realtime Agent started!")
        print(f"📡 Model: {config.get('model', 'unknown')}")
        print(f"⏱️  Pause timeout: {config.get('session_pause_timeout', 10)}s")
        print(f"🎙️  Voice: {config.get('voice', 'alloy')}")
        print("🔊 Listening for speech...")
        
        # Run agent main loop
        await agent.run()
        
    except KeyboardInterrupt:
        print("\n⌨️ Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("Agent error")
    finally:
        # Cleanup
        try:
            if 'agent' in locals():
                await agent.stop()
        except:
            pass
        print("👋 Goodbye!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI Realtime Agent")
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--pause-timeout', '-p',
        type=float,
        help='Session pause timeout in seconds'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (overrides environment variable)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override with command line arguments
        if args.pause_timeout:
            config['session_pause_timeout'] = args.pause_timeout
            
        if args.api_key:
            config['openai_api_key'] = args.api_key
            
        if args.verbose:
            config['log_level'] = logging.DEBUG
            
        # Setup logging
        setup_logging(config['log_level'])
        
        print("🤖 Starting OpenAI Realtime Agent...")
        
        # Run agent
        asyncio.run(run_standalone_agent(config))
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Set OPENAI_API_KEY environment variable or use --config option")
        exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.exception("Fatal error")
        exit(1)


if __name__ == '__main__':
    main()