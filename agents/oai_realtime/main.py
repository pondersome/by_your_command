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
from datetime import datetime

from agents.oai_realtime.oai_realtime_agent import OpenAIRealtimeAgent


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
        'conversation_timeout': 600.0,  # 10 minutes
        'vad_threshold': 0.5,
        'vad_prefix_padding': 300,
        'vad_silence_duration': 200,
        'vad_create_response': False,  # Use manual triggering for reliable responses
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
                # Look for agent-specific configuration
                if 'openai_realtime_agent' in file_config:
                    agent_config = file_config['openai_realtime_agent']
                    print(f"✅ Found openai_realtime_agent config with keys: {list(agent_config.keys())}")
                    if 'voice' in agent_config:
                        print(f"✅ Voice setting in YAML: {agent_config['voice']}")
                    config.update(agent_config)
                elif 'openai_command_agent' in file_config:
                    agent_config = file_config['openai_command_agent']
                    print(f"✅ Found openai_command_agent config with keys: {list(agent_config.keys())}")
                    if 'voice' in agent_config:
                        print(f"✅ Voice setting in YAML: {agent_config['voice']}")
                    config.update(agent_config)
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
        'MAX_DURATION': 'session_max_duration',
        'VAD_THRESHOLD': 'vad_threshold',
        'VAD_PREFIX_PADDING': 'vad_prefix_padding',
        'VAD_SILENCE_DURATION': 'vad_silence_duration'
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        print(f"🔍 Checking {env_var}: {'***' if env_var == 'OPENAI_API_KEY' and value else value or 'NOT SET'}")
        if value:
            # Convert numeric values
            if config_key.endswith('_timeout') or config_key.endswith('_duration') or config_key.startswith('vad_'):
                try:
                    if config_key == 'vad_threshold':
                        config[config_key] = float(value)
                    elif config_key.endswith('_padding') or config_key.endswith('_duration'):
                        config[config_key] = int(value)
                    else:
                        config[config_key] = float(value)
                except ValueError:
                    print(f"⚠️ Invalid numeric value for {env_var}: {value}")
            else:
                config[config_key] = value
    
    # Validate required configuration
    if not config.get('openai_api_key'):
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or add to config file.")
    
    # Debug: Show final config
    print(f"📋 Final config - voice: {config.get('voice', 'NOT SET')}, model: {config.get('model', 'NOT SET')}")
    
    return config


class AgentFormatter(logging.Formatter):
    """Custom formatter for agent logs"""
    def __init__(self, agent_type='conv'):
        self.agent_type = agent_type
        super().__init__()
        
    def format(self, record):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # Skip the module name prefix to reduce clutter
        msg = record.getMessage()
        return f"[{timestamp}] [agent:{self.agent_type}] {msg}"

def setup_logging(level: int = logging.INFO, config: Dict[str, Any] = None):
    """Setup logging configuration"""
    # Determine agent type from config
    agent_id = config.get('agent_id', 'openai_realtime') if config else 'openai_realtime'
    agent_type = 'cmd' if 'command' in agent_id.lower() else 'conv'
    
    # Create console handler with custom formatter
    handler = logging.StreamHandler()
    handler.setFormatter(AgentFormatter(agent_type))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
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
    parser.add_argument(
        '--prompt-id',
        type=str,
        help='Named prompt ID from prompts.yaml (overrides config file)'
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
            
        if args.prompt_id:
            config['prompt_id'] = args.prompt_id
            
        if args.verbose:
            config['log_level'] = logging.DEBUG
            
        # Setup logging with agent type
        setup_logging(config['log_level'], config)
        
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