#!/usr/bin/env python3
"""
Gemini Live Agent Main Entry Point

Standalone executable for running the Gemini Live multimodal agent with
ROS AI Bridge integration via WebSockets.

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

# Import the new Gemini agent with Pipecat pipeline
from agents.gemini_live.gemini_live_agent_new import GeminiLiveAgent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load agent configuration from file and environment"""
    
    # Default configuration
    config = {
        'agent_id': 'gemini_visual',
        'agent_type': 'multimodal',
        'api_key': '',
        'model': 'gemini-2.0-flash-exp',
        'bridge_connection': {
            'type': 'websocket',
            'host': 'localhost',
            'port': 8765,
            'reconnect_interval': 5.0,
            'max_reconnect_attempts': 10
        },
        'modalities': ['audio', 'vision', 'text'],
        'audio': {
            'input_sample_rate': 16000,
            'output_sample_rate': 24000,
            'voice': 'default'
        },
        'video': {
            'enabled': True,
            'fps': 1.0,
            'max_fps': 10.0,
            'min_fps': 0.1,
            'resolution': '480p',
            'dynamic_fps': True
        },
        'session_timeout': 300.0,
        'max_context_tokens': 4000,
        'log_level': logging.INFO
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
                    # Merge nested configs properly
                    for key, value in agent_config.items():
                        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                            config[key].update(value)
                        else:
                            config[key] = value
                else:
                    config.update(file_config)
            print(f"✅ Loaded configuration from {config_path}")
        except Exception as e:
            print(f"⚠️ Error loading config file {config_path}: {e}")
    
    # Override with environment variables
    env_mappings = {
        'GEMINI_API_KEY': 'api_key',
        'AGENT_TYPE': 'agent_type',
        'PAUSE_TIMEOUT': 'session_timeout'
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        print(f"🔍 Checking {env_var}: {'***' if env_var == 'GEMINI_API_KEY' and value else value or 'NOT SET'}")
        if value:
            # Convert numeric values
            if config_key.endswith('_timeout'):
                try:
                    config[config_key] = float(value)
                except ValueError:
                    print(f"⚠️ Invalid numeric value for {env_var}: {value}")
            else:
                config[config_key] = value
    
    # Validate required configuration
    if not config.get('api_key'):
        raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable or add to config file.")
    
    # Debug: Show final config
    print(f"📋 Final config - agent_type: {config.get('agent_type', 'NOT SET')}, model: {config.get('model', 'NOT SET')}")
    
    return config


class AgentFormatter(logging.Formatter):
    """Custom formatter for agent logs"""
    def __init__(self, agent_type='gemini'):
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
    agent_id = config.get('agent_id', 'gemini_visual') if config else 'gemini_visual'
    agent_type = 'visual' if 'visual' in agent_id.lower() else 'gemini'
    
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
    logging.getLogger('pipecat').setLevel(logging.INFO)


async def run_standalone_agent(config: Dict[str, Any]):
    """Run agent as standalone process connecting to bridge via WebSocket"""
    
    try:
        # Create and initialize agent (connects to bridge via WebSocket)
        agent = GeminiLiveAgent(config)
        await agent.initialize()
        
        print("🚀 Gemini Live Agent started!")
        print(f"📡 Model: {config.get('model', 'unknown')}")
        print(f"🎯 Agent Type: {config.get('agent_type', 'multimodal')}")
        print(f"📷 Video: {'Enabled' if config.get('video', {}).get('enabled') else 'Disabled'}")
        print(f"🎙️  Audio: {config.get('audio', {}).get('input_sample_rate', 16000)}Hz input")
        print("🔊 Listening for multimodal input...")
        
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
                await agent.shutdown()
        except:
            pass
        print("👋 Goodbye!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemini Live Multimodal Agent")
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
        help='Session timeout in seconds'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Gemini API key (overrides environment variable)'
    )
    parser.add_argument(
        '--agent-type',
        type=str,
        choices=['multimodal', 'conversation', 'command', 'visual'],
        help='Agent type (overrides config file)'
    )
    parser.add_argument(
        '--prompt-id',
        type=str,
        help='Prompt ID from prompts.yaml (overrides config file)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override with command line arguments
        if args.pause_timeout:
            config['session_timeout'] = args.pause_timeout
            
        if args.api_key:
            config['api_key'] = args.api_key
            
        if args.agent_type:
            config['agent_type'] = args.agent_type
            
        if args.prompt_id:
            config['prompt_id'] = args.prompt_id
            
        if args.verbose:
            config['log_level'] = logging.DEBUG
            
        # Setup logging with agent type
        setup_logging(config['log_level'], config)
        
        print("🤖 Starting Gemini Live Agent...")
        
        # Run agent
        asyncio.run(run_standalone_agent(config))
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Set GEMINI_API_KEY environment variable or use --config option")
        exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.exception("Fatal error")
        exit(1)


if __name__ == '__main__':
    main()