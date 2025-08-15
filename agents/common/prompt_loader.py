"""
Named Prompt System for OpenAI Realtime Agent

Loads and manages named system prompts with A/B testing, conditional selection,
and user prompt prefix injection capabilities.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import os
import yaml
import random
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class PromptInfo:
    """Information about a named prompt"""
    name: str
    description: str
    version: str
    tested_with: List[str]
    system_prompt: str
    parent: Optional[str] = None


@dataclass
class UserPrefixInfo:
    """Information about user prompt prefixes"""
    description: str
    prefix: str


class PromptLoader:
    """Load and select named system prompts with A/B testing"""
    
    def __init__(self, prompts_file: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Default prompts file path
        if prompts_file is None:
            # Try to find prompts.yaml in multiple possible locations
            possible_paths = [
                # Source directory (development) - from agents/common to config
                os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config', 'prompts.yaml'),
                # Install directory (installed package)
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', '..', 'src', 'by_your_command', 'config', 'prompts.yaml'),
                # ROS workspace source
                '/home/karim/ros2_ws/src/by_your_command/config/prompts.yaml',
                # Current working directory
                os.path.join(os.getcwd(), 'config', 'prompts.yaml'),
                # Relative to common directory
                os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'prompts.yaml')
            ]
            
            prompts_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    prompts_file = path
                    break
            
            if prompts_file is None:
                self.logger.warning(f"Could not find prompts.yaml in any of these locations:")
                for path in possible_paths:
                    self.logger.warning(f"  - {path}")
                # Use the first path as fallback for error reporting
                prompts_file = possible_paths[0]
            
        self.prompts_file = prompts_file
        self.prompts: Dict[str, PromptInfo] = {}
        self.user_prefixes: Dict[str, UserPrefixInfo] = {}
        self.macros: Dict[str, str] = {}
        self.selection_rules: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Load prompts
        self._load_prompts()
        
    def _load_prompts(self):
        """Load prompts from YAML file"""
        try:
            if not os.path.exists(self.prompts_file):
                self.logger.error(f"Prompts file not found: {self.prompts_file}")
                self._create_fallback_prompt()
                return
                
            with open(self.prompts_file, 'r') as f:
                data = yaml.safe_load(f)
                
            # Load macros first (before prompts so they can be expanded)
            self.macros = data.get('macros', {})
            self.logger.info(f"Loaded {len(self.macros)} macros: {list(self.macros.keys())}")
                
            # Load named prompts
            for prompt_id, prompt_data in data.get('prompts', {}).items():
                # Expand macros in system_prompt
                system_prompt = prompt_data.get('system_prompt', '')
                expanded_prompt = self._expand_macros(system_prompt)
                
                # Log macro expansion for debugging
                macro_count = system_prompt.count('{{')
                if macro_count > 0:
                    self.logger.info(f"Expanded {macro_count} macros in prompt '{prompt_id}'")
                
                self.prompts[prompt_id] = PromptInfo(
                    name=prompt_data.get('name', prompt_id),
                    description=prompt_data.get('description', ''),
                    version=prompt_data.get('version', '1.0'),
                    tested_with=prompt_data.get('tested_with', []),
                    system_prompt=expanded_prompt,
                    parent=prompt_data.get('parent')
                )
                
            # Load user prompt prefixes
            for prefix_id, prefix_data in data.get('user_prompt_prefixes', {}).items():
                self.user_prefixes[prefix_id] = UserPrefixInfo(
                    description=prefix_data.get('description', ''),
                    prefix=prefix_data.get('prefix', '')
                )
                
            # Load selection rules
            self.selection_rules = data.get('selection_rules', {})
            
            # Load metadata
            self.metadata = data.get('metadata', {})
            
            self.logger.info(f"Loaded {len(self.prompts)} prompts from {self.prompts_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")
            self._create_fallback_prompt()
            
    def _create_fallback_prompt(self):
        """Create fallback prompt if file loading fails"""
        fallback_prompt = """You are a helpful robotic assistant. You can control robot movements, 
answer questions, and engage in natural conversation. Be concise but friendly.
Respond naturally to the user's speech and provide helpful information or assistance.

You are capable of understanding and executing robot commands when requested.
Always prioritize safety and ask for clarification if commands are ambiguous."""
        
        self.prompts['fallback'] = PromptInfo(
            name="Fallback Assistant",
            description="Emergency fallback prompt when prompts.yaml unavailable",
            version="1.0",
            tested_with=["openai_realtime"],
            system_prompt=fallback_prompt
        )
        
        self.selection_rules = {'default': 'fallback'}
        self.logger.warning("Using fallback prompt due to loading error")
        
    def _expand_macros(self, text: str, max_depth: int = 10) -> str:
        """Expand macro placeholders in text recursively
        
        Args:
            text: Text containing {{macro_name}} placeholders
            max_depth: Maximum recursion depth to prevent infinite loops
            
        Returns:
            Text with all macros expanded
        """
        if max_depth <= 0:
            self.logger.warning("Maximum macro expansion depth reached")
            return text
            
        # Find all macro placeholders
        import re
        pattern = r'\{\{(\w+)\}\}'
        
        # Track if any replacements were made
        replacements_made = False
        
        def replace_macro(match):
            nonlocal replacements_made
            macro_name = match.group(1)
            
            if macro_name in self.macros:
                replacements_made = True
                # Get macro content and strip leading/trailing whitespace
                macro_content = self.macros[macro_name].strip()
                self.logger.debug(f"Expanding macro '{macro_name}'")
                return macro_content
            else:
                self.logger.warning(f"Macro '{macro_name}' not found")
                return match.group(0)  # Return original placeholder
        
        # Replace all macros in text
        expanded = re.sub(pattern, replace_macro, text)
        
        # If replacements were made, recursively expand any nested macros
        if replacements_made:
            expanded = self._expand_macros(expanded, max_depth - 1)
            
        return expanded
        
    def select_prompt(self, context: Dict[str, Any] = None) -> str:
        """Select appropriate prompt based on rules and context"""
        context = context or {}
        
        try:
            # Check conditional rules first
            for rule in self.selection_rules.get('conditional', []):
                condition = rule.get('condition', '')
                if self._evaluate_condition(condition, context):
                    prompt_id = rule.get('prompt')
                    if prompt_id and prompt_id in self.prompts:
                        self.logger.info(f"Selected prompt '{prompt_id}' via condition: {condition}")
                        return self.prompts[prompt_id].system_prompt
                        
            # Check A/B testing
            ab_tests = self.selection_rules.get('ab_tests', {})
            for test_name, test_config in ab_tests.items():
                if test_config.get('enabled', False):
                    selected_prompt = self._select_ab_variant(test_config)
                    if selected_prompt:
                        self.logger.info(f"Selected prompt '{selected_prompt}' via A/B test: {test_name}")
                        return self.prompts[selected_prompt].system_prompt
                        
            # Use default prompt
            default_prompt = self.selection_rules.get('default', 'fallback')
            if default_prompt in self.prompts:
                self.logger.info(f"Selected default prompt: {default_prompt}")
                return self.prompts[default_prompt].system_prompt
            else:
                self.logger.warning(f"Default prompt '{default_prompt}' not found, using fallback")
                return self.prompts['fallback'].system_prompt
                
        except Exception as e:
            self.logger.error(f"Error selecting prompt: {e}")
            return self.prompts.get('fallback', PromptInfo(
                name="Emergency", description="", version="1.0", 
                tested_with=[], system_prompt="You are a helpful assistant."
            )).system_prompt
            
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate conditional rule (simplified implementation)"""
        try:
            # Simple condition evaluation - can be extended for complex logic
            # Examples: "user_age < 10", "environment == 'crowded'"
            
            if '==' in condition:
                key, value = condition.split('==', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                return context.get(key) == value
                
            elif '<' in condition:
                key, value = condition.split('<', 1)
                key = key.strip()
                value = float(value.strip())
                context_value = context.get(key)
                if context_value is None:
                    return False  # None fails numeric comparisons
                return float(context_value) < value
                
            elif '>' in condition:
                key, value = condition.split('>', 1)
                key = key.strip()
                value = float(value.strip())
                context_value = context.get(key)
                if context_value is None:
                    return False  # None fails numeric comparisons  
                return float(context_value) > value
                
            else:
                # Simple boolean check
                return bool(context.get(condition.strip(), False))
                
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
            
    def _select_ab_variant(self, test_config: Dict[str, Any]) -> Optional[str]:
        """Select A/B test variant based on weights"""
        try:
            variants = test_config.get('variants', [])
            if not variants:
                return None
                
            # Calculate total weight
            total_weight = sum(variant.get('weight', 1) for variant in variants)
            
            # Random selection based on weights
            rand_value = random.random() * total_weight
            current_weight = 0
            
            for variant in variants:
                current_weight += variant.get('weight', 1)
                if rand_value <= current_weight:
                    prompt_id = variant.get('prompt')
                    if prompt_id in self.prompts:
                        return prompt_id
                        
            # Fallback to first variant
            first_prompt = variants[0].get('prompt')
            return first_prompt if first_prompt in self.prompts else None
            
        except Exception as e:
            self.logger.error(f"Error selecting A/B variant: {e}")
            return None
            
    def get_user_prefix(self, prefix_id: str, context: Dict[str, Any] = None) -> str:
        """Get user prompt prefix with context substitution"""
        context = context or {}
        
        if prefix_id not in self.user_prefixes:
            self.logger.warning(f"User prefix '{prefix_id}' not found")
            return ""
            
        try:
            prefix_template = self.user_prefixes[prefix_id].prefix
            
            # Simple template substitution
            for key, value in context.items():
                placeholder = f"{{{key}}}"
                if placeholder in prefix_template:
                    prefix_template = prefix_template.replace(placeholder, str(value))
                    
            return prefix_template
            
        except Exception as e:
            self.logger.error(f"Error processing user prefix '{prefix_id}': {e}")
            return ""
            
    def build_full_prompt(self, base_prompt: str = None, context: Dict[str, Any] = None, 
                         user_prefix_id: str = None) -> str:
        """Build complete prompt with base prompt, context, and user prefix"""
        context = context or {}
        
        # Select base prompt if not provided
        if base_prompt is None:
            base_prompt = self.select_prompt(context)
            
        # Add user prefix if specified
        user_prefix = ""
        if user_prefix_id:
            user_prefix = self.get_user_prefix(user_prefix_id, context)
            if user_prefix:
                user_prefix += "\n\n"
                
        return user_prefix + base_prompt
        
    def get_prompt_info(self, prompt_id: str) -> Optional[PromptInfo]:
        """Get information about a specific prompt"""
        return self.prompts.get(prompt_id)
        
    def list_prompts(self) -> List[str]:
        """List all available prompt IDs"""
        return list(self.prompts.keys())
        
    def list_macros(self) -> Dict[str, str]:
        """List all available macros and their content"""
        return self.macros.copy()
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get prompt metadata"""
        return self.metadata.copy()
        
    def reload_prompts(self):
        """Reload prompts from file"""
        self.logger.info("Reloading prompts from file...")
        self.prompts.clear()
        self.user_prefixes.clear()
        self.macros.clear()
        self.selection_rules.clear()
        self.metadata.clear()
        self._load_prompts()


# Factory function for easy instantiation
def create_prompt_loader(prompts_file: str = None) -> PromptLoader:
    """Create and return a PromptLoader instance"""
    return PromptLoader(prompts_file)


# Testing and debugging functions
def test_prompt_selection():
    """Test prompt selection with various contexts"""
    loader = PromptLoader()
    
    print("Available prompts:", loader.list_prompts())
    print("\nDefault prompt selection:")
    print(loader.select_prompt()[:100] + "...")
    
    print("\nTesting conditional selection:")
    context = {"user_age": 8, "environment": "crowded"}
    prompt = loader.select_prompt(context)
    print(f"Context {context}: {prompt[:100]}...")
    
    print("\nTesting user prefix:")
    prefix_context = {"last_topic": "robot movements", "last_command": "look left"}
    prefix = loader.get_user_prefix("context_reminder", prefix_context)
    print(f"Context reminder prefix: {prefix}")


if __name__ == "__main__":
    test_prompt_selection()