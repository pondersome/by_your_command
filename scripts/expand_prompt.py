#!/usr/bin/env python3
"""
Command-line utility to expand prompts.yaml macros for comparison and debugging.
Similar to xacro but for our prompt system.

Usage:
    expand_prompt.py <macro_name> [options]
    
Examples:
    # Expand to stdout
    expand_prompt.py barney_conversational
    
    # Save to file
    expand_prompt.py barney_conversational -o expanded.txt
    
    # Compare two macros
    expand_prompt.py barney_conversational_oai -o oai.txt
    expand_prompt.py barney_conversational_gemini -o gemini.txt
    diff oai.txt gemini.txt
    
    # Show macro names as comments
    expand_prompt.py barney_conversational --show-names
    
    # Show with indentation
    expand_prompt.py barney_conversational --show-names --indent
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Set

class PromptExpander:
    """Expand prompt macros with optional formatting"""
    
    def __init__(self, prompts_file: str = None, show_names: bool = False, 
                 indent: bool = False):
        """
        Initialize the prompt expander.
        
        Args:
            prompts_file: Path to prompts.yaml file
            show_names: Whether to show macro names as comments
            indent: Whether to indent nested macros
        """
        self.show_names = show_names
        self.indent_nested = indent
        self.indent_str = "  "  # Two spaces per level
        
        # Find prompts.yaml file
        if prompts_file and os.path.exists(prompts_file):
            self.prompts_file = prompts_file
        else:
            # Try multiple locations
            possible_paths = []
            
            # 1. Check ROS2 package share directory (installed location)
            try:
                from ament_index_python.packages import get_package_share_directory
                package_share = get_package_share_directory('by_your_command')
                share_config_path = Path(package_share) / "config" / "prompts.yaml"
                possible_paths.append(share_config_path)
            except:
                pass
            
            # 2. Check relative to script in source tree
            script_dir = Path(__file__).parent.parent
            source_config_path = script_dir / "config" / "prompts.yaml"
            possible_paths.append(source_config_path)
            
            # 3. Check current working directory
            cwd_config_path = Path.cwd() / "config" / "prompts.yaml"
            possible_paths.append(cwd_config_path)
            
            # Try each path
            for path in possible_paths:
                if path.exists():
                    self.prompts_file = str(path)
                    break
            else:
                raise FileNotFoundError(
                    "Could not find prompts.yaml. Please specify path with -p"
                )
        
        # Load prompts
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file"""
        with open(self.prompts_file, 'r') as f:
            data = yaml.safe_load(f)
            
        # Extract all sections that might contain macros
        all_prompts = {}
        
        # Main prompts section
        if 'prompts' in data:
            all_prompts.update(data['prompts'])
            
        # Macro definitions section
        if 'macros' in data:
            all_prompts.update(data['macros'])
            
        # Legacy sections (for backward compatibility)
        if 'user_prompt_prefixes' in data:
            all_prompts.update(data['user_prompt_prefixes'])
            
        return all_prompts
    
    def expand(self, prompt_id: str, depth: int = 0, 
               visited: Optional[Set[str]] = None) -> str:
        """
        Recursively expand a prompt macro.
        
        Args:
            prompt_id: The macro name to expand
            depth: Current nesting depth (for indentation)
            visited: Set of already visited macros (to detect cycles)
            
        Returns:
            Expanded prompt text
        """
        if visited is None:
            visited = set()
            
        # Check for circular references
        if prompt_id in visited:
            return f"[ERROR: Circular reference detected for {prompt_id}]"
            
        visited.add(prompt_id)
        
        # Get the prompt template
        if prompt_id not in self.prompts:
            return f"[ERROR: Macro '{prompt_id}' not found]"
            
        template = self.prompts[prompt_id]
        
        # Handle dict-style prompts (with 'system_prompt' field)
        if isinstance(template, dict):
            # If it has a system_prompt field, use that
            if 'system_prompt' in template:
                template = template['system_prompt']
            else:
                # Otherwise convert the whole dict to string
                template = str(template)
        elif not isinstance(template, str):
            template = str(template)
        
        # Build output with optional comment
        output_lines = []
        
        if self.show_names and depth > 0:
            # Add comment showing macro name
            indent = self.indent_str * (depth - 1) if self.indent_nested else ""
            comment = f"{indent}# --- Begin macro: {prompt_id} ---"
            output_lines.append(comment)
        
        # Process the template line by line to handle indentation
        lines = template.split('\n')
        expanded_lines = []
        
        for line in lines:
            # Check for macro references
            expanded_line = self._expand_line(line, depth, visited.copy())
            expanded_lines.append(expanded_line)
        
        # Apply indentation if needed
        if self.indent_nested and depth > 0:
            indent = self.indent_str * depth
            expanded_lines = [indent + line if line.strip() else line 
                            for line in expanded_lines]
        
        output_lines.extend(expanded_lines)
        
        if self.show_names and depth > 0:
            # Add comment showing end of macro
            indent = self.indent_str * (depth - 1) if self.indent_nested else ""
            comment = f"{indent}# --- End macro: {prompt_id} ---"
            output_lines.append(comment)
        
        return '\n'.join(output_lines)
    
    def _expand_line(self, line: str, depth: int, visited: Set[str]) -> str:
        """
        Expand macro references in a single line.
        
        Handles {{macro_name}} syntax.
        """
        import re
        
        # Pattern to match {{macro_name}}
        pattern = r'\{\{(\w+)\}\}'
        
        def replace_macro(match):
            macro_name = match.group(1)
            # Recursively expand the macro
            expanded = self.expand(macro_name, depth + 1, visited)
            return expanded
        
        # Replace all macro references
        expanded = re.sub(pattern, replace_macro, line)
        
        return expanded
    
    def format_output(self, expanded_text: str) -> str:
        """
        Apply any final formatting to the output.
        """
        # Clean up multiple blank lines
        lines = expanded_text.split('\n')
        cleaned = []
        prev_blank = False
        
        for line in lines:
            is_blank = not line.strip()
            # Don't add multiple consecutive blank lines
            if not (is_blank and prev_blank):
                cleaned.append(line)
            prev_blank = is_blank
        
        return '\n'.join(cleaned)


def main():
    parser = argparse.ArgumentParser(
        description='Expand prompt macros from prompts.yaml',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s barney_conversational                    # Expand to stdout
  %(prog)s barney_conversational -o expanded.txt    # Save to file
  %(prog)s barney_conversational --show-names       # Include macro names as comments
  %(prog)s barney_conversational --indent           # Indent nested macros
  %(prog)s barney_conversational -sni               # All formatting options
  
Compare two macros:
  %(prog)s barney_conversational_oai -o oai.txt
  %(prog)s barney_conversational_gemini -o gemini.txt
  diff oai.txt gemini.txt
        """
    )
    
    parser.add_argument('macro_name', 
                      nargs='?',
                      help='Name of the macro to expand')
    
    parser.add_argument('-o', '--output', 
                      help='Output file (default: stdout)')
    
    parser.add_argument('-p', '--prompts-file',
                      help='Path to prompts.yaml file')
    
    parser.add_argument('-s', '--show-names', 
                      action='store_true',
                      help='Show macro names as comments')
    
    parser.add_argument('-n', '--indent', 
                      action='store_true',
                      help='Indent nested macros')
    
    parser.add_argument('-i', '--indent-size',
                      type=int, default=2,
                      help='Number of spaces per indent level (default: 2)')
    
    parser.add_argument('-l', '--list',
                      action='store_true',
                      help='List all available macros and exit')
    
    parser.add_argument('-r', '--raw',
                      action='store_true',
                      help='Show raw template without expansion')
    
    args = parser.parse_args()
    
    try:
        # Create expander
        expander = PromptExpander(
            prompts_file=args.prompts_file,
            show_names=args.show_names,
            indent=args.indent
        )
        
        # Set indent size if specified
        if args.indent_size:
            expander.indent_str = " " * args.indent_size
        
        # List macros if requested
        if args.list:
            print("Available macros:")
            for name in sorted(expander.prompts.keys()):
                # Show first 60 chars of content as preview
                content = str(expander.prompts[name])[:60].replace('\n', ' ')
                if len(str(expander.prompts[name])) > 60:
                    content += "..."
                print(f"  {name:40} {content}")
            return 0
        
        # Check if macro_name is required
        if not args.macro_name:
            parser.error("macro_name is required unless using --list")
            return 1
        
        # Show raw template if requested
        if args.raw:
            if args.macro_name not in expander.prompts:
                print(f"Error: Macro '{args.macro_name}' not found", file=sys.stderr)
                return 1
            output = expander.prompts[args.macro_name]
        else:
            # Expand the macro
            output = expander.expand(args.macro_name)
            output = expander.format_output(output)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
                if not output.endswith('\n'):
                    f.write('\n')
            print(f"Expanded prompt written to: {args.output}")
        else:
            print(output)
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())