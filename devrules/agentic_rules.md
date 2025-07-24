# Agentic Coding Rules

This document defines the evolving rules for AI code generation and actions within the by_your_command package.

1. Always include descriptive logging.
2. Keep nodes modular and decoupled.
3. Validate parameters at startup. Always support parameters in launch files and code with overrideable defaults. Always support namespace and subsystem parameters. subsystem is used as a prefix for topics, services, actions and tf frames.
4. Follow ROS2 best practices for Python and C++ nodes.
5. Ensure secure handling of API keys.
6. Write tests for new features.
