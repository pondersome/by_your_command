#!/usr/bin/env python3
"""
Update all test files to use new topic naming convention.
Part of the topic renaming refactoring - Phase 6.
"""

import os
import re
from pathlib import Path

# Define the replacements
REPLACEMENTS = [
    # Direct topic name replacements (in quotes)
    (r"'voice_chunks'", "'prompt_voice'"),
    (r'"voice_chunks"', '"prompt_voice"'),
    (r"'text_input'", "'prompt_text'"),
    (r'"text_input"', '"prompt_text"'),
    (r"'audio_out'", "'response_voice'"),
    (r'"audio_out"', '"response_voice"'),
    (r"'llm_transcript'", "'response_text'"),
    (r'"llm_transcript"', '"response_text"'),
    (r"'command_transcript'", "'response_cmd'"),
    (r'"command_transcript"', '"response_cmd"'),

    # Topic names in paths/strings
    (r"/voice_chunks\b", "/prompt_voice"),
    (r"/text_input\b", "/prompt_text"),
    (r"/audio_out\b", "/response_voice"),
    (r"/llm_transcript\b", "/response_text"),
    (r"/command_transcript\b", "/response_cmd"),

    # Variable/parameter names (be more careful here)
    (r"\bvoice_chunks_topic\b", "prompt_voice_topic"),
    (r"\btext_input_topic\b", "prompt_text_topic"),
    (r"\baudio_out_topic\b", "response_voice_topic"),
    (r"\bllm_transcript_topic\b", "response_text_topic"),
    (r"\bcommand_transcript_topic\b", "response_cmd_topic"),
]

def update_file(filepath):
    """Update a single file with the new topic names."""
    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    changes_made = []

    for old_pattern, new_pattern in REPLACEMENTS:
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_pattern, content)
            changes_made.append(f"{old_pattern} -> {new_pattern}")

    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return changes_made
    return None

def main():
    # Find the tests directory
    tests_dir = Path("/home/karim/ros2_ws/src/by_your_command/tests")

    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return

    # Find all Python test files
    test_files = list(tests_dir.glob("**/*.py"))

    print(f"Found {len(test_files)} Python files in tests directory")
    print("=" * 60)

    updated_files = []
    skipped_files = []

    for test_file in sorted(test_files):
        relative_path = test_file.relative_to(tests_dir)
        changes = update_file(test_file)

        if changes:
            updated_files.append((relative_path, changes))
            print(f"✅ Updated: {relative_path}")
            for change in changes[:3]:  # Show first 3 changes
                print(f"   - {change}")
            if len(changes) > 3:
                print(f"   ... and {len(changes) - 3} more changes")
        else:
            skipped_files.append(relative_path)

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Updated: {len(updated_files)} files")
    print(f"  Skipped: {len(skipped_files)} files (no changes needed)")

    if updated_files:
        print("\nFiles updated:")
        for filepath, _ in updated_files:
            print(f"  - {filepath}")

if __name__ == "__main__":
    main()