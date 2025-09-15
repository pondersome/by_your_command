#!/usr/bin/env python3
"""
Update all documentation files to use new topic naming convention.
Part of the topic renaming refactoring - Phase 7.
"""

import os
import re
from pathlib import Path

# Define the replacements for documentation
# More conservative for docs - we want to preserve historical context in some places
REPLACEMENTS = [
    # Topic names in code blocks and examples
    (r"`voice_chunks`", "`prompt_voice`"),
    (r"`text_input`", "`prompt_text`"),
    (r"`audio_out`", "`response_voice`"),
    (r"`llm_transcript`", "`response_text`"),
    (r"`command_transcript`", "`response_cmd`"),

    # Topic names in paths
    (r"/voice_chunks\b", "/prompt_voice"),
    (r"/text_input\b", "/prompt_text"),
    (r"/audio_out\b", "/response_voice"),
    (r"/llm_transcript\b", "/response_text"),
    (r"/command_transcript\b", "/response_cmd"),

    # Topic names in ROS commands
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

    # Topic references in text (be careful with these)
    (r"\bvoice_chunks topic", "prompt_voice topic"),
    (r"\btext_input topic", "prompt_text topic"),
    (r"\baudio_out topic", "response_voice topic"),
    (r"\bllm_transcript topic", "response_text topic"),
    (r"\bcommand_transcript topic", "response_cmd topic"),
]

def should_skip_file(filepath):
    """Check if a file should be skipped (e.g., the refactoring PRD itself)"""
    # Don't update the topic renaming PRD itself - it documents the change
    if "topic_renaming_refactoring_prd.md" in str(filepath):
        return True
    return False

def add_historical_note(filepath, content):
    """Add a historical note to PRD files that reference old topics"""
    if "/specs/" in str(filepath) and "prd" in filepath.name.lower():
        # Check if this is a historical document that should have a note
        if any(old in content for old in ["voice_chunks", "text_input", "audio_out", "llm_transcript", "command_transcript"]):
            if "Note: This document predates the topic renaming refactoring" not in content:
                # Add note after the header
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('# ') and i < 5:  # Found main header
                        lines.insert(i + 1, "\n> **Note: This document predates the topic renaming refactoring (2025-09-14). Some topic names mentioned here have been updated in the implementation. See `topic_renaming_refactoring_prd.md` for current naming.**\n")
                        return '\n'.join(lines)
    return content

def update_file(filepath):
    """Update a single file with the new topic names."""
    if should_skip_file(filepath):
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    changes_made = []

    for old_pattern, new_pattern in REPLACEMENTS:
        if re.search(old_pattern, content):
            count = len(re.findall(old_pattern, content))
            content = re.sub(old_pattern, new_pattern, content)
            changes_made.append(f"{old_pattern} -> {new_pattern} ({count} occurrences)")

    # Add historical note to PRD files if needed
    content = add_historical_note(filepath, content)
    if content != original_content and "Note: This document predates" in content:
        changes_made.append("Added historical note")

    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return changes_made
    return None

def main():
    # Find documentation files
    base_dir = Path("/home/karim/ros2_ws/src/by_your_command")

    # Find all markdown files
    md_files = []
    md_files.extend(base_dir.glob("*.md"))  # Root level docs
    md_files.extend(base_dir.glob("specs/*.md"))  # Spec docs

    print(f"Found {len(md_files)} documentation files")
    print("=" * 60)

    updated_files = []
    skipped_files = []

    for md_file in sorted(md_files):
        relative_path = md_file.relative_to(base_dir)
        changes = update_file(md_file)

        if changes:
            updated_files.append((relative_path, changes))
            print(f"✅ Updated: {relative_path}")
            for change in changes[:3]:  # Show first 3 changes
                print(f"   - {change}")
            if len(changes) > 3:
                print(f"   ... and {len(changes) - 3} more changes")
        else:
            if should_skip_file(md_file):
                print(f"⏭️  Skipped: {relative_path} (intentionally preserved)")
            else:
                skipped_files.append(relative_path)

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Updated: {len(updated_files)} files")
    print(f"  Skipped: {len(skipped_files) + 1} files")  # +1 for the PRD we intentionally skip

    if updated_files:
        print("\nFiles updated:")
        for filepath, _ in updated_files:
            print(f"  - {filepath}")

if __name__ == "__main__":
    main()