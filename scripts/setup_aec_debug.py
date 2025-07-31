#!/usr/bin/env python3
"""
Setup AEC debug directories for WAV file recordings.
Can be run standalone or as a ROS2 node.
"""

import os
import sys
from pathlib import Path


def setup_aec_debug_dirs():
    """Create directory structure for AEC debugging."""
    
    # Base directory for voice recordings
    base_dir = Path("/tmp/voice_chunks")
    
    # Subdirectories with descriptions
    subdirs = {
        "test_tone": "Records generated test tones (24kHz)",
        "test_raw": "Records raw microphone input with echo (16kHz)",
        "test_filtered": "Records AEC-filtered output (16kHz)",
        "vad_chunks": "General VAD recordings from normal operation",
        "realtime": "Recordings from realtime mode",
    }
    
    print("Setting up AEC debug directories...")
    
    # Create base directory
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created base directory: {base_dir}")
    
    # Create subdirectories
    for subdir, description in subdirs.items():
        dir_path = base_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        # Set permissions to 777 (readable/writable by all)
        dir_path.chmod(0o777)
        print(f"Created: {dir_path}")
    
    # Set base directory permissions
    base_dir.chmod(0o777)
    
    print("\nAEC debug directories setup complete!")
    print(f"\nDirectory structure:")
    print(f"{base_dir}/")
    for subdir, description in subdirs.items():
        print(f"├── {subdir:<15} # {description}")
    
    print("\nTo test AEC, run: ros2 launch by_your_command aec_test.launch.py")
    print("WAV files will be saved to the directories above.")
    
    # Also create a README in the base directory
    readme_path = base_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# AEC Debug Recordings\n\n")
        f.write("This directory contains WAV file recordings for debugging Acoustic Echo Cancellation.\n\n")
        f.write("## Directory Structure\n\n")
        for subdir, description in subdirs.items():
            f.write(f"- **{subdir}/**: {description}\n")
        f.write("\n## Usage\n\n")
        f.write("1. Run the AEC test: `ros2 launch by_your_command aec_test.launch.py`\n")
        f.write("2. WAV files will be automatically saved here\n")
        f.write("3. Compare test_raw vs test_filtered to measure echo reduction\n")
        f.write("4. Use Audacity or sox to analyze the recordings\n\n")
        f.write("## Analysis Example\n\n")
        f.write("```bash\n")
        f.write("# Show signal statistics\n")
        f.write("sox test_raw/*.wav -n stat\n")
        f.write("sox test_filtered/*.wav -n stat\n\n")
        f.write("# Generate spectrograms\n")
        f.write("sox test_raw/*.wav -n spectrogram\n")
        f.write("```\n")
    
    readme_path.chmod(0o666)
    print(f"\nCreated README at: {readme_path}")


def check_existing_files():
    """Check if there are existing recordings and report."""
    base_dir = Path("/tmp/voice_chunks")
    
    if not base_dir.exists():
        return
    
    print("\nExisting recordings found:")
    total_files = 0
    total_size = 0
    
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            wav_files = list(subdir.glob("*.wav"))
            if wav_files:
                size = sum(f.stat().st_size for f in wav_files)
                total_files += len(wav_files)
                total_size += size
                print(f"  {subdir.name}: {len(wav_files)} files ({size/1024/1024:.1f} MB)")
    
    if total_files > 0:
        print(f"\nTotal: {total_files} files ({total_size/1024/1024:.1f} MB)")
        
        # Check if running interactively
        if sys.stdin.isatty():
            try:
                response = input("\nDelete existing recordings? (y/N): ")
                if response.lower() == 'y':
                    for subdir in base_dir.iterdir():
                        if subdir.is_dir():
                            for wav_file in subdir.glob("*.wav"):
                                wav_file.unlink()
                    print("Existing recordings deleted.")
            except (EOFError, KeyboardInterrupt):
                print("\nKeeping existing recordings.")
        else:
            print("(Run interactively to delete existing recordings)")


if __name__ == "__main__":
    # Check for existing files first
    check_existing_files()
    
    # Setup directories
    setup_aec_debug_dirs()
    
    # If running as ROS node (optional)
    if "--ros" in sys.argv:
        try:
            import rclpy
            from rclpy.node import Node
            
            class SetupNode(Node):
                def __init__(self):
                    super().__init__('aec_setup_node')
                    self.get_logger().info("AEC debug directories have been set up")
                    
            rclpy.init()
            node = SetupNode()
            rclpy.shutdown()
        except ImportError:
            pass