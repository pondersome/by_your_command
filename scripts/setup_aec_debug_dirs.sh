#!/bin/bash
# Setup directories for AEC debugging WAV file recordings
# This script creates the necessary folder structure under /tmp for voice recordings

echo "Setting up AEC debug directories..."

# Base directory for voice recordings
BASE_DIR="/tmp/voice_chunks"

# Create main directory
mkdir -p "$BASE_DIR"

# Create subdirectories for different recording types
SUBDIRS=(
    "test_tone"      # Records generated test tones (24kHz)
    "test_raw"       # Records raw microphone input (16kHz)
    "test_filtered"  # Records AEC-filtered output (16kHz)
    "vad_chunks"     # General VAD recordings from normal operation
    "realtime"       # Recordings from realtime mode
)

for dir in "${SUBDIRS[@]}"; do
    mkdir -p "$BASE_DIR/$dir"
    echo "Created: $BASE_DIR/$dir"
done

# Set permissions to allow ROS nodes to write
chmod -R 777 "$BASE_DIR"

echo "AEC debug directories setup complete!"
echo ""
echo "Directory structure:"
echo "$BASE_DIR/"
echo "├── test_tone/      # Clean test tone recordings"
echo "├── test_raw/       # Raw microphone with echo"
echo "├── test_filtered/  # Echo-cancelled output"
echo "├── vad_chunks/     # General voice recordings"
echo "└── realtime/       # Realtime mode recordings"
echo ""
echo "To test AEC, run: ros2 launch by_your_command aec_test.launch.py"
echo "WAV files will be saved to the directories above."