#!/usr/bin/env bash
# setup.sh: Install Python dependencies for ByYourCommand
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Setup voice chunk directories
echo ""
echo "Setting up voice chunk directories..."
mkdir -p /tmp/voice_chunks/{vad_chunks,assistant_output,user_input}
chmod -R 777 /tmp/voice_chunks
echo "Created voice chunk directories at /tmp/voice_chunks/"

echo ""
echo "Setup complete!"
