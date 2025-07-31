#!/usr/bin/env bash
# setup.sh: Install Python dependencies for ByYourCommand
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Setup AEC debug directories
echo ""
echo "Setting up AEC debug directories..."
if [ -f "$SCRIPT_DIR/../scripts/setup_aec_debug.py" ]; then
    python3 "$SCRIPT_DIR/../scripts/setup_aec_debug.py"
else
    # Fallback if script not found
    mkdir -p /tmp/voice_chunks/{test_tone,test_raw,test_filtered,vad_chunks,realtime}
    chmod -R 777 /tmp/voice_chunks
    echo "Created AEC debug directories at /tmp/voice_chunks/"
fi

echo ""
echo "Setup complete!"
