#!/usr/bin/env bash
# setup.sh: Install Python dependencies for ByYourCommand
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install -r "$SCRIPT_DIR/requirements.txt"
