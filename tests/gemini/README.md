# Gemini Live API Test Suite

This directory contains comprehensive tests for the Google Gemini Live API, documenting both successful patterns and debugging approaches.

## Test Files

### Core Functionality Tests

- **`test_text_communication.py`** - Text input/output with Gemini Live
  - Sends text messages and receives text responses
  - Tests proper use of `send_client_content()` with `turn_complete=True`

- **`test_audio_communication.py`** - Audio-to-audio conversation
  - Sends 16kHz PCM16 audio input
  - Receives 24kHz PCM16 audio output
  - Saves output to `../output/audio_to_audio_verified.wav`

- **`test_image_audio.py`** - Image input with audio description
  - Sends images and receives audio descriptions
  - Tests multimodal capabilities
  - Saves complete audio response (not just fragments)

- **`test_object_detection.py`** - Basic object detection with bounding boxes
  - Sends images for object detection
  - Returns JSON with normalized coordinates (0-1000 scale)
  - Converts to pixel coordinates
  - Saves results to `../output/object_detection_results.json`

- **`test_object_detection_visual.py`** - Visual object detection with overlays
  - Detects objects including background elements (sand, sea, sky)
  - Draws color-coded bounding boxes on the original image
  - Shows labels with confidence scores
  - Saves annotated image to `../output/detection_annotated.png`

- **`test_detection_visual_improved.py`** - Enhanced visual detection
  - Improved drawing order (larger boxes first for better visibility)
  - Quality indicators (★★★ for >95%, ★★ for 90-95%, etc.)
  - Corner markers for clearer bounding boxes
  - Option to test multiple models with `--preview` flag
  - Saves model-specific outputs

- **`test_video_detection.py`** - Video frames and detection
  - Tests inline video frame handling
  - Compares detection vs description prompts
  - Verifies coordinate system (0,0 at top-left)

- **`test_completion_signals.py`** - Response completion detection
  - Tests for `generation_complete` and `turn_complete` signals
  - Important for knowing when audio streaming is finished

### Debugging and Troubleshooting Tests

These tests were crucial for understanding the API and debugging issues:

- **`test_gemini_minimal.py`** - Minimal connection test
  - Simplest possible test to verify API connection
  - Good starting point for debugging

- **`test_gemini_inspect.py`** - API method inspection
  - Explores available session methods
  - Helps understand the API surface

- **`test_model_names.py`** - Model name format testing
  - Discovered that models need `models/` prefix
  - Tests different model name formats

- **`test_gemini_connection.py`** - Connection debugging
  - Tests connection establishment
  - Helped identify timeout and authentication issues

- **`test_gemini_ready.py`** - Session readiness testing
  - Tests when sessions are ready for input
  - Gemini sessions are ready immediately (unlike OpenAI)

## Output Directory

All test outputs are saved to `../output/`:
- Audio files (`.wav`)
- Detection results (`.json`)
- Text descriptions (`.txt`)

The output directory has a `.gitignore` to keep it clean in version control.

## Key Learnings

1. **Model names MUST use `models/` prefix** (e.g., `models/gemini-2.0-flash-live-001`)
2. **Audio format**: Input 16kHz PCM16, Output 24kHz PCM16
3. **No `turn_complete` with audio** - causes errors
4. **Coordinates**: Normalized 0-1000, format `[ymin, xmin, ymax, xmax]`, (0,0) at top-left
5. **Completion signals**: Look for `server_content.turn_complete=True`
6. **Response streaming**: Audio comes in many small chunks (9600-11520 bytes)
7. **Object detection quality**: 
   - Includes background elements (sand, sea, sky)
   - Confidence varies - consider 90%+ threshold for reliability
   - Sometimes returns malformed JSON requiring robust parsing
   - Detection quality may vary between models
8. **Visual detection tips**:
   - Draw larger boxes first for better visibility
   - Fine details detected well (e.g., small shells)
   - Bounding box accuracy varies with confidence level

## Running Tests

```bash
cd tests/gemini

# Basic tests
python3 test_text_communication.py      # Start here
python3 test_audio_communication.py      # Test audio I/O
python3 test_object_detection.py         # Test vision capabilities

# Visual detection tests
python3 test_object_detection_visual.py  # Creates annotated image
python3 test_detection_visual_improved.py # Enhanced visualization
python3 test_detection_visual_improved.py --preview  # Test multiple models

# Debugging tests (if having issues)
python3 test_gemini_minimal.py          # Minimal connection test
python3 test_model_names.py             # Test model name formats
```

Make sure to set your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Dependencies

```bash
pip install google-genai soundfile librosa pillow
```