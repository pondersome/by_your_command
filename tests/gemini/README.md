# Gemini Live API Test Suite

This directory contains comprehensive tests for the Google Gemini Live API, documenting both successful patterns and debugging approaches.

## Test Files

### Integration Tests (ROS2)

- **`test_gemini_with_image.py`** - Test Gemini agent with real camera integration
  - Validates the agent can see through camera and respond to voice
  - Uses unified `session.send(input={...})` API for images
  - Confirms multimodal (audio + vision) processing works

- **`test_gemini_garbage_images.py`** - Image validation testing
  - Tests Gemini's response to various invalid image data
  - Proves Gemini validates images (garbage causes errors)
  - Confirms valid images work when sent correctly
  - Critical for debugging image processing issues

### Core Functionality Tests

- **`test_text_communication.py`** - Text input/output with Gemini Live
  - Sends text messages and receives text responses
  - Tests proper use of `send_client_content()` with `turn_complete=True`

- **`test_audio_communication.py`** - Audio-to-audio conversation
  - Sends 16kHz PCM16 audio input
  - Receives 24kHz PCM16 audio output
  - Saves output to `../output/audio_to_audio_verified.wav`

- **`test_streaming_audio.py`** - Streaming audio input test
  - Demonstrates sending audio in chunks (streaming)
  - Shows correct receive generator timing (create after first send)
  - Proves no buffering needed for audio input

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

- **`test_receive_issue.py`** - Receive generator timing issue
  - Demonstrates the problem with creating receiver before sending
  - Critical for understanding the generator lifecycle

- **`test_receive_after_send.py`** - Correct receive pattern
  - Shows creating receiver AFTER sending works correctly
  - Proves the timing requirement for receive generators

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
9. **Receive generator pattern**: 
   - MUST create `session.receive()` AFTER sending input, not before
   - Generator exhausts immediately if no pending responses
   - One generator per conversation turn (not persistent like OpenAI)
   - Supports streaming audio input (send chunks as they arrive)
10. **Image integration (CRITICAL FINDING)**:
    - **MUST use `session.send(input={...})` for realtime context**
    - Images need base64 encoding when sent via unified API
    - `send_client_content()` doesn't work for realtime multimodal
    - Latest frame pattern: Store most recent image, send with voice/text

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