# Product Requirements Document: Acoustic Echo Cancellation (AEC)

## Executive Summary

This document details the implementation of Acoustic Echo Cancellation (AEC) for the by_your_command human/robot interaction system. AEC is critical for enabling natural bidirectional conversations by removing the robot's own voice from the microphone input, allowing users to interrupt while the robot is speaking. In normal operator-robot interaction, a headset worn by the user is recommended to limit the need for AEC. But when the robot is dealing with ambient audio conditions to interact with passersby, the robot's voice is output on open speakers and an open microphone is used to capture the visitor's voice. In this scenario, AEC becomes essential.

I embarked on an extensive journey implementing Acoustic Echo Cancellation (AEC) for a ROS2-based voice assistant system. After exhaustive testing of multiple software AEC solutions (WebRTC, Speex, custom adaptive filters), I determined that my software-based approaches achieved disappointingly poor echo reduction (~1.5dB at best). The project ultimately concluded with the realization that PulseAudio's built-in `module-echo-cancel` provides superior system-level echo cancellation, rendering my custom implementations unnecessary. This document serves as a cautionary tale and learning resource for future AEC endeavors.

## Problem Statement

### Initial Challenge
- Voice assistant system unable to listen while speaking
- Crude microphone muting prevented natural conversation flow
- Users couldn't interrupt the assistant mid-sentence
- System vulnerable to feedback loops if unmuted

### Goal
Enable continuous microphone listening during assistant speech for natural, interruptible conversations.

## The Wild Goose Chase: Implementation Journey

### Phase 1: Software AEC Implementation

I implemented a comprehensive `aec_node` supporting three different algorithms:

1. **WebRTC AudioProcessingModule**
   - Industry-standard echo cancellation
   - Strict 10ms frame requirements
   - Result: Only -1.5dB average reduction

2. **Speex Echo Canceller**
   - Lightweight DSP solution
   - Flexible frame sizes
   - Result: Abysmal -0.1dB reduction

3. **Custom NLMS Adaptive Filter**
   - Educational implementation
   - Simple but ineffective
   - Result: Marginally better than Speex

### Phase 2: Debugging and Optimization Attempts

#### Acoustic Delay Measurement
I built custom tools to measure the actual acoustic delay:
- Expected: ~50ms based on distance
- Measured: ~73ms with correlation
- Reality: 110-199ms variable delay during operation

#### Buffer Management Nightmares
- WebRTC's strict 10ms chunks conflicted with our 512-sample Silero VAD requirements
- Complex buffering logic for frame alignment
- Resampling between 24kHz (OpenAI) and 16kHz (system standard)

#### Library Compatibility Hell
- NumPy 2.x broke scipy, forcing custom resampling implementation
- WebRTC Python bindings required specific SWIG versions
- Speex documentation sparse, required reading C++ source

### Phase 3: The Reality Check

After days of implementation and testing:
- Best case: -1.5dB reduction (barely noticeable)
- Complex acoustic environment defeated all algorithms
- High computational cost for minimal benefit
- Extensive code complexity for poor results

## The Solution: PulseAudio Echo Cancellation

After all custom implementation efforts failed, I discovered:

```bash
# One simple command
pactl load-module module-echo-cancel
```

This system-level solution:
- Operates before audio reaches ROS nodes
- Dramatically reduces echo (20-30 RMS → 0.5-1.0 RMS)
- No custom code required
- Superior performance to all our implementations

## Final Architecture

### What I Built (Then Deleted)
- 1000+ lines of AEC implementation code
- Complex test infrastructure with tone generators
- Multiple launch configurations
- Extensive debugging and analysis tools

### What Actually Works
1. PulseAudio echo cancellation at system level
2. Simple amplitude threshold in Silero VAD for residual filtering
3. Standard 512-sample chunks throughout pipeline
4. No custom AEC code whatsoever

## Lessons Learned

### Technical Insights
1. **System-level solutions trump application-level ones** - OS and hardware have inherent advantages
2. **Complex acoustics defeat simple algorithms** - Real environments have reflections, reverb, and non-linearities
3. **Library compatibility is fragile** - Version conflicts can derail implementations
4. **Measurement != Reality** - Lab measurements don't reflect operational conditions

### Project Management Insights
1. **Try system solutions first** - Could have saved days of effort
2. **Validate early and often** - Should have tested echo reduction sooner
3. **Document the journey** - Failed attempts provide valuable learning
4. **Know when to quit** - Sunk cost fallacy led to continued effort despite poor results

## Setup Instructions

For future deployments, skip the custom AEC entirely:

1. Enable PulseAudio echo cancellation:
   ```bash
   pactl load-module module-echo-cancel
   ```

2. Configure Silero VAD with amplitude filtering:
   ```yaml
   amplitude_threshold: 100.0  # Filter residual echo
   ```

3. Use standard audio configuration:
   - 16kHz sample rate throughout
   - 512-sample chunks (32ms)
   - Default audio devices

## Conclusion

This project exemplifies how sophisticated software solutions can be outperformed by existing system-level tools. While the implementation journey provided valuable learning about audio processing, real-time constraints, and algorithm limitations, the final solution required zero custom code. Sometimes the best code is the code you delete.

### Time and Token Cost
- Development time: ~3 days
- Code written then deleted: ~1500 lines
- Final solution: 1 command
- Tokens consumed: "too many" (per user feedback)

This document preserves my journey as a reminder: always investigate existing system-level solutions before embarking on complex custom implementations.