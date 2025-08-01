#!/usr/bin/env python3
"""
Acoustic Echo Cancellation Node with Resampling

Removes speaker output from microphone input using adaptive filtering,
allowing users to interrupt while the assistant is speaking.
Handles resampling from 24kHz (assistant) to 16kHz (microphone).

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from audio_common_msgs.msg import AudioData, AudioStamped
from std_msgs.msg import Bool
import numpy as np
from collections import deque
import time
from datetime import datetime

# Resampling imports
SCIPY_AVAILABLE = False  # Disable scipy due to numpy compatibility issues
# try:
#     from scipy import signal
#     SCIPY_AVAILABLE = True
# except ImportError as e:
#     print(f"Warning: scipy import failed: {e}")
#     SCIPY_AVAILABLE = False
# except Exception as e:
#     # Catch any other errors (like version conflicts)
#     print(f"Warning: scipy error: {e}")
#     SCIPY_AVAILABLE = False

# Try to import WebRTC audio processing
try:
    from webrtc_audio_processing import AudioProcessingModule as AP
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    
# Try to import speexdsp as fallback
try:
    from speexdsp import EchoCanceller
    SPEEX_AVAILABLE = True
    print("Speex import successful at module level")
except ImportError as e:
    SPEEX_AVAILABLE = False
    print(f"Speex import failed: {e}")
except Exception as e:
    SPEEX_AVAILABLE = False
    print(f"Speex import error: {type(e).__name__}: {e}")


def get_timestamp():
    """Get formatted timestamp HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]


class AcousticEchoCancellerNode(Node):
    def __init__(self):
        super().__init__('aec_node')
        self.setup_node()
        
    def log_info(self, msg):
        """Log with custom format"""
        self.get_logger().info(f"[{get_timestamp()}] [aec] {msg}")
        
    def log_debug(self, msg):
        """Debug log with custom format"""
        if hasattr(self, 'debug_logging') and self.debug_logging:
            self.get_logger().debug(f"[{get_timestamp()}] [aec] DEBUG: {msg}")
            
    def log_warning(self, msg):
        """Warning log with custom format"""
        self.get_logger().warning(f"[{get_timestamp()}] [aec] WARNING: {msg}")
        
    def log_error(self, msg):
        """Error log with custom format"""
        self.get_logger().error(f"[{get_timestamp()}] [aec] ERROR: {msg}")
        
    def setup_node(self):
        """Setup node after logging methods are defined"""
        # Parameters
        self.declare_parameter('mic_sample_rate', 16000)
        self.declare_parameter('speaker_sample_rate', 24000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('frame_size', 160)  # 10ms at 16kHz
        self.declare_parameter('filter_length', 2048)  # ~128ms at 16kHz
        self.declare_parameter('aec_method', 'adaptive')  # webrtc, speex, or adaptive
        self.declare_parameter('suppression_level', 'moderate')  # low, moderate, high
        self.declare_parameter('learning_rate', 0.3)
        self.declare_parameter('debug_logging', False)
        self.declare_parameter('bypass', False)  # Bypass AEC for testing
        self.declare_parameter('test_mode', False)  # Generate test tones
        
        self.mic_sample_rate = self.get_parameter('mic_sample_rate').value
        self.speaker_sample_rate = self.get_parameter('speaker_sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.frame_size = self.get_parameter('frame_size').value
        self.filter_length = self.get_parameter('filter_length').value
        self.aec_method = self.get_parameter('aec_method').value
        self.suppression_level = self.get_parameter('suppression_level').value
        self.learning_rate = self.get_parameter('learning_rate').value
        self.debug_logging = self.get_parameter('debug_logging').value
        self.bypass = self.get_parameter('bypass').value
        self.test_mode = self.get_parameter('test_mode').value
        
# Validate resampling capability
        if self.speaker_sample_rate != self.mic_sample_rate and not SCIPY_AVAILABLE:
            self.get_logger().warning("scipy not available for resampling, using simple decimation")
            # For 24kHz to 16kHz, we can use simple 3:2 decimation
            if self.speaker_sample_rate == 24000 and self.mic_sample_rate == 16000:
                self.use_simple_resampling = True
            else:
                self.get_logger().error("Unsupported sample rate conversion without scipy!")
                raise ImportError("Please install scipy: pip install scipy")
        else:
            self.use_simple_resampling = False
        
# Initialize AEC based on available libraries
        self.aec = None
        self.aec_type = None  # Track which type is active
        if self.aec_method == 'webrtc' and WEBRTC_AVAILABLE:
            self._init_webrtc_aec()
        elif self.aec_method == 'speex' and SPEEX_AVAILABLE:
            self._init_speex_aec()
        else:
            self._init_adaptive_aec()
            
        # Buffers for audio alignment
        self.speaker_buffer = deque(maxlen=int(self.mic_sample_rate * 0.5))  # 500ms buffer at 16kHz (for non-WebRTC)
        self.speaker_buffer_24k = deque(maxlen=int(self.speaker_sample_rate * 0.5))  # 500ms buffer at 24kHz (for WebRTC)
        self.mic_buffer = deque(maxlen=int(self.mic_sample_rate * 0.5))
        
        # Resampling ratio
        self.resample_ratio = self.mic_sample_rate / self.speaker_sample_rate
        
        # Delay compensation (speaker to mic path delay)
        self.delay_samples = int(0.13 * self.mic_sample_rate)  # Start with 130ms based on measurements
        self.delay_estimator = DelayEstimator(self.mic_sample_rate)
        self.measured_delays = deque(maxlen=50)  # Track last 50 delay measurements
        self.delay_update_counter = 0
        
        # State tracking
        self.assistant_speaking = False
        self.last_speaker_time = 0
        
        # Metrics tracking
        self.mic_chunks_processed = 0
        self.speaker_chunks_received = 0
        self.echo_reduction_sum = 0.0
        self.webrtc_chunks_processed = 0
        self.webrtc_underruns = 0
        self.last_status_time = time.time()
        
        # Test tone generation
        if self.test_mode:
            self.test_tone_freq = 1000  # 1kHz
            self.test_tone_duration = 1.0  # 1 second
            self.test_tone_interval = 3.0  # Every 3 seconds
            self.last_tone_time = 0
            self.tone_start_time = 0
            self.tone_detected_time = 0
            self.tone_in_mic_power = 0
            self.tone_in_filtered_power = 0
            self.log_info("Test mode enabled - will generate tones every 3 seconds")
        
        # QoS profile for audio topics (match audio_common)
        audio_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers and subscribers
        self.create_subscription(
            AudioStamped, 
            'audio',  # Raw microphone input
            self.mic_callback, 
            audio_qos
        )
        
        self.create_subscription(
            AudioData,
            'audio_out',  # Speaker output (assistant voice)
            self.speaker_callback,
            audio_qos  # Use same QoS as other audio topics
        )
        
        self.audio_pub = self.create_publisher(
            AudioStamped, 
            'audio_filtered',  # Echo-cancelled output
            audio_qos
        )
        
        self.status_pub = self.create_publisher(
            Bool,
            'aec_active',  # Whether AEC is actively cancelling
            10
        )
        
        # Test tone publisher (only if in test mode)
        if self.test_mode:
            self.tone_pub = self.create_publisher(
                AudioData,
                'test_tone_out',  # Different topic to avoid feedback loop
                audio_qos
            )
            # Timer for periodic tone generation
            self.create_timer(0.1, self._test_tone_callback)
        
        # Log startup diagnostics
        self.log_info("=== AEC Node Startup Diagnostics ===")
        self.log_info(f"Requested method: {self.aec_method}")
        self.log_info(f"Libraries available:")
        self.log_info(f"  - WebRTC: {WEBRTC_AVAILABLE}")
        self.log_info(f"  - Speex: {SPEEX_AVAILABLE}")
        self.log_info(f"  - SciPy: {SCIPY_AVAILABLE}")
        self.log_info(f"Initialized with: {self.aec_type}")
        self.log_info(f"Bypass mode: {self.bypass}")
        self.log_info(f"Test mode: {self.test_mode}")
        if self.test_mode:
            self.log_info(f"  - Tone frequency: {self.test_tone_freq}Hz")
            self.log_info(f"  - Tone duration: {self.test_tone_duration}s")
            self.log_info(f"  - Tone interval: {self.test_tone_interval}s")
        self.log_info(f"Resampling: {self.speaker_sample_rate}Hz → {self.mic_sample_rate}Hz")
        self.log_info(f"Debug logging: {self.debug_logging}")
        self.log_info("===================================")
        
    def _init_webrtc_aec(self):
        """Initialize WebRTC echo canceller"""
        try:
            # WebRTC audio processing module with AEC enabled
            # Constructor: AP(aec_type=2, enable_ns=False, enable_agc=False, enable_vad=False)
            # aec_type: 0=disabled, 1=AEC1, 2=AEC2 (better)
            aec_type = 2  # Use AEC2 for better echo cancellation
            enable_ns = True  # Also enable noise suppression
            enable_agc = False  # We don't need AGC
            enable_vad = False  # We use Silero for VAD
            
            self.aec = AP(aec_type, enable_ns, enable_agc, enable_vad)
            
            # Set stream formats for both mic and speaker
            self.aec.set_stream_format(self.mic_sample_rate, self.channels)
            self.aec.set_reverse_stream_format(self.speaker_sample_rate, self.channels)
            
            # Set system delay based on our measurements (~130ms)
            # Note: delay_samples is set later, so we use a fixed 130ms for now
            delay_ms = 130
            self.aec.set_system_delay(delay_ms)
            self.log_info(f"WebRTC system delay set to {delay_ms}ms")
            
            # Configure noise suppression level if enabled
            if enable_ns:
                ns_level = {
                    'low': 0,
                    'moderate': 1,
                    'high': 2
                }.get(self.suppression_level, 1)
                self.aec.set_ns_level(ns_level)
            
            # WebRTC requires exactly 10ms chunks
            self.webrtc_mic_frame_size = int(self.mic_sample_rate * 0.01)  # 10ms at 16kHz = 160 samples
            self.webrtc_speaker_frame_size = int(self.speaker_sample_rate * 0.01)  # 10ms at 24kHz = 240 samples
            self.webrtc_mic_buffer = []
            self.webrtc_speaker_buffer = []
            
            self.aec_type = 'webrtc'
            self.log_info(f"Initialized WebRTC AEC2 + NS (mic: {self.webrtc_mic_frame_size} samples, "
                         f"speaker: {self.webrtc_speaker_frame_size} samples @ 10ms)")
            
        except Exception as e:
            self.log_error(f"Failed to init WebRTC AEC: {e}")
            import traceback
            self.log_error(f"Traceback: {traceback.format_exc()}")
            self._init_adaptive_aec()
        
    def _init_speex_aec(self):
        """Initialize Speex echo canceller"""
        try:
            # Check if parameters are valid
            self.log_info(f"Attempting Speex init with frame_size={self.frame_size}, "
                         f"filter_length={self.filter_length}, sample_rate={self.mic_sample_rate}")
            
            # EchoCanceller.create() takes exactly 3 arguments
            self.aec = EchoCanceller.create(
                self.frame_size,      # frame_size (must match processing chunk size)
                self.filter_length,   # filter_length (echo tail length)
                self.mic_sample_rate  # sample_rate
            )
            self.aec_type = 'speex'
            self.log_info(f"Successfully initialized Speex AEC!")
        except Exception as e:
            self.log_error(f"Failed to init Speex AEC: {type(e).__name__}: {e}")
            self.log_error(f"Speex available: {SPEEX_AVAILABLE}")
            import traceback
            self.log_error(f"Traceback: {traceback.format_exc()}")
            self._init_adaptive_aec()
        
    def _init_adaptive_aec(self):
        """Initialize custom adaptive filter AEC"""
        self.aec = AdaptiveEchoCanceller(
            filter_length=self.filter_length,
            frame_size=self.frame_size,
            sample_rate=self.mic_sample_rate,
            learning_rate=self.learning_rate
        )
        self.aec_type = 'adaptive'
        self.log_info("Initialized adaptive filter AEC")
        
        
    def speaker_callback(self, msg):
        """Handle speaker output audio"""
        # Convert to numpy array
        audio_data = np.array(msg.int16_data, dtype=np.int16)
        
        self.speaker_chunks_received += 1
        if self.speaker_chunks_received == 1:
            self.log_info("First speaker audio received!")
        elif self.speaker_chunks_received % 10 == 0:  # Log more frequently
            self.log_info(f"Received {self.speaker_chunks_received} speaker chunks, size={len(audio_data)}")
        
        # Store original 24kHz audio for WebRTC
        self.speaker_buffer_24k.extend(audio_data)
        
        # For non-WebRTC methods, also store resampled version
        if self.speaker_sample_rate != self.mic_sample_rate:
            if self.use_simple_resampling:
                # Simple 3:2 decimation for 24kHz to 16kHz
                audio_data_16k = self._simple_resample_24k_to_16k(audio_data)
            else:
                # Resample using scipy (24k to 16k is 3:2 ratio)
                audio_data_16k = signal.resample_poly(audio_data, 2, 3)
            self.log_debug(f"Resampled speaker audio: {len(audio_data)} → {len(audio_data_16k)} samples")
        else:
            audio_data_16k = audio_data
            
        # Add resampled audio to 16kHz buffer for non-WebRTC methods
        self.speaker_buffer.extend(audio_data_16k)
        
        # Update state
        self.assistant_speaking = True
        self.last_speaker_time = time.time()
        
        # Log buffer state periodically
        if self.speaker_chunks_received % 10 == 0:
            self.log_debug(f"Speaker buffer size: {len(self.speaker_buffer)} samples "
                         f"({len(self.speaker_buffer)/self.mic_sample_rate:.1f}s)")
        
        # Publish AEC active status
        self.status_pub.publish(Bool(data=True))
            
    def mic_callback(self, msg):
        """Handle microphone input and perform echo cancellation"""
        # Convert to numpy array - AudioStamped has nested structure
        mic_data = np.array(msg.audio.audio_data.int16_data, dtype=np.int16)
        
        self.mic_chunks_processed += 1
        if self.mic_chunks_processed == 1:
            self.log_info("First microphone audio received!")
        
        # Add to mic buffer
        self.mic_buffer.extend(mic_data)
        
        # Log chunk size on first few chunks
        if self.mic_chunks_processed <= 5:
            self.log_info(f"Mic chunk #{self.mic_chunks_processed}: {len(mic_data)} samples")
        
        # Periodic status report
        current_time = time.time()
        if current_time - self.last_status_time > 5.0:  # Every 5 seconds
            self._log_status()
            self.last_status_time = current_time
        
        # Check if assistant has been speaking recently
        time_since_speaker = time.time() - self.last_speaker_time
        if time_since_speaker > 0.5:  # 500ms silence
            self.assistant_speaking = False
            self.status_pub.publish(Bool(data=False))
        
        # Check if bypass mode
        if self.bypass:
            processed = mic_data
            self.log_debug("Bypass mode - no AEC applied")
        # Only perform AEC if assistant is/was recently speaking
        elif self.assistant_speaking or time_since_speaker < 1.0:
            # Get aligned speaker reference (compensate for delay)
            # For WebRTC, we need to handle different sample rates properly
            if self.aec_type == 'webrtc':
                # WebRTC needs speaker data at original 24kHz sample rate
                # Calculate delay in 24kHz samples
                delay_samples_24k = int(self.delay_samples * self.speaker_sample_rate / self.mic_sample_rate)
                speaker_chunk_24k = int(len(mic_data) * self.speaker_sample_rate / self.mic_sample_rate)
                
                if len(self.speaker_buffer_24k) > delay_samples_24k + speaker_chunk_24k:
                    # Extract 24kHz speaker reference
                    start_idx = len(self.speaker_buffer_24k) - delay_samples_24k - speaker_chunk_24k
                    end_idx = len(self.speaker_buffer_24k) - delay_samples_24k
                    
                    if start_idx >= 0:
                        speaker_ref = np.array(list(self.speaker_buffer_24k))[start_idx:end_idx]
                        # Log when we start processing
                        if self.mic_chunks_processed <= 5 or self.mic_chunks_processed % 100 == 0:
                            self.log_debug(f"WebRTC processing: speaker_buf_24k={len(self.speaker_buffer_24k)}, "
                                         f"delay_24k={delay_samples_24k}, mic_len={len(mic_data)}, "
                                         f"speaker_ref_len={len(speaker_ref)} @ 24kHz")
                        
                        processed = self._process_webrtc(mic_data, speaker_ref)
                    else:
                        # Not enough history, skip AEC
                        processed = mic_data
                        self.log_debug("Not enough speaker buffer history for delay compensation")
                else:
                    # Not enough speaker data, skip AEC
                    processed = mic_data
                    if self.mic_chunks_processed % 50 == 0:
                        self.log_debug(f"Insufficient speaker buffer: {len(self.speaker_buffer_24k)} samples")
            else:
                # For non-WebRTC methods, use resampled 16kHz data
                if len(self.speaker_buffer) > self.delay_samples + len(mic_data):
                    # Extract speaker reference that corresponds to current mic input
                    # The mic hears what was played delay_samples ago
                    # So we need the speaker audio from delay_samples back
                    start_idx = len(self.speaker_buffer) - self.delay_samples - len(mic_data)
                    end_idx = len(self.speaker_buffer) - self.delay_samples
                    
                    if start_idx >= 0:
                        speaker_ref = np.array(list(self.speaker_buffer))[start_idx:end_idx]
                    else:
                        # Not enough history, use what we have
                        speaker_ref = np.zeros(len(mic_data), dtype=np.int16)
                else:
                    speaker_ref = np.zeros(len(mic_data), dtype=np.int16)
                
                # Log when we start processing
                if self.mic_chunks_processed <= 5 or self.mic_chunks_processed % 100 == 0:
                    self.log_debug(f"Processing with AEC: speaker_buf={len(self.speaker_buffer)}, "
                                 f"delay={self.delay_samples}, mic_len={len(mic_data)}, "
                                 f"speaker_ref_len={len(speaker_ref)}")
                
                # Perform echo cancellation based on initialized type
                if self.aec_type == 'speex':
                    processed = self._process_speex(mic_data, speaker_ref)
                else:
                    processed = self._process_adaptive(mic_data, speaker_ref)
            
            # Calculate echo reduction
            input_power = np.mean(mic_data**2)
            output_power = np.mean(processed**2)
            if input_power > 0 and output_power > 0:
                reduction_db = 10 * np.log10(output_power / input_power)
                if not np.isnan(reduction_db) and not np.isinf(reduction_db):
                    self.echo_reduction_sum += reduction_db
                    self.log_debug(f"AEC active: input={input_power:.0f}, output={output_power:.0f}, "
                                 f"reduction={reduction_db:.1f}dB")
                    
                    # Log if no reduction with Speex
                    if self.aec_type == 'speex' and abs(reduction_db) < 0.1:
                        if self.mic_chunks_processed % 50 == 0:  # Every 50 chunks
                            self.log_warning(f"Speex showing no reduction! Check initialization.")
            
            # Periodically measure and update delay
            self.delay_update_counter += 1
            if self.delay_update_counter % 50 == 0:  # Every 50 chunks
                # Estimate delay using recent audio
                if len(self.speaker_buffer) > self.delay_estimator.correlation_window:
                    speaker_array = np.array(list(self.speaker_buffer))[-self.delay_estimator.correlation_window:]
                    mic_array = np.array(list(self.mic_buffer))[-self.delay_estimator.correlation_window:]
                    
                    measured_delay = self.delay_estimator.estimate_delay(speaker_array, mic_array)
                    if measured_delay > 0:
                        self.measured_delays.append(measured_delay)
                        
                        # Update delay if we have enough measurements
                        if len(self.measured_delays) >= 5:
                            avg_delay = int(np.median(self.measured_delays))
                            if abs(avg_delay - self.delay_samples) > 80:  # More than 5ms difference
                                old_delay_ms = self.delay_samples * 1000 / self.mic_sample_rate
                                new_delay_ms = avg_delay * 1000 / self.mic_sample_rate
                                self.log_info(f"Updating delay: {old_delay_ms:.1f}ms -> {new_delay_ms:.1f}ms")
                                self.delay_samples = avg_delay
                                
                                # Update WebRTC delay if applicable
                                if self.aec_type == 'webrtc' and hasattr(self.aec, 'set_system_delay'):
                                    self.aec.set_system_delay(int(new_delay_ms))
        else:
            # No echo cancellation needed
            processed = mic_data
            
        # Publish filtered audio
        filtered_msg = AudioStamped()
        filtered_msg.header = msg.header
        # Copy the original audio message structure
        filtered_msg.audio = msg.audio
        # Update the audio data
        filtered_msg.audio.audio_data.int16_data = processed.astype(np.int16).tolist()
        self.audio_pub.publish(filtered_msg)
        
        # Test mode: detect tone in mic and filtered signals
        if self.test_mode and hasattr(self, 'tone_start_time'):
            time_since_tone = time.time() - self.tone_start_time
            if 0.1 < time_since_tone < self.test_tone_duration + 0.5:
                # Detect tone power in both signals
                mic_tone_power = self._detect_tone_in_signal(mic_data, self.mic_sample_rate)
                filtered_tone_power = self._detect_tone_in_signal(processed, self.mic_sample_rate)
                
                # First detection of tone in mic
                if self.tone_detected_time == 0 and mic_tone_power > -40:
                    self.tone_detected_time = time.time()
                    delay_ms = (self.tone_detected_time - self.tone_start_time) * 1000
                    self.log_info(f"Tone detected in mic after {delay_ms:.1f}ms")
                    
                # Log tone cancellation performance
                if mic_tone_power > -40 or self.debug_logging:
                    reduction = mic_tone_power - filtered_tone_power
                    # Also log the actual signal power to debug
                    mic_rms = np.sqrt(np.mean(mic_data**2))
                    self.log_info(f"Tone power - Mic: {mic_tone_power:.1f}dB (RMS: {mic_rms:.0f}), "
                                f"Filtered: {filtered_tone_power:.1f}dB, "
                                f"Reduction: {reduction:.1f}dB")
            elif time_since_tone > self.test_tone_duration + 0.5:
                # Reset for next tone
                self.tone_detected_time = 0
        
    def _process_webrtc(self, mic_data, speaker_ref):
        """Process with WebRTC AEC - requires 10ms chunks"""
        try:
            # Add new data to buffers
            self.webrtc_mic_buffer.extend(mic_data.tolist())
            # speaker_ref is now at 24kHz for WebRTC
            self.webrtc_speaker_buffer.extend(speaker_ref.tolist())
            
            # Log buffer state periodically
            if self.webrtc_chunks_processed % 100 == 0 or self.webrtc_chunks_processed < 5:
                self.log_debug(f"WebRTC buffers before processing: mic={len(self.webrtc_mic_buffer)}, "
                             f"speaker={len(self.webrtc_speaker_buffer)}")
            
            processed_output = []
            chunks_dropped = 0
            
            # Process in 10ms chunks - only when BOTH buffers have sufficient data
            while (len(self.webrtc_mic_buffer) >= self.webrtc_mic_frame_size and 
                   len(self.webrtc_speaker_buffer) >= self.webrtc_speaker_frame_size):
                
                # Extract mic chunk
                mic_chunk = np.array(self.webrtc_mic_buffer[:self.webrtc_mic_frame_size], dtype=np.int16)
                self.webrtc_mic_buffer = self.webrtc_mic_buffer[self.webrtc_mic_frame_size:]
                
                # Extract speaker chunk
                speaker_chunk = np.array(self.webrtc_speaker_buffer[:self.webrtc_speaker_frame_size], dtype=np.int16)
                self.webrtc_speaker_buffer = self.webrtc_speaker_buffer[self.webrtc_speaker_frame_size:]
                
                # Convert to bytes for WebRTC
                mic_bytes = mic_chunk.tobytes()
                speaker_bytes = speaker_chunk.tobytes()
                
                # Process far-end (speaker) signal first
                # Note: WebRTC API doesn't have set_delay method
                self.aec.process_reverse_stream(speaker_bytes)
                
                # Then process near-end (microphone) signal with AEC
                processed_bytes = self.aec.process_stream(mic_bytes)
                
                # Convert back to numpy array
                processed_chunk = np.frombuffer(processed_bytes, dtype=np.int16)
                processed_output.extend(processed_chunk.tolist())
                
                self.webrtc_chunks_processed += 1
                if self.webrtc_chunks_processed == 1:
                    self.log_info("First WebRTC chunk processed successfully!")
                    # Check if arrays are identical
                    if np.array_equal(mic_chunk, processed_chunk):
                        self.log_warning("WebRTC output is identical to input - no processing happening!")
                    # Check correlation between mic and speaker
                    if len(mic_chunk) == len(speaker_chunk) and np.any(speaker_chunk):
                        correlation = np.corrcoef(mic_chunk.astype(float), speaker_chunk.astype(float))[0,1]
                        self.log_info(f"Mic-Speaker correlation: {correlation:.3f}")
                    # Check if WebRTC detects echo
                    if hasattr(self.aec, 'has_echo'):
                        self.log_info(f"WebRTC echo detected: {self.aec.has_echo}")
                elif self.webrtc_chunks_processed % 100 == 0:
                    # Check echo reduction
                    mic_power = np.mean(mic_chunk**2)
                    proc_power = np.mean(processed_chunk**2)
                    speaker_power = np.mean(speaker_chunk**2)
                    if mic_power > 0:
                        reduction = 10 * np.log10(proc_power / mic_power)
                        self.log_debug(f"WebRTC chunk {self.webrtc_chunks_processed}: mic_pwr={mic_power:.0f}, "
                                     f"spk_pwr={speaker_power:.0f}, proc_pwr={proc_power:.0f}, "
                                     f"reduction={reduction:.1f}dB")
                    # Log AEC status
                    if hasattr(self.aec, 'has_echo') and self.aec.has_echo:
                        self.log_debug(f"WebRTC detecting echo, AEC level: {self.aec.aec_level}")
            
            # Check for buffer underrun (couldn't process all data)
            if len(self.webrtc_mic_buffer) >= self.webrtc_mic_frame_size:
                # We have mic data but no speaker data - this is an underrun
                self.webrtc_underruns += 1
                self.log_warning(f"Speaker buffer underrun #{self.webrtc_underruns}! "
                               f"Mic buffer: {len(self.webrtc_mic_buffer)}, "
                               f"Speaker buffer: {len(self.webrtc_speaker_buffer)}")
                # Clear buffers to prevent desynchronization
                self.webrtc_mic_buffer.clear()
                self.webrtc_speaker_buffer.clear()
                # Return original audio when AEC fails
                return mic_data
            
            # If we have processed data, return it; otherwise return original
            if processed_output:
                # Ensure output matches input length exactly
                if len(processed_output) != len(mic_data):
                    self.log_warning(f"Output length mismatch: {len(processed_output)} vs {len(mic_data)}")
                    # Pad or truncate as needed
                    if len(processed_output) < len(mic_data):
                        processed_output.extend([0] * (len(mic_data) - len(processed_output)))
                    else:
                        processed_output = processed_output[:len(mic_data)]
                return np.array(processed_output, dtype=np.int16)
            else:
                # Not enough data for a full 10ms chunk yet
                return mic_data
                
        except Exception as e:
            self.log_error(f"WebRTC processing error: {e}")
            # Initialize adaptive filter if not already done
            if self.aec_type == 'webrtc':
                self.log_warning("Falling back to adaptive filter")
                self._init_adaptive_aec()
            # Fall back to adaptive filter
            return self._process_adaptive(mic_data, speaker_ref)
        
    def _process_speex(self, mic_data, speaker_ref):
        """Process with Speex AEC"""
        try:
            # Speex expects raw bytes (int16 audio data)
            # Ensure we have matching lengths
            min_len = min(len(mic_data), len(speaker_ref))
            if min_len < self.frame_size:
                self.log_debug(f"Not enough data for Speex: {min_len} < {self.frame_size}")
                return mic_data
                
            # Process in frame_size chunks
            processed_output = []
            
            # Process multiple frames if we have enough data
            num_frames = min_len // self.frame_size
            
            # Log first time we process with Speex
            if not hasattr(self, '_speex_processed_count'):
                self._speex_processed_count = 0
            
            for i in range(num_frames):
                start_idx = i * self.frame_size
                end_idx = start_idx + self.frame_size
                
                # Get frame chunks
                mic_frame = mic_data[start_idx:end_idx]
                speaker_frame = speaker_ref[start_idx:end_idx]
                
                # Convert to bytes for Speex
                mic_bytes = mic_frame.astype(np.int16).tobytes()
                speaker_bytes = speaker_frame.astype(np.int16).tobytes()
                
                # Process with Speex
                processed_bytes = self.aec.process(mic_bytes, speaker_bytes)
                
                # Convert back to numpy array
                processed_frame = np.frombuffer(processed_bytes, dtype=np.int16)
                processed_output.extend(processed_frame)
                
                self._speex_processed_count += 1
                if self._speex_processed_count == 1:
                    self.log_info("First Speex frame processed successfully!")
                    # Check if arrays are identical
                    if np.array_equal(mic_frame, processed_frame):
                        self.log_warning("Speex output is identical to input - no processing happening!")
                    # Check correlation between mic and speaker
                    if len(mic_frame) == len(speaker_frame):
                        correlation = np.corrcoef(mic_frame.astype(float), speaker_frame.astype(float))[0,1]
                        self.log_info(f"Mic-Speaker correlation: {correlation:.3f}")
                elif self._speex_processed_count % 100 == 0:
                    # Check if there's any difference
                    mic_power = np.mean(mic_frame**2)
                    proc_power = np.mean(processed_frame**2)
                    speaker_power = np.mean(speaker_frame**2)
                    if mic_power > 0:
                        reduction = 10 * np.log10(proc_power / mic_power)
                        self.log_debug(f"Speex frame {self._speex_processed_count}: mic_pwr={mic_power:.0f}, "
                                     f"spk_pwr={speaker_power:.0f}, proc_pwr={proc_power:.0f}, "
                                     f"reduction={reduction:.1f}dB")
                
            # If we have any unprocessed samples at the end, pass them through
            if len(processed_output) < len(mic_data):
                processed_output.extend(mic_data[len(processed_output):])
                
            return np.array(processed_output, dtype=np.int16)
        except Exception as e:
            self.log_error(f"Speex processing error: {e}")
            return mic_data
        
    def _process_adaptive(self, mic_data, speaker_ref):
        """Process with adaptive filter"""
        return self.aec.process(mic_data, speaker_ref)
    
    def _simple_resample_24k_to_16k(self, audio_data):
        """Simple 3:2 decimation from 24kHz to 16kHz
        
        This is a basic linear interpolation resampler.
        For every 3 samples at 24kHz, we output 2 samples at 16kHz.
        """
        output_length = len(audio_data) * 2 // 3
        output = np.zeros(output_length, dtype=np.int16)
        
        for i in range(output_length):
            # Map output sample index to input sample position
            input_pos = i * 1.5
            input_idx = int(input_pos)
            fraction = input_pos - input_idx
            
            if input_idx + 1 < len(audio_data):
                # Linear interpolation between two samples
                sample1 = audio_data[input_idx]
                sample2 = audio_data[input_idx + 1]
                output[i] = int(sample1 * (1 - fraction) + sample2 * fraction)
            elif input_idx < len(audio_data):
                output[i] = audio_data[input_idx]
                
        return output
    
    def _test_tone_callback(self):
        """Generate test tones periodically"""
        current_time = time.time()
        
        # First callback - log that timer is working
        if not hasattr(self, '_timer_logged'):
            self._timer_logged = True
            self.log_info("Test tone timer is running")
        
        # Check if it's time to generate a new tone
        if current_time - self.last_tone_time >= self.test_tone_interval:
            self.last_tone_time = current_time
            self.tone_start_time = current_time
            
            # Generate 1kHz sine wave at 24kHz sample rate
            duration_samples = int(self.test_tone_duration * self.speaker_sample_rate)
            t = np.arange(duration_samples) / self.speaker_sample_rate
            tone = (np.sin(2 * np.pi * self.test_tone_freq * t) * 30000).astype(np.int16)  # Louder tone
            
            self.log_info(f"Generating {self.test_tone_freq}Hz test tone: {duration_samples} samples")
            
            # Publish tone
            msg = AudioData()
            msg.int16_data = tone.tolist()
            self.tone_pub.publish(msg)
            
            self.log_info(f"Published test tone to test_tone_out")
    
    def _detect_tone_in_signal(self, signal, sample_rate):
        """Detect presence and power of test tone in signal"""
        if len(signal) < sample_rate * 0.01:  # Need at least 10ms
            return 0.0
            
        # Use FFT to find power at test frequency
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # Find bin closest to test frequency
        tone_bin = np.argmin(np.abs(freqs - self.test_tone_freq))
        
        # Calculate power at tone frequency (in dB)
        tone_power = np.abs(fft[tone_bin]) / len(signal)
        if tone_power > 0:
            return 20 * np.log10(tone_power)
        return -100  # Very low power
    
    def _log_status(self):
        """Log periodic status report"""
        avg_reduction = 0.0
        if self.mic_chunks_processed > 0:
            avg_reduction = self.echo_reduction_sum / self.mic_chunks_processed
            
        # Get delay statistics
        delay_info = ""
        if self.measured_delays:
            delays_ms = [d * 1000 / self.mic_sample_rate for d in self.measured_delays]
            avg_delay_ms = np.mean(delays_ms)
            std_delay_ms = np.std(delays_ms)
            min_delay_ms = np.min(delays_ms)
            max_delay_ms = np.max(delays_ms)
            delay_info = f"\nDelay measurements: {avg_delay_ms:.1f}±{std_delay_ms:.1f}ms (range: {min_delay_ms:.1f}-{max_delay_ms:.1f}ms)"
            
        self.log_info("=== AEC Status Report ===")
        self.log_info(f"AEC Type: {self.aec_type}")
        self.log_info(f"Mic chunks: {self.mic_chunks_processed}")
        self.log_info(f"Speaker chunks: {self.speaker_chunks_received}")
        self.log_info(f"Assistant speaking: {self.assistant_speaking}")
        self.log_info(f"Average echo reduction: {avg_reduction:.1f}dB")
        self.log_info(f"Current delay: {self.delay_samples * 1000 / self.mic_sample_rate:.1f}ms{delay_info}")
        
        if self.aec_type == 'webrtc':
            self.log_info(f"WebRTC chunks processed: {self.webrtc_chunks_processed}")
            self.log_info(f"WebRTC underruns: {self.webrtc_underruns}")
            self.log_info(f"WebRTC buffer sizes: mic={len(self.webrtc_mic_buffer)}, "
                         f"speaker={len(self.webrtc_speaker_buffer)}")
        
        self.log_info("========================")


class AdaptiveEchoCanceller:
    """Custom adaptive echo canceller using NLMS algorithm"""
    
    def __init__(self, filter_length, frame_size, sample_rate, learning_rate=0.3):
        self.filter_length = filter_length
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        
        # Adaptive filter coefficients
        self.filter_coeffs = np.zeros(filter_length)
        
        # Reference signal buffer
        self.ref_buffer = np.zeros(filter_length)
        
        # Regularization to prevent division by zero
        self.regularization = 1e-6
        
        # Adaptation control
        self.adaptation_enabled = True
        self.double_talk_threshold = 0.7
        
    def process(self, mic_signal, speaker_signal):
        """Cancel echo from microphone signal"""
        # Update reference buffer with speaker signal
        self.ref_buffer = np.roll(self.ref_buffer, -len(speaker_signal))
        self.ref_buffer[-len(speaker_signal):] = speaker_signal
        
        # Estimate echo using convolution with filter
        echo_estimate = np.zeros_like(mic_signal)
        for i in range(len(mic_signal)):
            if i < len(self.ref_buffer) - self.filter_length:
                ref_segment = self.ref_buffer[i:i+self.filter_length]
                echo_estimate[i] = np.dot(ref_segment, self.filter_coeffs)
        
        # Calculate error (echo-cancelled signal)
        error = mic_signal - echo_estimate
        
        # Double-talk detection (when user speaks while assistant speaks)
        if len(speaker_signal) > 0:
            mic_power = np.mean(mic_signal**2)
            echo_power = np.mean(echo_estimate**2)
            if echo_power > 0:
                ser = mic_power / echo_power  # Signal to Echo Ratio
                self.adaptation_enabled = ser < self.double_talk_threshold
        
        # Update filter coefficients using NLMS if no double-talk
        if self.adaptation_enabled and len(speaker_signal) > 0:
            for i in range(0, len(error), 10):  # Update every 10 samples for efficiency
                if i < len(self.ref_buffer) - self.filter_length:
                    ref_vec = self.ref_buffer[i:i+self.filter_length]
                    
                    # Normalized LMS update
                    norm_factor = np.dot(ref_vec, ref_vec) + self.regularization
                    update = (self.learning_rate * error[i] * ref_vec) / norm_factor
                    self.filter_coeffs += update
                    
                    # Constrain filter coefficients
                    self.filter_coeffs = np.clip(self.filter_coeffs, -1.0, 1.0)
                
        return error.astype(np.int16)


class DelayEstimator:
    """Estimate delay between speaker and microphone paths"""
    
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.max_delay_ms = 300  # Maximum expected delay in milliseconds (increased for safety)
        self.max_delay_samples = int(self.max_delay_ms * sample_rate / 1000)
        self.correlation_window = int(0.2 * sample_rate)  # 200ms window for better accuracy
        
    def estimate_delay(self, speaker_signal, mic_signal):
        """Estimate delay using cross-correlation"""
        if len(speaker_signal) < self.correlation_window or len(mic_signal) < self.correlation_window:
            return 0
            
        # Use recent portions of the signals
        speaker_segment = speaker_signal[-self.correlation_window:]
        mic_segment = mic_signal[-self.correlation_window:]
        
        # Check if there's enough signal power
        speaker_power = np.mean(speaker_segment**2)
        mic_power = np.mean(mic_segment**2)
        if speaker_power < 100 or mic_power < 100:  # Too quiet
            return 0
        
        # Normalize signals
        speaker_norm = (speaker_segment - np.mean(speaker_segment)) / (np.std(speaker_segment) + 1e-10)
        mic_norm = (mic_segment - np.mean(mic_segment)) / (np.std(mic_segment) + 1e-10)
        
        # Compute cross-correlation for positive delays only
        max_lag = min(self.max_delay_samples, len(speaker_norm))
        correlation = np.zeros(max_lag)
        
        # Use scipy for efficient correlation if available
        try:
            if SCIPY_AVAILABLE:
                # Full correlation, then take only positive lags
                full_corr = signal.correlate(mic_norm, speaker_norm, mode='full')
                # Find the center (zero lag) and take positive delays
                mid = len(full_corr) // 2
                correlation = full_corr[mid:mid+max_lag]
            else:
                raise ImportError("scipy not available")
        except (ImportError, AttributeError):
            # Fallback to manual correlation
            for lag in range(max_lag):
                if lag < len(mic_norm):
                    correlation[lag] = np.dot(
                        speaker_norm[:len(speaker_norm)-lag],
                        mic_norm[lag:]
                    ) / (len(speaker_norm) - lag)
        
        # Find peak with minimum threshold
        if np.max(correlation) > 0.1:  # Minimum correlation threshold
            delay = np.argmax(correlation)
            # Sanity check - delays should typically be 50-200ms
            expected_min = int(0.05 * self.sample_rate)  # 50ms
            expected_max = int(0.25 * self.sample_rate)  # 250ms
            if expected_min <= delay <= expected_max:
                return int(delay)
        
        return 0  # No valid delay found


def main(args=None):
    rclpy.init(args=args)
    node = AcousticEchoCancellerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()