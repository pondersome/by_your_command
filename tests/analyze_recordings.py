#!/usr/bin/env python3
"""
Analyze recorded audio samples
Visualize waveforms and extract features
"""

import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob
from pathlib import Path

def load_wav(filepath):
    """Load WAV file and return audio data and sample rate"""
    with wave.open(filepath, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(num_frames)
        
    # Convert to numpy array
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    
    return audio_float, sample_rate

def analyze_waveform(audio, sample_rate, title=""):
    """Analyze and return waveform characteristics"""
    # Calculate envelope
    envelope = np.abs(audio)
    
    # Smooth envelope
    window_size = int(sample_rate * 0.001)  # 1ms smoothing
    if len(envelope) > window_size:
        envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
    else:
        envelope_smooth = envelope
    
    # Find peak
    peak_idx = np.argmax(envelope_smooth)
    peak_value = envelope_smooth[peak_idx]
    
    # Find onset (10% of peak, looking backwards from peak)
    onset_threshold = peak_value * 0.1
    onset_idx = 0
    for i in range(peak_idx, 0, -1):
        if envelope_smooth[i] < onset_threshold:
            onset_idx = i
            break
    
    # Find offset (10% of peak, looking forward from peak)
    offset_idx = len(envelope_smooth) - 1
    for i in range(peak_idx, len(envelope_smooth)):
        if envelope_smooth[i] < onset_threshold:
            offset_idx = i
            break
    
    # Calculate timings
    rise_time_ms = (peak_idx - onset_idx) / sample_rate * 1000
    fall_time_ms = (offset_idx - peak_idx) / sample_rate * 1000
    duration_ms = (offset_idx - onset_idx) / sample_rate * 1000
    
    # Calculate RMS
    rms = np.sqrt(np.mean(audio**2))
    
    # Calculate zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
    zcr = zero_crossings / len(audio)
    
    # Calculate spectral centroid
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    magnitude = np.abs(fft)
    
    if np.sum(magnitude) > 0:
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    else:
        spectral_centroid = 0
    
    # Calculate spectral flatness
    power = magnitude**2
    power = power[power > 1e-10]
    if len(power) > 0:
        geometric_mean = np.exp(np.mean(np.log(power)))
        arithmetic_mean = np.mean(power)
        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
    else:
        spectral_flatness = 0
    
    return {
        'peak': peak_value,
        'peak_idx': peak_idx,
        'onset_idx': onset_idx,
        'offset_idx': offset_idx,
        'rise_time_ms': rise_time_ms,
        'fall_time_ms': fall_time_ms,
        'duration_ms': duration_ms,
        'rms': rms,
        'zcr': zcr,
        'spectral_centroid': spectral_centroid,
        'spectral_flatness': spectral_flatness,
        'envelope': envelope_smooth
    }

def plot_comparison(sample_dir="/tmp/audio_samples"):
    """Plot comparison of different sound types"""
    
    # Load samples
    clap_files = sorted(glob.glob(os.path.join(sample_dir, "clap_*.wav")))[:3]
    speech_files = sorted(glob.glob(os.path.join(sample_dir, "speech_*.wav")))[:3]
    double_files = sorted(glob.glob(os.path.join(sample_dir, "double_*.wav")))[:2]
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Audio Sample Analysis - Reverberant Room', fontsize=14)
    
    # Plot claps
    for i, filepath in enumerate(clap_files):
        if i >= 3:
            break
        audio, sr = load_wav(filepath)
        features = analyze_waveform(audio, sr)
        
        ax = axes[0, i]
        time = np.arange(len(audio)) / sr
        
        # Plot waveform
        ax.plot(time, audio, 'b-', alpha=0.6, linewidth=0.5, label='Waveform')
        ax.plot(time, features['envelope'], 'r-', linewidth=1.5, label='Envelope')
        
        # Mark onset, peak, offset
        ax.axvline(features['onset_idx']/sr, color='g', linestyle='--', alpha=0.5, label='Onset')
        ax.axvline(features['peak_idx']/sr, color='m', linestyle='--', alpha=0.5, label='Peak')
        ax.axvline(features['offset_idx']/sr, color='orange', linestyle='--', alpha=0.5, label='Offset')
        
        ax.set_title(f"Clap {i+1}\nRise: {features['rise_time_ms']:.1f}ms, "
                     f"Fall: {features['fall_time_ms']:.1f}ms\n"
                     f"ZCR: {features['zcr']:.3f}, Flatness: {features['spectral_flatness']:.3f}")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Plot speech
    for i, filepath in enumerate(speech_files):
        if i >= 3:
            break
        audio, sr = load_wav(filepath)
        features = analyze_waveform(audio, sr)
        
        ax = axes[1, i]
        time = np.arange(len(audio)) / sr
        
        ax.plot(time, audio, 'b-', alpha=0.6, linewidth=0.5)
        ax.plot(time, features['envelope'], 'r-', linewidth=1.5)
        
        ax.axvline(features['onset_idx']/sr, color='g', linestyle='--', alpha=0.5)
        ax.axvline(features['peak_idx']/sr, color='m', linestyle='--', alpha=0.5)
        ax.axvline(features['offset_idx']/sr, color='orange', linestyle='--', alpha=0.5)
        
        ax.set_title(f"Speech {i+1}\nRise: {features['rise_time_ms']:.1f}ms, "
                     f"Fall: {features['fall_time_ms']:.1f}ms\n"
                     f"ZCR: {features['zcr']:.3f}, Flatness: {features['spectral_flatness']:.3f}")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    # Plot double claps
    for i, filepath in enumerate(double_files):
        if i >= 2:
            break
        audio, sr = load_wav(filepath)
        features = analyze_waveform(audio, sr)
        
        ax = axes[2, i]
        time = np.arange(len(audio)) / sr
        
        ax.plot(time, audio, 'b-', alpha=0.6, linewidth=0.5)
        ax.plot(time, features['envelope'], 'r-', linewidth=1.5)
        
        ax.set_title(f"Double Clap {i+1}\n"
                     f"Peak: {features['peak']:.3f}, Duration: {features['duration_ms']:.1f}ms\n"
                     f"ZCR: {features['zcr']:.3f}, Flatness: {features['spectral_flatness']:.3f}")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    # Spectrogram of one clap
    if clap_files:
        audio, sr = load_wav(clap_files[0])
        ax = axes[2, 2]
        
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=256)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Clap Spectrogram\n(showing reverb)')
        ax.set_ylim(0, 4000)
    
    plt.tight_layout()
    plt.savefig('/tmp/audio_analysis.png', dpi=150)
    print("Saved plot to /tmp/audio_analysis.png")
    # plt.show()  # Commented out for headless operation

def print_statistics(sample_dir="/tmp/audio_samples"):
    """Print statistics for all samples"""
    
    print("\n" + "="*60)
    print("AUDIO SAMPLE STATISTICS")
    print("="*60)
    
    for pattern in ["clap_*.wav", "speech_*.wav", "double_*.wav"]:
        files = sorted(glob.glob(os.path.join(sample_dir, pattern)))
        
        if not files:
            continue
            
        sample_type = pattern.split('_')[0].upper()
        print(f"\n{sample_type} SAMPLES ({len(files)} files):")
        print("-" * 40)
        
        all_features = []
        for filepath in files:
            audio, sr = load_wav(filepath)
            features = analyze_waveform(audio, sr)
            all_features.append(features)
            
            filename = os.path.basename(filepath)
            print(f"{filename:30s} | Peak: {features['peak']:5.3f} | "
                  f"Rise: {features['rise_time_ms']:6.1f}ms | "
                  f"Fall: {features['fall_time_ms']:6.1f}ms | "
                  f"ZCR: {features['zcr']:.3f} | "
                  f"Flat: {features['spectral_flatness']:.3f}")
        
        # Calculate averages
        if all_features:
            avg_peak = np.mean([f['peak'] for f in all_features])
            avg_rise = np.mean([f['rise_time_ms'] for f in all_features])
            avg_fall = np.mean([f['fall_time_ms'] for f in all_features])
            avg_zcr = np.mean([f['zcr'] for f in all_features])
            avg_flat = np.mean([f['spectral_flatness'] for f in all_features])
            avg_centroid = np.mean([f['spectral_centroid'] for f in all_features])
            
            print(f"\nAVERAGES:")
            print(f"  Peak amplitude: {avg_peak:.3f}")
            print(f"  Rise time: {avg_rise:.1f}ms (std: {np.std([f['rise_time_ms'] for f in all_features]):.1f})")
            print(f"  Fall time: {avg_fall:.1f}ms (std: {np.std([f['fall_time_ms'] for f in all_features]):.1f})")
            print(f"  Zero crossing rate: {avg_zcr:.3f}")
            print(f"  Spectral flatness: {avg_flat:.3f}")
            print(f"  Spectral centroid: {avg_centroid:.1f} Hz")
    
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("-" * 40)
    print("1. Rise times show attack characteristics")
    print("2. Fall times show reverb decay in the room")
    print("3. ZCR indicates frequency content")
    print("4. Spectral flatness shows tonality")
    print("5. Multiple peaks in double claps should be visible")
    print("="*60)

if __name__ == "__main__":
    # Analyze samples
    print_statistics()
    
    # Create visualization
    try:
        plot_comparison()
    except Exception as e:
        print(f"Could not create plot: {e}")
        print("Install matplotlib if needed: pip install matplotlib")