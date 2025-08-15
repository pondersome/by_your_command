#!/usr/bin/env python3
"""
Serializers for Gemini Live Agent

Handle conversion between ROS message formats and Gemini Live API formats.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import base64
import json
import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np
from dataclasses import dataclass

# Try to import ROS message types
try:
    from sensor_msgs.msg import Image, CompressedImage
    from audio_common_msgs.msg import AudioData
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    

@dataclass
class SerializedAudio:
    """Container for serialized audio data"""
    data: str  # Base64 encoded audio
    mime_type: str
    sample_rate: int
    channels: int = 1
    

@dataclass 
class SerializedImage:
    """Container for serialized image data"""
    data: str  # Base64 encoded image
    mime_type: str
    width: int
    height: int
    

class GeminiSerializer:
    """
    Serialize ROS messages to Gemini Live API format.
    
    Handles:
    - Audio data (PCM, various sample rates)
    - Images (RGB, compressed)
    - Text messages
    - Command responses
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def serialize_audio(self, 
                       audio_data: Union[Any, Dict, List], 
                       sample_rate: int = 16000) -> Optional[SerializedAudio]:
        """
        Serialize audio data to Gemini format.
        
        Args:
            audio_data: ROS AudioData message, dict, or raw PCM data
            sample_rate: Audio sample rate in Hz
            
        Returns:
            SerializedAudio object or None if serialization fails
        """
        try:
            pcm_bytes = None
            
            if ROS_AVAILABLE and isinstance(audio_data, AudioData):
                # ROS AudioData message
                if audio_data.int16_data:
                    pcm_bytes = np.array(audio_data.int16_data, dtype=np.int16).tobytes()
                elif audio_data.uint8_data:
                    # Convert uint8 to int16
                    uint8_array = np.array(audio_data.uint8_data, dtype=np.uint8)
                    int16_array = uint8_array.astype(np.int16) - 128
                    pcm_bytes = int16_array.tobytes()
                    
            elif isinstance(audio_data, dict):
                # Dictionary representation
                if 'int16_data' in audio_data:
                    pcm_bytes = np.array(audio_data['int16_data'], dtype=np.int16).tobytes()
                elif 'float32_data' in audio_data:
                    # Convert float32 to int16
                    float_array = np.array(audio_data['float32_data'], dtype=np.float32)
                    int16_array = (float_array * 32767).astype(np.int16)
                    pcm_bytes = int16_array.tobytes()
                    
            elif isinstance(audio_data, (list, np.ndarray)):
                # Raw array data
                if isinstance(audio_data, list):
                    audio_data = np.array(audio_data)
                    
                if audio_data.dtype == np.int16:
                    pcm_bytes = audio_data.tobytes()
                elif audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    # Convert float to int16
                    int16_array = (audio_data * 32767).astype(np.int16)
                    pcm_bytes = int16_array.tobytes()
                else:
                    # Try to convert to int16
                    pcm_bytes = audio_data.astype(np.int16).tobytes()
                    
            elif isinstance(audio_data, bytes):
                # Already bytes
                pcm_bytes = audio_data
                
            if pcm_bytes:
                # Encode to base64
                audio_base64 = base64.b64encode(pcm_bytes).decode('utf-8')
                
                return SerializedAudio(
                    data=audio_base64,
                    mime_type=f"audio/pcm;rate={sample_rate}",
                    sample_rate=sample_rate,
                    channels=1
                )
                
        except Exception as e:
            self.logger.error(f"Error serializing audio: {e}")
            
        return None
        
    def serialize_image(self, 
                       image_data: Union[Any, Dict, np.ndarray],
                       encoding: str = "rgb8") -> Optional[SerializedImage]:
        """
        Serialize image data to Gemini format.
        
        Args:
            image_data: ROS Image message, dict, or numpy array
            encoding: Image encoding (rgb8, bgr8, jpeg, etc.)
            
        Returns:
            SerializedImage object or None if serialization fails
        """
        try:
            import cv2
            
            img_array = None
            width = height = 0
            
            if ROS_AVAILABLE and isinstance(image_data, Image):
                # ROS Image message
                width = image_data.width
                height = image_data.height
                
                # Convert to numpy array
                if image_data.encoding == "rgb8":
                    img_array = np.frombuffer(image_data.data, dtype=np.uint8)
                    img_array = img_array.reshape((height, width, 3))
                elif image_data.encoding == "bgr8":
                    img_array = np.frombuffer(image_data.data, dtype=np.uint8)
                    img_array = img_array.reshape((height, width, 3))
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                elif image_data.encoding == "mono8":
                    img_array = np.frombuffer(image_data.data, dtype=np.uint8)
                    img_array = img_array.reshape((height, width))
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    
            elif ROS_AVAILABLE and isinstance(image_data, CompressedImage):
                # Compressed image - already in JPEG/PNG format
                return SerializedImage(
                    data=base64.b64encode(image_data.data).decode('utf-8'),
                    mime_type=f"image/{image_data.format}",
                    width=0,  # Unknown from compressed data
                    height=0
                )
                
            elif isinstance(image_data, dict):
                # Dictionary representation
                width = image_data.get('width', 0)
                height = image_data.get('height', 0)
                
                if 'data' in image_data:
                    data = image_data['data']
                    if isinstance(data, str):
                        # Already base64 encoded
                        return SerializedImage(
                            data=data,
                            mime_type="image/jpeg",
                            width=width,
                            height=height
                        )
                    else:
                        # Raw pixel data
                        img_array = np.array(data, dtype=np.uint8)
                        if len(img_array.shape) == 1:
                            # Reshape based on encoding
                            channels = 3 if encoding in ["rgb8", "bgr8"] else 1
                            img_array = img_array.reshape((height, width, channels))
                            
            elif isinstance(image_data, np.ndarray):
                # Numpy array
                img_array = image_data
                if len(img_array.shape) == 2:
                    height, width = img_array.shape
                    # Convert grayscale to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif len(img_array.shape) == 3:
                    height, width = img_array.shape[:2]
                    
            if img_array is not None:
                # Encode as JPEG
                success, buffer = cv2.imencode('.jpg', img_array)
                if success:
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    return SerializedImage(
                        data=img_base64,
                        mime_type="image/jpeg",
                        width=width,
                        height=height
                    )
                    
        except Exception as e:
            self.logger.error(f"Error serializing image: {e}")
            
        return None
        
    def serialize_text(self, text_data: Union[Any, Dict, str]) -> Optional[str]:
        """
        Serialize text data.
        
        Args:
            text_data: ROS String message, dict, or raw string
            
        Returns:
            Text string or None
        """
        try:
            if ROS_AVAILABLE and isinstance(text_data, String):
                return text_data.data
            elif isinstance(text_data, dict):
                return text_data.get('data', '')
            elif isinstance(text_data, str):
                return text_data
            else:
                return str(text_data)
                
        except Exception as e:
            self.logger.error(f"Error serializing text: {e}")
            return None
            
    def deserialize_audio(self, 
                         audio_base64: str,
                         sample_rate: int = 24000) -> Optional[Dict]:
        """
        Deserialize audio from Gemini format to ROS format.
        
        Args:
            audio_base64: Base64 encoded audio data
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary compatible with ROS AudioData message
        """
        try:
            # Decode base64
            pcm_bytes = base64.b64decode(audio_base64)
            
            # Convert to int16 array
            int16_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            
            return {
                'int16_data': int16_array.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error deserializing audio: {e}")
            return None
            
    def serialize_scene_analysis(self, analysis: Dict) -> str:
        """
        Serialize scene analysis to JSON string.
        
        Args:
            analysis: Scene analysis dictionary
            
        Returns:
            JSON string
        """
        try:
            return json.dumps(analysis, indent=2)
        except Exception as e:
            self.logger.error(f"Error serializing scene analysis: {e}")
            return "{}"
            
    def deserialize_command(self, command_text: str) -> Optional[Dict]:
        """
        Parse command text into structured format.
        
        Args:
            command_text: Command string (e.g., "move@forward")
            
        Returns:
            Structured command dictionary
        """
        try:
            # Check for @ separator
            if '@' in command_text:
                action, parameter = command_text.split('@', 1)
                return {
                    'action': action.strip(),
                    'parameter': parameter.strip(),
                    'raw': command_text
                }
            else:
                return {
                    'action': command_text.strip(),
                    'parameter': None,
                    'raw': command_text
                }
                
        except Exception as e:
            self.logger.error(f"Error deserializing command: {e}")
            return None