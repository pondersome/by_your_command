"""
Pause Detection for Session Cycling

Monitors message timing from bridge interface to detect conversation pauses
and trigger aggressive session cycling for cost optimization.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import time
import logging
from typing import Optional, Dict


class PauseDetector:
    """Monitor for pause conditions to trigger session cycling"""
    
    def __init__(self, pause_timeout: float = 10.0):
        self.pause_timeout = pause_timeout
        self.last_message_time: Optional[float] = None
        self.llm_response_complete = True
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.pauses_detected = 0
        self.total_pause_time = 0.0
        self.last_pause_duration = 0.0
        
    def reset(self):
        """Reset pause detection state"""
        self.last_message_time = None
        self.llm_response_complete = True
        self.logger.debug("Pause detector reset")
        
    def record_message(self, message_type: str = "unknown"):
        """Record that we received a message"""
        current_time = time.time()
        
        # Calculate pause duration if we had a previous message
        if self.last_message_time:
            pause_duration = current_time - self.last_message_time
            if pause_duration > 1.0:  # Log pauses > 1 second
                self.logger.debug(f"Message gap: {pause_duration:.1f}s (type: {message_type})")
                
        self.last_message_time = current_time
        
    def mark_llm_response_complete(self):
        """Called when LLM finishes speaking"""
        self.llm_response_complete = True
        self.logger.debug("LLM response marked complete")
        
    def mark_llm_response_active(self):
        """Called when LLM starts speaking"""
        self.llm_response_complete = False
        self.logger.debug("LLM response marked active")
        
    def check_pause_condition(self) -> bool:
        """Check if pause conditions are met for session cycling"""
        if not self.last_message_time or not self.llm_response_complete:
            return False
            
        time_since_last = time.time() - self.last_message_time
        is_paused = time_since_last > self.pause_timeout
        
        if is_paused:
            self.last_pause_duration = time_since_last
            self.total_pause_time += time_since_last
            self.pauses_detected += 1
            self.logger.info(f"Pause detected: {time_since_last:.1f}s since last message")
            
        return is_paused
    
    def get_time_since_last_message(self) -> Optional[float]:
        """Get seconds since last message, or None if no messages"""
        if not self.last_message_time:
            return None
        return time.time() - self.last_message_time
    
    def is_llm_speaking(self) -> bool:
        """Check if LLM is currently speaking"""
        return not self.llm_response_complete
    
    def update_pause_timeout(self, new_timeout: float):
        """Update pause timeout duration"""
        old_timeout = self.pause_timeout
        self.pause_timeout = new_timeout
        self.logger.info(f"Pause timeout updated: {old_timeout:.1f}s → {new_timeout:.1f}s")
        
    def get_metrics(self) -> Dict:
        """Get pause detection metrics"""
        return {
            'pause_timeout': self.pause_timeout,
            'pauses_detected': self.pauses_detected,
            'total_pause_time': self.total_pause_time,
            'last_pause_duration': self.last_pause_duration,
            'time_since_last_message': self.get_time_since_last_message(),
            'llm_speaking': self.is_llm_speaking()
        }
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary"""
        if not self.last_message_time:
            return "No messages received"
            
        time_since = self.get_time_since_last_message()
        llm_status = "speaking" if self.is_llm_speaking() else "idle"
        
        return f"Last message: {time_since:.1f}s ago, LLM: {llm_status}, Pauses: {self.pauses_detected}"