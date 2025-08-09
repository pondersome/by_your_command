"""
Conversation Lifecycle Monitor

Tracks conversation boundaries through timeout detection and external signals.
Generates timestamp-based conversation IDs for logging and correlation.

Author: Karim Virani  
Version: 1.0
Date: August 2025
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any


class ConversationMonitor:
    """
    Monitors conversation lifecycle and manages conversation IDs.
    
    Features:
    - Generates timestamp-based conversation IDs
    - Detects conversation timeout from voice chunk activity
    - Publishes conversation ID changes
    - Handles external conversation resets
    """
    
    def __init__(self, timeout: float = 600.0, 
                 on_conversation_change: Optional[Callable[[str, str, bool], None]] = None):
        """
        Initialize conversation monitor.
        
        Args:
            timeout: Conversation timeout in seconds (default 600s = 10 minutes)
            on_conversation_change: Callback when conversation ID changes (old_id, new_id, is_external)
        """
        self.timeout = timeout
        self.on_conversation_change = on_conversation_change
        self.logger = logging.getLogger(__name__)
        
        # Conversation state
        self.current_conversation_id = self._generate_conversation_id()
        self.conversation_start_time = time.time()
        self.last_activity_time = time.time()
        
        # Monitoring state
        self.monitoring_active = True
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.conversations_started = 1
        self.timeouts_detected = 0
        self.external_resets = 0
        
        self.logger.info(f"🎭 Conversation monitor initialized - ID: {self.current_conversation_id}, timeout: {timeout}s")
        
    def _generate_conversation_id(self) -> str:
        """
        Generate timestamp-based conversation ID.
        
        Format: conv_YYYYMMDD_HHMMSS_microseconds
        Example: conv_20250804_143052_123456
        """
        now = datetime.now()
        return f"conv_{now.strftime('%Y%m%d_%H%M%S_%f')}"
        
    def record_activity(self):
        """Record voice chunk activity to reset timeout."""
        self.last_activity_time = time.time()
        
    def get_time_since_activity(self) -> float:
        """Get seconds since last voice chunk activity."""
        return time.time() - self.last_activity_time
        
    def check_timeout(self) -> bool:
        """Check if conversation has timed out."""
        return self.get_time_since_activity() > self.timeout
        
    async def start_monitoring(self):
        """Start async monitoring for conversation timeout."""
        if self._monitor_task and not self._monitor_task.done():
            self.logger.warning("Monitor already running")
            return
            
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("🔄 Started conversation timeout monitoring")
        
    async def stop_monitoring(self):
        """Stop timeout monitoring."""
        self.monitoring_active = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("🛑 Stopped conversation timeout monitoring")
        
    async def _monitor_loop(self):
        """Monitor for conversation timeout in background."""
        while self.monitoring_active:
            try:
                # Check every 5 seconds
                await asyncio.sleep(5.0)
                
                if self.check_timeout():
                    self.logger.info(f"⏰ Conversation timeout detected after {self.get_time_since_activity():.1f}s")
                    await self._handle_timeout()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in conversation monitor: {e}")
                
    async def _handle_timeout(self):
        """Handle conversation timeout by generating new ID."""
        old_id = self.current_conversation_id
        new_id = self._generate_conversation_id()
        
        # Update state
        self.current_conversation_id = new_id
        self.conversation_start_time = time.time()
        self.last_activity_time = time.time()
        self.timeouts_detected += 1
        self.conversations_started += 1
        
        self.logger.info(f"🔄 Conversation reset on timeout: {old_id} → {new_id}")
        
        # Notify callback (False = not external, this is a timeout)
        if self.on_conversation_change:
            self.on_conversation_change(old_id, new_id, False)
            
    def handle_external_reset(self, new_conversation_id: str) -> bool:
        """
        Handle external conversation reset.
        
        Args:
            new_conversation_id: New conversation ID from external source
            
        Returns:
            bool: True if conversation was reset, False if ID unchanged
        """
        if new_conversation_id == self.current_conversation_id:
            return False
            
        old_id = self.current_conversation_id
        self.current_conversation_id = new_conversation_id
        self.conversation_start_time = time.time()
        self.last_activity_time = time.time()
        self.external_resets += 1
        self.conversations_started += 1
        
        self.logger.info(f"🔄 Conversation reset externally: {old_id} → {new_conversation_id}")
        
        # Notify callback (True = external reset)
        if self.on_conversation_change:
            self.on_conversation_change(old_id, new_conversation_id, True)
            
        return True
        
    def get_conversation_duration(self) -> float:
        """Get current conversation duration in seconds."""
        return time.time() - self.conversation_start_time
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get conversation monitoring metrics."""
        return {
            'current_conversation_id': self.current_conversation_id,
            'conversation_duration': self.get_conversation_duration(),
            'time_since_activity': self.get_time_since_activity(),
            'timeout_seconds': self.timeout,
            'conversations_started': self.conversations_started,
            'timeouts_detected': self.timeouts_detected,
            'external_resets': self.external_resets,
            'monitoring_active': self.monitoring_active
        }
        
    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        duration = self.get_conversation_duration()
        inactive = self.get_time_since_activity()
        
        return (f"Conv: {self.current_conversation_id[-12:]}, "
                f"Duration: {duration:.0f}s, "
                f"Inactive: {inactive:.0f}s/{self.timeout:.0f}s")