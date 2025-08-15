"""
Conversation Context Management

Handles conversation state preservation and context transfer between
OpenAI Realtime API sessions for seamless conversation continuity.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ConversationTurn:
    """A single conversation turn for context preservation"""
    role: str  # 'user' or 'assistant'
    timestamp: float
    text: str  # Transcribed content
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Conversation context for session transfer"""
    turns: List[ConversationTurn] = field(default_factory=list)
    system_state: Dict[str, Any] = field(default_factory=dict)
    active_tools: List[str] = field(default_factory=list)
    personality_modifiers: Dict[str, Any] = field(default_factory=dict)
    
    def to_text_summary(self, max_tokens: int = 2000) -> str:
        """Convert context to text summary for injection"""
        if not self.turns:
            return ""
            
        # Simple approach - keep recent turns
        # TODO: Implement intelligent summarization for long conversations
        recent_turns = self._get_recent_turns(max_tokens)
        summary_lines = []
        
        for turn in recent_turns:
            role_prefix = "User" if turn.role == "user" else "Assistant"
            summary_lines.append(f"{role_prefix}: {turn.text}")
            
        return "\n".join(summary_lines)
    
    def _get_recent_turns(self, max_tokens: int) -> List[ConversationTurn]:
        """Get recent turns that fit within token limit"""
        # Rough token estimation: ~4 chars per token
        max_chars = max_tokens * 4
        current_chars = 0
        recent_turns = []
        
        # Work backwards from most recent
        for turn in reversed(self.turns):
            turn_chars = len(turn.text) + 20  # +20 for role prefix and formatting
            if current_chars + turn_chars > max_chars and recent_turns:
                break
            recent_turns.insert(0, turn)  # Insert at beginning to maintain order
            current_chars += turn_chars
            
        return recent_turns[-10:]  # Keep max 10 turns even if under token limit
    
    def add_turn(self, role: str, text: str, metadata: Optional[Dict] = None):
        """Add a new conversation turn"""
        turn = ConversationTurn(
            role=role,
            timestamp=time.time(),
            text=text.strip(),
            metadata=metadata or {}
        )
        
        # Only add non-empty turns
        if turn.text:
            self.turns.append(turn)
    
    def get_turn_count(self) -> int:
        """Get total number of turns"""
        return len(self.turns)
    
    def get_user_turn_count(self) -> int:
        """Get number of user turns"""
        return len([t for t in self.turns if t.role == "user"])
    
    def get_assistant_turn_count(self) -> int:
        """Get number of assistant turns"""
        return len([t for t in self.turns if t.role == "assistant"])
    
    def get_conversation_duration(self) -> float:
        """Get conversation duration in seconds"""
        if not self.turns:
            return 0.0
        return self.turns[-1].timestamp - self.turns[0].timestamp
    
    def clear_old_turns(self, max_age_seconds: float = 3600):
        """Remove turns older than max_age_seconds"""
        if not self.turns:
            return
            
        cutoff_time = time.time() - max_age_seconds
        self.turns = [t for t in self.turns if t.timestamp >= cutoff_time]
    
    def merge_context(self, other: 'ConversationContext'):
        """Merge another context into this one"""
        # Combine turns and sort by timestamp
        all_turns = self.turns + other.turns
        all_turns.sort(key=lambda t: t.timestamp)
        self.turns = all_turns
        
        # Merge other fields
        self.system_state.update(other.system_state)
        self.active_tools.extend([tool for tool in other.active_tools if tool not in self.active_tools])
        self.personality_modifiers.update(other.personality_modifiers)


class ContextManager:
    """Manages conversation context with intelligent summarization"""
    
    def __init__(self, max_context_tokens: int = 2000, max_context_age: float = 3600):
        self.max_context_tokens = max_context_tokens
        self.max_context_age = max_context_age
        self.current_context = ConversationContext()
        
    def build_system_prompt(self, base_prompt: str, context: Optional[ConversationContext] = None) -> str:
        """Build system prompt with optional context injection"""
        if not context or not context.turns:
            return base_prompt
            
        context_summary = context.to_text_summary(self.max_context_tokens)
        if not context_summary:
            return base_prompt
            
        return f"""{base_prompt}

CONVERSATION CONTEXT:
You are continuing an ongoing conversation with the user. Here is the relevant context from our discussion so far:

{context_summary}

Please continue naturally from where we left off. The user may continue speaking momentarily."""
    
    def add_turn(self, role: str, text: str, metadata: Optional[Dict] = None):
        """Add a turn to the current context"""
        self.current_context.add_turn(role, text, metadata)
        self._cleanup_old_context()
        
    def get_current_context(self) -> ConversationContext:
        """Get the current conversation context"""
        return self.current_context
        
    def reset_context(self):
        """Reset the conversation context"""
        self.current_context = ConversationContext()
        
    def _cleanup_old_context(self):
        """Clean up old conversation turns"""
        self.current_context.clear_old_turns(self.max_context_age)
        
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context statistics"""
        return {
            'total_turns': self.current_context.get_turn_count(),
            'user_turns': self.current_context.get_user_turn_count(), 
            'assistant_turns': self.current_context.get_assistant_turn_count(),
            'conversation_duration': self.current_context.get_conversation_duration(),
            'context_tokens_estimate': len(self.current_context.to_text_summary()) // 4
        }