"""
Conversational Agent - Focuses purely on conversation without direct memory management
Memory updates are handled by a separate background process
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
from openai import OpenAI
from config import Config
from conversation_history import ConversationHistory, ConversationTurn
from memory_manager import create_memory_manager, BaseMemoryManager, MemoryMode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ConversationConfig:
    """Configuration for the conversational agent"""
    enable_memory_context: bool = True  # Include memory in context but don't update
    enable_conversation_history: bool = True
    max_memory_context: int = 10
    temperature: float = 0.7
    max_tokens: int = 4096


class ConversationalAgent:
    """
    Pure conversational agent that focuses on dialogue
    Reads memory for context but doesn't update it directly
    """
    
    def __init__(self, 
                 user_id: str,
                 api_key: Optional[str] = None,
                 config: Optional[ConversationConfig] = None,
                 memory_mode: MemoryMode = MemoryMode.NOTES,
                 verbose: bool = False):
        """
        Initialize the conversational agent
        
        Args:
            user_id: Unique user identifier
            api_key: API key for Kimi/Moonshot
            config: Agent configuration
            memory_mode: Memory storage mode
            verbose: Enable verbose logging
        """
        self.user_id = user_id
        self.verbose = verbose
        self.config = config or ConversationConfig()
        self.memory_mode = memory_mode
        
        # Initialize OpenAI client
        api_key = api_key or Config.MOONSHOT_API_KEY
        if not api_key:
            raise ValueError("API key required. Set MOONSHOT_API_KEY environment variable.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        self.model = "kimi-k2-0905-preview"
        
        # Initialize memory manager (read-only access)
        self.memory_manager = create_memory_manager(user_id, memory_mode)
        
        # Initialize conversation history
        self.conversation_history = ConversationHistory(user_id) if self.config.enable_conversation_history else None
        
        # Track current session
        self.session_id = self._generate_session_id()
        self.conversation = []
        
        # Initialize system prompt
        self._init_system_prompt()
        
        logger.info(f"ConversationalAgent initialized for user {user_id}")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return f"session-{uuid.uuid4().hex[:8]}"
    
    def _init_system_prompt(self):
        """Initialize the system prompt"""
        system_content = """You are a helpful and personalized assistant. You have access to information about the user from previous conversations, which helps you provide personalized and contextual responses.

## Key Behaviors:
1. Be conversational and natural
2. Reference relevant user information when appropriate
3. Maintain consistency with what you know about the user
4. Be helpful and personalized based on user context
5. Focus on having a good conversation

User context and memories will be provided with each message."""

        self.conversation = [
            {
                "role": "system",
                "content": system_content
            }
        ]
    
    def _get_memory_context(self) -> str:
        """Get current memory context as a string"""
        if not self.config.enable_memory_context:
            return ""
        
        context_parts = []
        
        # Add memory summary
        memory_str = self.memory_manager.get_context_string()
        if memory_str:
            context_parts.append("=== USER CONTEXT ===")
            context_parts.append(memory_str)
            context_parts.append("")
        
        # Add recent conversation history
        if self.conversation_history:
            recent = self.conversation_history.get_recent_turns(limit=3)
            if recent:
                context_parts.append("=== RECENT CONVERSATIONS ===")
                for turn in recent:
                    context_parts.append(f"[{turn.timestamp}]")
                    context_parts.append(f"User: {turn.user_message[:150]}...")
                    context_parts.append(f"Assistant: {turn.assistant_message[:150]}...")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """
        Get the full conversation context for background memory processing
        
        Returns:
            List of conversation messages
        """
        # Return a copy of the conversation without system prompt
        return [msg for msg in self.conversation[1:] if msg.get('role') != 'system']
    
    def chat(self, message: str) -> str:
        """
        Have a conversation with the user
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
        # Add memory context to the user message
        memory_context = self._get_memory_context()
        
        if memory_context:
            full_message = f"{message}\n\n{memory_context}"
        else:
            full_message = message
        
        # Add to conversation
        self.conversation.append({"role": "user", "content": full_message})
        
        try:
            # Call the model
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to conversation
            self.conversation.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Save to conversation history
            if self.conversation_history:
                self.conversation_history.add_turn(
                    session_id=self.session_id,
                    user_message=message,
                    assistant_message=assistant_message
                )
            
            if self.verbose:
                logger.info(f"User: {message[:100]}...")
                logger.info(f"Assistant: {assistant_message[:100]}...")
            
            return assistant_message
            
        except Exception as e:
            error_msg = f"Error during conversation: {str(e)}"
            logger.error(error_msg)
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def reset_session(self):
        """Start a new conversation session"""
        self.session_id = self._generate_session_id()
        self._init_system_prompt()
        logger.info(f"Started new session: {self.session_id}")
    
    def get_session_id(self) -> str:
        """Get the current session ID"""
        return self.session_id
