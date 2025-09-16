"""
User Memory Agent with Kimi K2 and React pattern
Following the system-hint project's tool-based approach
"""

import json
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from openai import OpenAI
from config import Config, MemoryMode
from memory_manager import create_memory_manager, BaseMemoryManager
from conversation_history import ConversationHistory, ConversationTurn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a single tool call with tracking"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserMemoryConfig:
    """Configuration for the user memory agent"""
    enable_memory_updates: bool = True
    enable_conversation_history: bool = True
    enable_memory_search: bool = True
    memory_mode: MemoryMode = MemoryMode.NOTES
    max_memory_context: int = 10  # Max memory items to include in context
    save_trajectory: bool = True
    trajectory_file: str = "memory_trajectory.json"


class UserMemoryAgent:
    """
    User Memory Agent with tool-based React pattern
    """
    
    def __init__(self, 
                 user_id: str,
                 api_key: Optional[str] = None,
                 config: Optional[UserMemoryConfig] = None,
                 verbose: bool = True):
        """
        Initialize the agent
        
        Args:
            user_id: Unique user identifier
            api_key: API key for Kimi/Moonshot (defaults to env)
            config: Agent configuration
            verbose: Enable verbose logging
        """
        self.user_id = user_id
        self.verbose = verbose
        self.config = config or UserMemoryConfig()
        
        # Initialize OpenAI client for Kimi K2
        api_key = api_key or Config.MOONSHOT_API_KEY
        if not api_key:
            raise ValueError("API key required. Set MOONSHOT_API_KEY environment variable.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        self.model = "kimi-k2-0905-preview"  # Use K2 model
        
        # Initialize memory manager
        self.memory_manager = create_memory_manager(user_id, self.config.memory_mode)
        
        # Initialize conversation history
        self.conversation_history = ConversationHistory(user_id) if self.config.enable_conversation_history else None
        
        # Track tool calls
        self.tool_calls: List[ToolCall] = []
        self.tool_call_counts: Dict[str, int] = {}
        
        # Initialize conversation
        self.conversation = []
        self.session_id = self._start_session()
        
        # Initialize system prompt
        self._init_system_prompt()
        
        logger.info(f"UserMemoryAgent initialized for user {user_id} with Kimi K2 model")
    
    def _start_session(self) -> str:
        """Start a new session"""
        return f"session-{uuid.uuid4().hex[:8]}"
    
    def _init_system_prompt(self):
        """Initialize the system prompt with memory context"""
        system_content = """You are an intelligent assistant with persistent memory across conversations. 
You have access to various tools to manage user memories and search conversation history.

## Memory Management:
- Use `read_memories` to access all current user memories
- Use `add_memory` to store new important information about the user
- Use `update_memory` to modify existing memories
- Use `delete_memory` to remove outdated or incorrect memories
- Use `search_memories` to find specific memories

## Conversation Management:
- Use `search_conversations` to find relevant past conversations
- Conversations are automatically saved

## Key Behaviors:
1. Always check existing memories at the start of conversations
2. Proactively update memories when learning new information about the user
3. Reference relevant memories when responding
4. Maintain consistency with previously stored information
5. Be personalized based on what you know about the user

Current Memory Context will be provided with each message.

Important: When you have completed all tasks, clearly state "FINAL ANSWER:" followed by your response."""

        self.conversation = [
            {
                "role": "system",
                "content": system_content
            }
        ]
    
    def _get_memory_context(self) -> str:
        """Get current memory context as a string"""
        context_parts = []
        
        # Add memory summary
        context_parts.append("=== USER MEMORIES ===")
        context_parts.append(self.memory_manager.get_context_string())
        context_parts.append("")
        
        # Add recent conversation history if available
        if self.conversation_history and self.config.enable_conversation_history:
            recent = self.conversation_history.get_recent_turns(limit=5)
            if recent:
                context_parts.append("=== RECENT CONVERSATIONS ===")
                for turn in recent:
                    context_parts.append(f"[{turn.session_id}] User: {turn.user_message[:100]}...")
                    context_parts.append(f"[{turn.session_id}] Assistant: {turn.assistant_message[:100]}...")
                    context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _get_tools_description(self) -> List[Dict[str, Any]]:
        """Get tool descriptions for the model"""
        tools = []
        
        # Memory management tools
        if self.config.enable_memory_updates:
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "read_memories",
                        "description": "Read all current user memories",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "add_memory",
                        "description": "Add a new memory about the user",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The memory content to store"
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional tags for categorizing the memory"
                                }
                            },
                            "required": ["content"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "update_memory",
                        "description": "Update an existing memory",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "memory_id": {
                                    "type": "string",
                                    "description": "ID of the memory to update"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "New content for the memory"
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional new tags"
                                }
                            },
                            "required": ["memory_id", "content"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "delete_memory",
                        "description": "Delete a memory",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "memory_id": {
                                    "type": "string",
                                    "description": "ID of the memory to delete"
                                }
                            },
                            "required": ["memory_id"]
                        }
                    }
                }
            ])
        
        # Search tools
        if self.config.enable_memory_search:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_memories",
                    "description": "Search through user memories",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        if self.config.enable_conversation_history and self.conversation_history:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_conversations",
                    "description": "Search through past conversations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        return tools
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """
        Execute a tool and return the result
        
        Returns:
            Tuple of (result, error_message)
        """
        try:
            if tool_name == "read_memories":
                result = self._tool_read_memories()
            elif tool_name == "add_memory":
                result = self._tool_add_memory(**arguments)
            elif tool_name == "update_memory":
                result = self._tool_update_memory(**arguments)
            elif tool_name == "delete_memory":
                result = self._tool_delete_memory(**arguments)
            elif tool_name == "search_memories":
                result = self._tool_search_memories(**arguments)
            elif tool_name == "search_conversations":
                result = self._tool_search_conversations(**arguments)
            else:
                error = f"Unknown tool: {tool_name}"
                return {"error": error}, error
            
            return result, None
            
        except Exception as e:
            error_msg = f"Tool '{tool_name}' failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}, error_msg
    
    # Tool implementations
    def _tool_read_memories(self) -> Dict[str, Any]:
        """Read all current memories"""
        memory_str = self.memory_manager.get_context_string()
        
        if self.config.memory_mode == MemoryMode.NOTES:
            count = len(self.memory_manager.notes)
        else:
            count = sum(len(items) for subcat in self.memory_manager.memory_cards.values() 
                       for items in subcat.values())
        
        return {
            "success": True,
            "memory_count": count,
            "memories": memory_str
        }
    
    def _tool_add_memory(self, content: str, tags: List[str] = None) -> Dict[str, Any]:
        """Add a new memory"""
        if self.config.memory_mode == MemoryMode.NOTES:
            memory_id = self.memory_manager.add_memory(
                content=content,
                session_id=self.session_id,
                tags=tags or []
            )
        else:
            # For JSON cards, parse content to extract structure
            # Simple heuristic: try to extract category/subcategory from content
            parts = content.split(':')
            if len(parts) >= 2:
                category = "personal"
                subcategory = "info"
                key = parts[0].strip().replace(' ', '_').lower()
                value = ':'.join(parts[1:]).strip()
            else:
                category = "general"
                subcategory = "notes"
                key = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                value = content
            
            memory_id = self.memory_manager.add_memory(
                content={
                    'category': category,
                    'subcategory': subcategory,
                    'key': key,
                    'value': value
                },
                session_id=self.session_id
            )
        
        return {
            "success": True,
            "memory_id": memory_id,
            "message": f"Memory added successfully"
        }
    
    def _tool_update_memory(self, memory_id: str, content: str, tags: List[str] = None) -> Dict[str, Any]:
        """Update an existing memory"""
        if self.config.memory_mode == MemoryMode.NOTES:
            success = self.memory_manager.update_memory(
                memory_id=memory_id,
                content=content,
                session_id=self.session_id,
                tags=tags
            )
        else:
            # For JSON cards, parse the memory_id and content
            parts = memory_id.split('.')
            if len(parts) == 3:
                success = self.memory_manager.update_memory(
                    memory_id=memory_id,
                    content={'value': content},
                    session_id=self.session_id
                )
            else:
                success = False
        
        return {
            "success": success,
            "message": "Memory updated successfully" if success else "Memory not found"
        }
    
    def _tool_delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory"""
        self.memory_manager.delete_memory(memory_id)
        return {
            "success": True,
            "message": f"Memory {memory_id} deleted"
        }
    
    def _tool_search_memories(self, query: str) -> Dict[str, Any]:
        """Search through memories"""
        results = self.memory_manager.search_memories(query)
        
        if self.config.memory_mode == MemoryMode.NOTES:
            formatted_results = [
                {
                    "id": note.note_id,
                    "content": note.content,
                    "tags": note.tags,
                    "updated": note.updated_at
                }
                for note in results
            ]
        else:
            formatted_results = [
                {
                    "path": path,
                    "value": data.get('value'),
                    "updated": data.get('updated_at')
                }
                for path, data in results
            ]
        
        return {
            "success": True,
            "count": len(formatted_results),
            "results": formatted_results
        }
    
    def _tool_search_conversations(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search through conversation history"""
        if not self.conversation_history:
            return {
                "success": False,
                "message": "Conversation history not enabled"
            }
        
        results = self.conversation_history.search_history(query, limit)
        
        formatted_results = [
            {
                "session_id": turn.session_id,
                "user": turn.user_message[:200],
                "assistant": turn.assistant_message[:200],
                "timestamp": turn.timestamp
            }
            for turn in results
        ]
        
        return {
            "success": True,
            "count": len(formatted_results),
            "results": formatted_results
        }
    
    def _save_trajectory(self, iteration: int, final_answer: Optional[str] = None):
        """Save current trajectory to file for debugging"""
        if not self.config.save_trajectory:
            return
        
        trajectory_data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "model": self.model,
            "conversation": self.conversation,
            "tool_calls": [
                {
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "result": call.result,
                    "error": call.error,
                    "timestamp": call.timestamp
                }
                for call in self.tool_calls
            ],
            "memory_state": self.memory_manager.get_context_string(),
            "final_answer": final_answer
        }
        
        try:
            with open(self.config.trajectory_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
            
            if self.verbose:
                logger.info(f"Trajectory saved to {self.config.trajectory_file}")
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")
    
    def execute_task(self, task: str, max_iterations: int = 15) -> Dict[str, Any]:
        """
        Execute a task using React pattern with tool calls
        
        Args:
            task: The task/message from user
            max_iterations: Maximum number of tool call iterations
            
        Returns:
            Task execution result
        """
        # Add user message with memory context
        memory_context = self._get_memory_context()
        full_message = f"{task}\n\n{memory_context}"
        
        self.conversation.append({"role": "user", "content": full_message})
        
        iteration = 0
        final_answer = None
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{max_iterations}")
            
            # Save trajectory
            self._save_trajectory(iteration)
            
            try:
                # Call the model with tools
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation,
                    tools=self._get_tools_description(),
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=4096
                )
                
                message = response.choices[0].message
                
                # Check for final answer
                if message.content and "FINAL ANSWER:" in message.content:
                    final_answer = message.content.split("FINAL ANSWER:")[1].strip()
                    logger.info(f"Final answer found: {final_answer[:100]}...")
                    self.conversation.append(message.model_dump())
                    
                    # Save conversation to history
                    if self.conversation_history:
                        self.conversation_history.add_turn(
                            session_id=self.session_id,
                            user_message=task,
                            assistant_message=final_answer
                        )
                    
                    # Save final trajectory
                    self._save_trajectory(iteration, final_answer)
                    break
                
                # Handle tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    self.conversation.append(message.model_dump())
                    
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Track tool call count
                        self.tool_call_counts[function_name] = self.tool_call_counts.get(function_name, 0) + 1
                        call_number = self.tool_call_counts[function_name]
                        
                        logger.info(f"Executing tool: {function_name} (call #{call_number})")
                        
                        # Execute the tool
                        result, error = self._execute_tool(function_name, function_args)
                        
                        # Log result
                        if error:
                            logger.info(f"  ❌ Error: {error}")
                        else:
                            logger.info(f"  ✅ Success: {json.dumps(result)[:200]}")
                        
                        # Record tool call
                        tool_call_record = ToolCall(
                            tool_name=function_name,
                            arguments=function_args,
                            result=result if not error else None,
                            error=error
                        )
                        self.tool_calls.append(tool_call_record)
                        
                        # Add tool result to conversation
                        self.conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })
                
                elif message.content:
                    # Regular assistant message without final answer
                    self.conversation.append(message.model_dump())
                    
            except Exception as e:
                logger.error(f"Error during task execution: {str(e)}")
                self._save_trajectory(iteration)
                return {
                    "error": str(e),
                    "tool_calls": self.tool_calls,
                    "iterations": iteration,
                    "trajectory_file": self.config.trajectory_file if self.config.save_trajectory else None
                }
        
        # Save final trajectory
        self._save_trajectory(iteration, final_answer)
        
        return {
            "final_answer": final_answer,
            "tool_calls": self.tool_calls,
            "iterations": iteration,
            "success": final_answer is not None,
            "memory_state": self.memory_manager.get_context_string(),
            "trajectory_file": self.config.trajectory_file if self.config.save_trajectory else None
        }
    
    def chat(self, message: str) -> str:
        """
        Simple chat interface (wraps execute_task for compatibility)
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
        result = self.execute_task(message)
        return result.get('final_answer', 'I apologize, but I was unable to generate a response.')
    
    def reset(self):
        """Reset the agent's state for a new conversation"""
        self.tool_calls = []
        self.tool_call_counts = {}
        self.session_id = self._start_session()
        self._init_system_prompt()
        logger.info("Agent state reset")
