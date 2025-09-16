"""
Background Memory Processor - Analyzes conversation context and updates memories
Runs separately from the main conversational agent
"""

import json
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from config import Config, MemoryMode
from memory_manager import create_memory_manager, BaseMemoryManager
from conversation_history import ConversationHistory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MemoryUpdate:
    """Represents a memory update decision"""
    action: str  # 'add', 'update', 'delete', 'none'
    memory_id: Optional[str] = None
    content: Optional[str] = None
    reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class MemoryProcessorConfig:
    """Configuration for the background memory processor"""
    conversation_interval: int = 1  # Process after N conversation rounds (default: every round)
    min_conversation_turns: int = 1  # Minimum turns before processing
    context_window: int = 10  # Number of recent turns to analyze
    update_threshold: float = 0.7  # Confidence threshold for updates
    enable_auto_processing: bool = True
    temperature: float = 0.3  # Lower temperature for analysis
    output_operations: bool = True  # Output detailed memory operations


class BackgroundMemoryProcessor:
    """
    Background processor that analyzes conversations and updates memory
    Runs separately from the main conversation flow
    """
    
    def __init__(self,
                 user_id: str,
                 api_key: Optional[str] = None,
                 config: Optional[MemoryProcessorConfig] = None,
                 memory_mode: MemoryMode = MemoryMode.NOTES,
                 verbose: bool = False):
        """
        Initialize the background memory processor
        
        Args:
            user_id: Unique user identifier
            api_key: API key for Kimi/Moonshot
            config: Processor configuration
            memory_mode: Memory storage mode
            verbose: Enable verbose logging
        """
        self.user_id = user_id
        self.verbose = verbose
        self.config = config or MemoryProcessorConfig()
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
        
        # Initialize managers
        self.memory_manager = create_memory_manager(user_id, memory_mode)
        self.conversation_history = ConversationHistory(user_id)
        
        # Background processing state
        self.processing_thread = None
        self.stop_processing = False
        self.last_processed_timestamp = None
        self.processing_lock = threading.Lock()
        self.conversation_count = 0  # Track conversation rounds
        self.last_processed_count = 0  # Track last processed conversation count
        
        logger.info(f"BackgroundMemoryProcessor initialized for user {user_id}")
    
    def _create_analysis_prompt(self, conversation_context: List[Dict[str, str]], 
                               current_memories: str) -> str:
        """
        Create a prompt for analyzing conversation and determining memory updates
        """
        # Format conversation
        conversation_str = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_context
        ])
        
        prompt = f"""Analyze the following conversation and determine what memories should be updated about the user.

## Current User Memories:
{current_memories if current_memories else "No existing memories"}

## Recent Conversation:
{conversation_str}

## Instructions:
1. Identify any new information about the user that should be remembered
2. Check if any existing memories need to be updated or corrected
3. Determine if any memories are outdated and should be deleted
4. Consider the full conversation context, not just individual messages
5. Only suggest updates for significant, persistent information about the user

For each memory update needed, provide:
- Action: "add", "update", or "delete"
- Memory ID (for update/delete): The ID of the existing memory
- Content: The new or updated information (for add/update)
- Reason: Why this update is needed
- Tags: Relevant categories for the memory
- Confidence: Your confidence level (0.0-1.0)

Return your analysis as a JSON array of memory updates. If no updates are needed, return an empty array [].

Example format:
[
  {{
    "action": "add",
    "content": "User prefers Python for data science projects",
    "reason": "User explicitly mentioned their preference",
    "tags": ["preferences", "programming"],
    "confidence": 0.9
  }},
  {{
    "action": "update",
    "memory_id": "note_123",
    "content": "User works at NewCompany (changed from OldCompany)",
    "reason": "User mentioned job change",
    "tags": ["work", "career"],
    "confidence": 0.85
  }}
]

Analyze the conversation and provide memory updates:"""

        return prompt
    
    def _parse_memory_updates(self, analysis: str) -> List[MemoryUpdate]:
        """
        Parse the LLM's analysis into MemoryUpdate objects
        """
        try:
            # Extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', analysis, re.DOTALL)
            if not json_match:
                return []
            
            updates_json = json.loads(json_match.group())
            
            updates = []
            for item in updates_json:
                update = MemoryUpdate(
                    action=item.get('action', 'none'),
                    memory_id=item.get('memory_id'),
                    content=item.get('content'),
                    reason=item.get('reason'),
                    tags=item.get('tags', []),
                    confidence=item.get('confidence', 0.0)
                )
                updates.append(update)
            
            return updates
            
        except Exception as e:
            logger.error(f"Failed to parse memory updates: {e}")
            return []
    
    def analyze_conversation(self, conversation_context: List[Dict[str, str]]) -> List[MemoryUpdate]:
        """
        Analyze conversation context and determine memory updates
        
        Args:
            conversation_context: List of conversation messages
            
        Returns:
            List of memory updates to apply
        """
        if len(conversation_context) < self.config.min_conversation_turns * 2:
            return []
        
        # Get current memory state
        current_memories = self.memory_manager.get_context_string()
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(conversation_context, current_memories)
        
        try:
            # Call LLM for analysis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory management assistant that analyzes conversations and determines what information about users should be remembered, updated, or forgotten."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=2048
            )
            
            analysis = response.choices[0].message.content
            
            if self.verbose:
                logger.info(f"Memory analysis: {analysis[:200]}...")
            
            # Parse the analysis
            updates = self._parse_memory_updates(analysis)
            
            # Filter by confidence threshold
            filtered_updates = [
                u for u in updates 
                if u.confidence >= self.config.update_threshold
            ]
            
            return filtered_updates
            
        except Exception as e:
            logger.error(f"Failed to analyze conversation: {e}")
            return []
    
    def apply_memory_updates(self, updates: List[MemoryUpdate]) -> Dict[str, Any]:
        """
        Apply memory updates to the memory manager
        
        Args:
            updates: List of memory updates to apply
            
        Returns:
            Summary of applied updates
        """
        results = {
            'added': 0,
            'updated': 0,
            'deleted': 0,
            'failed': 0,
            'details': []
        }
        
        for update in updates:
            try:
                if update.action == 'add' and update.content:
                    memory_id = self.memory_manager.add_memory(
                        content=update.content,
                        session_id=f"background-{datetime.now().isoformat()}",
                        tags=update.tags
                    )
                    results['added'] += 1
                    results['details'].append(f"Added: {update.content[:50]}...")
                    
                    # Always print to console for demo purposes
                    print(f"  📝 [ADD] Memory: {update.content}")
                    
                    if self.verbose:
                        logger.info(f"Added memory: {update.content}")
                
                elif update.action == 'update' and update.memory_id and update.content:
                    success = self.memory_manager.update_memory(
                        memory_id=update.memory_id,
                        content=update.content,
                        session_id=f"background-{datetime.now().isoformat()}",
                        tags=update.tags
                    )
                    if success:
                        results['updated'] += 1
                        results['details'].append(f"Updated {update.memory_id}: {update.content[:50]}...")
                        
                        # Always print to console for demo purposes
                        print(f"  ✏️  [UPDATE] Memory (ID: {update.memory_id[:8]}...): {update.content}")
                    else:
                        results['failed'] += 1
                    
                    if self.verbose:
                        logger.info(f"Updated memory {update.memory_id}: {update.content}")
                
                elif update.action == 'delete' and update.memory_id:
                    self.memory_manager.delete_memory(update.memory_id)
                    results['deleted'] += 1
                    results['details'].append(f"Deleted: {update.memory_id}")
                    
                    # Always print to console for demo purposes
                    print(f"  🗑️  [DELETE] Memory ID: {update.memory_id}")
                    
                    if self.verbose:
                        logger.info(f"Deleted memory: {update.memory_id}")
                
            except Exception as e:
                logger.error(f"Failed to apply update {update.action}: {e}")
                results['failed'] += 1
        
        return results
    
    def process_recent_conversations(self) -> Dict[str, Any]:
        """
        Process recent conversations and update memories
        
        Returns:
            Processing results with list of operations
        """
        with self.processing_lock:
            # Get recent conversation turns
            recent_turns = self.conversation_history.get_recent_turns(
                limit=self.config.context_window
            )
            
            if not recent_turns:
                return {
                    'message': 'No recent conversations to process',
                    'operations': [],
                    'summary': {'added': 0, 'updated': 0, 'deleted': 0}
                }
            
            # Convert to conversation format
            conversation_context = []
            for turn in recent_turns:
                conversation_context.append({
                    'role': 'user',
                    'content': turn.user_message
                })
                conversation_context.append({
                    'role': 'assistant',
                    'content': turn.assistant_message
                })
            
            # Analyze conversation
            updates = self.analyze_conversation(conversation_context)
            
            # Create operations list
            operations = []
            for update in updates:
                operation = {
                    'action': update.action,
                    'content': update.content[:100] + '...' if update.content and len(update.content) > 100 else update.content,
                    'reason': update.reason,
                    'confidence': update.confidence
                }
                if update.memory_id:
                    operation['memory_id'] = update.memory_id
                if update.tags:
                    operation['tags'] = update.tags
                operations.append(operation)
            
            if not updates:
                return {
                    'message': 'No memory updates needed',
                    'analyzed_turns': len(recent_turns),
                    'operations': [],
                    'summary': {'added': 0, 'updated': 0, 'deleted': 0}
                }
            
            # Apply updates
            results = self.apply_memory_updates(updates)
            
            # Format final results
            final_results = {
                'analyzed_turns': len(recent_turns),
                'operations': operations,
                'summary': {
                    'added': results['added'],
                    'updated': results['updated'], 
                    'deleted': results['deleted'],
                    'failed': results.get('failed', 0)
                },
                'details': results.get('details', [])
            }
            
            # Update last processed timestamp and count
            self.last_processed_timestamp = datetime.now()
            self.last_processed_count = self.conversation_count
            
            return final_results
    
    def should_process(self) -> bool:
        """
        Check if memory processing should be triggered based on conversation count
        
        Returns:
            True if processing should occur
        """
        if self.conversation_count == 0:
            return False
        
        # Check if we've reached the conversation interval
        conversations_since_last = self.conversation_count - self.last_processed_count
        return conversations_since_last >= self.config.conversation_interval
    
    def increment_conversation_count(self):
        """
        Increment the conversation counter
        """
        self.conversation_count += 1
        
        if self.verbose:
            logger.info(f"Conversation count: {self.conversation_count}, Last processed: {self.last_processed_count}")
    
    def _background_processing_loop(self):
        """
        Background loop for automatic memory processing based on conversation count
        """
        logger.info(f"Starting background memory processing (interval: every {self.config.conversation_interval} conversations)")
        
        while not self.stop_processing:
            try:
                # Check every second if we should process
                time.sleep(1)
                
                if self.stop_processing:
                    break
                
                # Check if we should process based on conversation count
                if self.should_process():
                    results = self.process_recent_conversations()
                    
                    if self.config.output_operations:
                        self._output_operations(results)
                    
                    if self.verbose:
                        logger.info(f"Background processing results: {results.get('summary')}")
                
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
        
        logger.info("Background memory processing stopped")
    
    def _output_operations(self, results: Dict[str, Any]):
        """
        Output memory operations in a formatted way
        
        Args:
            results: Processing results with operations
        """
        operations = results.get('operations', [])
        summary = results.get('summary', {})
        
        if not operations:
            logger.info("📝 Memory Operations: None (no updates needed)")
            return
        
        logger.info(f"\n📝 Memory Operations ({len(operations)} total):")
        logger.info("-" * 50)
        
        for i, op in enumerate(operations, 1):
            icon = {
                'add': '➕',
                'update': '📝',
                'delete': '🗑️'
            }.get(op['action'], '❓')
            
            logger.info(f"{i}. {icon} {op['action'].upper()}")
            if op.get('content'):
                logger.info(f"   Content: {op['content']}")
            if op.get('memory_id'):
                logger.info(f"   Memory ID: {op['memory_id']}")
            if op.get('reason'):
                logger.info(f"   Reason: {op['reason']}")
            logger.info(f"   Confidence: {op['confidence']:.2%}")
            if op.get('tags'):
                logger.info(f"   Tags: {', '.join(op['tags'])}")
            logger.info("")
        
        logger.info(f"Summary: {summary.get('added', 0)} added, {summary.get('updated', 0)} updated, {summary.get('deleted', 0)} deleted")
        logger.info("-" * 50)
    
    def start_background_processing(self):
        """Start the background memory processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Background processing already running")
            return
        
        self.stop_processing = False
        self.processing_thread = threading.Thread(
            target=self._background_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        logger.info("Background memory processing started")
    
    def stop_background_processing(self):
        """Stop the background memory processing thread"""
        self.stop_processing = True
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Background memory processing stopped")
    
    def process_conversation_batch(self, conversation_contexts: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """
        Process multiple conversation contexts in batch
        
        Args:
            conversation_contexts: List of conversation contexts
            
        Returns:
            List of processing results
        """
        results = []
        
        for context in conversation_contexts:
            updates = self.analyze_conversation(context)
            
            operations = []
            for update in updates:
                operation = {
                    'action': update.action,
                    'content': update.content[:100] + '...' if update.content and len(update.content) > 100 else update.content,
                    'confidence': update.confidence
                }
                if update.memory_id:
                    operation['memory_id'] = update.memory_id
                operations.append(operation)
            
            if updates:
                apply_result = self.apply_memory_updates(updates)
                result = {
                    'operations': operations,
                    'summary': {
                        'added': apply_result['added'],
                        'updated': apply_result['updated'],
                        'deleted': apply_result['deleted']
                    }
                }
            else:
                result = {
                    'message': 'No updates needed',
                    'operations': [],
                    'summary': {'added': 0, 'updated': 0, 'deleted': 0}
                }
            
            results.append(result)
        
        return results
