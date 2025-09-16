"""
Memory Manager module for handling different memory mechanisms
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import logging
from config import Config, MemoryMode

logger = logging.getLogger(__name__)


@dataclass
class MemoryNote:
    """Represents a single memory note"""
    note_id: str
    content: str
    session_id: str
    created_at: str
    updated_at: str
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNote':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class MemoryCard:
    """Represents a memory card in JSON structure"""
    category: str
    subcategory: str
    key: str
    value: Any
    session_id: str
    created_at: str
    updated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryCard':
        """Create from dictionary"""
        return cls(**data)


class BaseMemoryManager(ABC):
    """Base class for memory managers"""
    
    def __init__(self, user_id: str):
        """
        Initialize memory manager
        
        Args:
            user_id: Unique identifier for the user
        """
        self.user_id = user_id
        self.memory_file = os.path.join(Config.MEMORY_STORAGE_DIR, f"{user_id}_memory.json")
        self.load_memory()
    
    @abstractmethod
    def load_memory(self):
        """Load memory from storage"""
        pass
    
    @abstractmethod
    def save_memory(self):
        """Save memory to storage"""
        pass
    
    @abstractmethod
    def add_memory(self, content: Any, session_id: str, **kwargs):
        """Add a new memory item"""
        pass
    
    @abstractmethod
    def update_memory(self, memory_id: str, content: Any, session_id: str, **kwargs):
        """Update an existing memory item"""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str):
        """Delete a memory item"""
        pass
    
    @abstractmethod
    def get_context_string(self) -> str:
        """Get memory as a formatted string for LLM context"""
        pass
    
    @abstractmethod
    def search_memories(self, query: str) -> List[Any]:
        """Search memories by query"""
        pass


class NotesMemoryManager(BaseMemoryManager):
    """Memory manager using notes list approach"""
    
    def __init__(self, user_id: str):
        self.notes: List[MemoryNote] = []
        super().__init__(user_id)
    
    def load_memory(self):
        """Load notes from storage"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.notes = [MemoryNote.from_dict(note) for note in data.get('notes', [])]
                logger.info(f"Loaded {len(self.notes)} notes for user {self.user_id}")
            except Exception as e:
                logger.error(f"Error loading notes: {e}")
                self.notes = []
        else:
            self.notes = []
            logger.info(f"No existing memory file for user {self.user_id}")
    
    def save_memory(self):
        """Save notes to storage"""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                data = {
                    'user_id': self.user_id,
                    'type': 'notes',
                    'updated_at': datetime.now().isoformat(),
                    'notes': [note.to_dict() for note in self.notes]
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.notes)} notes for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving notes: {e}")
    
    def add_memory(self, content: str, session_id: str, tags: List[str] = None):
        """Add a new note"""
        note = MemoryNote(
            note_id=str(uuid.uuid4()),
            content=content,
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=tags or []
        )
        self.notes.append(note)
        
        # Keep only the most recent notes if limit exceeded
        if len(self.notes) > Config.MAX_MEMORY_ITEMS:
            # Sort by updated_at and keep the most recent
            self.notes.sort(key=lambda n: n.updated_at, reverse=True)
            self.notes = self.notes[:Config.MAX_MEMORY_ITEMS]
        
        self.save_memory()
        return note.note_id
    
    def update_memory(self, memory_id: str, content: str, session_id: str, tags: List[str] = None):
        """Update an existing note"""
        for note in self.notes:
            if note.note_id == memory_id:
                note.content = content
                note.session_id = session_id
                note.updated_at = datetime.now().isoformat()
                if tags is not None:
                    note.tags = tags
                self.save_memory()
                return True
        return False
    
    def delete_memory(self, memory_id: str):
        """Delete a note"""
        self.notes = [note for note in self.notes if note.note_id != memory_id]
        self.save_memory()
    
    def get_context_string(self) -> str:
        """Get notes as formatted string for LLM context"""
        if not self.notes:
            return "No previous memory notes available."
        
        context = "User Memory Notes:\n\n"
        for i, note in enumerate(self.notes, 1):
            context += f"Note {i} (ID: {note.note_id}, Session: {note.session_id}):\n"
            context += f"  Content: {note.content}\n"
            if note.tags:
                context += f"  Tags: {', '.join(note.tags)}\n"
            context += f"  Updated: {note.updated_at}\n\n"
        
        return context
    
    def search_memories(self, query: str) -> List[MemoryNote]:
        """Search notes by query (simple text search)"""
        query_lower = query.lower()
        results = []
        for note in self.notes:
            if query_lower in note.content.lower() or any(query_lower in tag.lower() for tag in note.tags):
                results.append(note)
        return results


class JSONMemoryManager(BaseMemoryManager):
    """Memory manager using hierarchical JSON cards approach"""
    
    def __init__(self, user_id: str):
        self.memory_cards: Dict[str, Dict[str, Dict[str, Any]]] = {}
        super().__init__(user_id)
    
    def load_memory(self):
        """Load JSON memory cards from storage"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memory_cards = data.get('memory_cards', {})
                logger.info(f"Loaded memory cards for user {self.user_id}")
            except Exception as e:
                logger.error(f"Error loading memory cards: {e}")
                self.memory_cards = {}
        else:
            self.memory_cards = {}
            logger.info(f"No existing memory file for user {self.user_id}")
    
    def save_memory(self):
        """Save JSON memory cards to storage"""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                data = {
                    'user_id': self.user_id,
                    'type': 'json_cards',
                    'updated_at': datetime.now().isoformat(),
                    'memory_cards': self.memory_cards
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved memory cards for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving memory cards: {e}")
    
    def add_memory(self, content: Dict[str, Any], session_id: str, **kwargs):
        """
        Add a new memory card
        
        Args:
            content: Dictionary with 'category', 'subcategory', 'key', and 'value'
            session_id: Session identifier
        """
        category = content.get('category', 'general')
        subcategory = content.get('subcategory', 'info')
        key = content.get('key', str(uuid.uuid4()))
        value = content.get('value')
        
        if category not in self.memory_cards:
            self.memory_cards[category] = {}
        
        if subcategory not in self.memory_cards[category]:
            self.memory_cards[category][subcategory] = {}
        
        self.memory_cards[category][subcategory][key] = {
            'value': value,
            'source': session_id,
            'updated_at': datetime.now().isoformat()
        }
        
        self.save_memory()
        return f"{category}.{subcategory}.{key}"
    
    def update_memory(self, memory_id: str, content: Dict[str, Any], session_id: str, **kwargs):
        """Update an existing memory card"""
        parts = memory_id.split('.')
        if len(parts) != 3:
            return False
        
        category, subcategory, key = parts
        
        if (category in self.memory_cards and 
            subcategory in self.memory_cards[category] and 
            key in self.memory_cards[category][subcategory]):
            
            value = content.get('value')
            self.memory_cards[category][subcategory][key] = {
                'value': value,
                'source': session_id,
                'updated_at': datetime.now().isoformat()
            }
            self.save_memory()
            return True
        
        return False
    
    def delete_memory(self, memory_id: str):
        """Delete a memory card"""
        parts = memory_id.split('.')
        if len(parts) != 3:
            return
        
        category, subcategory, key = parts
        
        if (category in self.memory_cards and 
            subcategory in self.memory_cards[category] and 
            key in self.memory_cards[category][subcategory]):
            
            del self.memory_cards[category][subcategory][key]
            
            # Clean up empty subcategories and categories
            if not self.memory_cards[category][subcategory]:
                del self.memory_cards[category][subcategory]
            if not self.memory_cards[category]:
                del self.memory_cards[category]
            
            self.save_memory()
    
    def get_context_string(self) -> str:
        """Get memory cards as formatted string for LLM context"""
        if not self.memory_cards:
            return "No previous memory cards available."
        
        context = "User Memory Cards (Hierarchical JSON):\n\n"
        context += json.dumps(self.memory_cards, indent=2, ensure_ascii=False)
        return context
    
    def search_memories(self, query: str) -> List[Tuple[str, Any]]:
        """Search memory cards by query"""
        query_lower = query.lower()
        results = []
        
        for category, subcategories in self.memory_cards.items():
            for subcategory, items in subcategories.items():
                for key, data in items.items():
                    memory_path = f"{category}.{subcategory}.{key}"
                    value_str = str(data.get('value', '')).lower()
                    
                    if (query_lower in category.lower() or 
                        query_lower in subcategory.lower() or 
                        query_lower in key.lower() or 
                        query_lower in value_str):
                        
                        results.append((memory_path, data))
        
        return results


def create_memory_manager(user_id: str, mode: MemoryMode = None) -> BaseMemoryManager:
    """
    Factory function to create appropriate memory manager
    
    Args:
        user_id: User identifier
        mode: Memory mode (defaults to config setting)
    
    Returns:
        Memory manager instance
    """
    mode = mode or Config.MEMORY_MODE
    
    if mode == MemoryMode.NOTES:
        return NotesMemoryManager(user_id)
    elif mode == MemoryMode.JSON_CARDS:
        return JSONMemoryManager(user_id)
    else:
        raise ValueError(f"Unknown memory mode: {mode}")
