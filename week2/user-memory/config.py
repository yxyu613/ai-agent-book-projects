"""
Configuration module for User Memory System
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from enum import Enum

# Load environment variables
load_dotenv()


class MemoryMode(Enum):
    """Memory management modes"""
    NOTES = "notes"  # List of notes with session references
    JSON_CARDS = "json_cards"  # Hierarchical JSON memory cards


class Config:
    """Configuration settings for the user memory system"""
    
    # Kimi Model Configuration (similar to week1/context)
    MOONSHOT_API_KEY: str = os.getenv("MOONSHOT_API_KEY", "")
    MOONSHOT_BASE_URL: str = "https://api.moonshot.cn/v1"
    
    # Model Configuration (Using Kimi K2)
    MODEL_NAME: str = os.getenv("MODEL_NAME", "kimi-k2-0905-preview")  # Kimi K2 model
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
    MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "4096"))
    
    # Memory Configuration
    MEMORY_MODE: MemoryMode = MemoryMode(os.getenv("MEMORY_MODE", "notes").lower())
    MAX_MEMORY_ITEMS: int = int(os.getenv("MAX_MEMORY_ITEMS", "100"))
    MEMORY_UPDATE_TEMPERATURE: float = float(os.getenv("MEMORY_UPDATE_TEMPERATURE", "0.2"))
    
    # Dify Configuration for conversation history search
    DIFY_API_KEY: str = os.getenv("DIFY_API_KEY", "")
    DIFY_BASE_URL: str = os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1")
    DIFY_DATASET_ID: str = os.getenv("DIFY_DATASET_ID", "")
    ENABLE_HISTORY_SEARCH: bool = os.getenv("ENABLE_HISTORY_SEARCH", "false").lower() == "true"
    
    # Session Configuration
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))  # seconds
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))  # tokens
    
    # LOCOMO Benchmark Configuration
    LOCOMO_DATASET_PATH: str = os.getenv("LOCOMO_DATASET_PATH", "data/locomo")
    LOCOMO_OUTPUT_DIR: str = os.getenv("LOCOMO_OUTPUT_DIR", "results/locomo")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "logs/user_memory.log")
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    
    # Storage paths
    MEMORY_STORAGE_DIR: str = os.getenv("MEMORY_STORAGE_DIR", "data/memories")
    CONVERSATION_HISTORY_DIR: str = os.getenv("CONVERSATION_HISTORY_DIR", "data/conversations")
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate required configuration
        
        Returns:
            True if configuration is valid
        """
        if not cls.MOONSHOT_API_KEY:
            print("ERROR: MOONSHOT_API_KEY is not set")
            print("Please set it in .env file or as environment variable")
            return False
        
        if cls.ENABLE_HISTORY_SEARCH and not cls.DIFY_API_KEY:
            print("WARNING: History search enabled but DIFY_API_KEY not set")
            print("History search will be disabled")
            cls.ENABLE_HISTORY_SEARCH = False
        
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.MEMORY_STORAGE_DIR, exist_ok=True)
        os.makedirs(cls.CONVERSATION_HISTORY_DIR, exist_ok=True)
        os.makedirs(cls.LOCOMO_OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.LOG_FILE) if cls.LOG_FILE else "logs", exist_ok=True)
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """
        Get model configuration as dictionary
        
        Returns:
            Model configuration dict
        """
        return {
            "model": cls.MODEL_NAME,
            "temperature": cls.MODEL_TEMPERATURE,
            "max_tokens": cls.MODEL_MAX_TOKENS
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("\n" + "="*50)
        print("USER MEMORY SYSTEM CONFIGURATION")
        print("="*50)
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Memory Mode: {cls.MEMORY_MODE.value}")
        print(f"Max Memory Items: {cls.MAX_MEMORY_ITEMS}")
        print(f"History Search: {'Enabled' if cls.ENABLE_HISTORY_SEARCH else 'Disabled'}")
        print(f"Kimi API Key Set: {'Yes' if cls.MOONSHOT_API_KEY else 'No'}")
        print(f"Dify API Key Set: {'Yes' if cls.DIFY_API_KEY else 'No'}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("="*50 + "\n")
