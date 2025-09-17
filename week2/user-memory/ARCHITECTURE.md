# Separated Memory Architecture

## Overview

The user-memory system now uses a **separated architecture** where:

1. **Conversational Agent** - Handles dialogue and maintains conversation flow
2. **Background Memory Processor** - Analyzes conversations and updates memory separately

This separation provides cleaner code organization and more flexible memory management.

## Key Components

### 1. ConversationalAgent (`conversational_agent.py`)

The main conversational interface that:
- Focuses purely on having natural conversations with users
- Reads existing memories for context (read-only)
- Maintains conversation history
- Does NOT directly update memories

**Key Features:**
- Lightweight and responsive
- Memory-aware but not memory-managing
- Session management for conversations
- Full conversation context tracking

### 2. BackgroundMemoryProcessor (`background_memory_processor.py`)

A separate processor that:
- Analyzes full conversation context (not just current turn)
- Determines what memories need to be added, updated, or deleted
- Applies memory updates incrementally
- Can run in background or be triggered manually

**Key Features:**
- Analyzes entire conversation context holistically
- Supports both manual and automatic processing
- Thread-safe background operation

### 3. Updated Main Entry Point (`main.py`)

Refactored to:
- Use ConversationalAgent for dialogue
- Optionally enable BackgroundMemoryProcessor
- Support various execution modes
- Coordinate between conversation and memory processing

## Architecture Benefits

### 1. Separation of Concerns
- **Conversation Logic** - Isolated in ConversationalAgent
- **Memory Management** - Isolated in BackgroundMemoryProcessor
- Clean, maintainable code structure

### 2. Better Context Analysis
- Background processor sees full conversation context
- Can make more informed decisions about what to remember
- Avoids redundant or conflicting memory updates

### 3. Flexible Processing
- **Real-time Mode**: Conversations happen immediately
- **Conversation-Based Trigger**: Memory updates after N conversation rounds (default: 1)
- **Manual Mode**: Full control over when memories are processed
- **Operation List**: Clear output of zero, one, or more memory operations

### 4. Performance Optimization
- Conversations are not blocked by memory operations
- Memory processing can be batched and optimized
- Reduced API calls during conversation

## Usage Examples

### Interactive Mode with Background Processing

```bash
python main.py --mode interactive --user alice --conversation-interval 1
```

This starts an interactive session where:
- Conversations happen in real-time
- Memory updates are processed after each conversation round (configurable)
- User can chat naturally without waiting for memory operations

### Single Conversation Mode

```bash
python main.py --mode single --user bob --message "I work as a data scientist at TechCorp"
```

Sends a single message and gets a response. Memory processing can be triggered separately.

### Demo Mode

```bash
python main.py --mode demo
```

Demonstrates the full separated architecture with example conversations and memory processing.

## Configuration Options

### ConversationConfig

```python
ConversationConfig(
    enable_memory_context=True,    # Include memories in conversation context
    enable_conversation_history=True,  # Track conversation history
    temperature=0.7,               # Response creativity
    max_tokens=4096                # Maximum response length
)
```

### MemoryProcessorConfig

```python
MemoryProcessorConfig(
    conversation_interval=1,       # Process after N conversation rounds
    min_conversation_turns=1,      # Minimum turns before processing
    context_window=10,             # Recent turns to analyze
    enable_auto_processing=True,   # Enable background thread
    output_operations=True         # Output detailed operation list
)
```

## Testing

### Test the Separated Flow

```bash
python test_separated_architecture.py --test flow
```

This test:
1. Creates a conversational agent
2. Has a multi-turn conversation
3. Manually triggers memory processing
4. Verifies memory updates
5. Tests memory recall in new session

### Compare Manual vs Background Processing

```bash
python test_separated_architecture.py --test comparison
```

Compares the two processing modes side-by-side.

## Migration from Previous Architecture

### Old Architecture (React Pattern)
- Single `UserMemoryAgent` class
- Memory tools (`add_memory`, `update_memory`, etc.) called during conversation
- Synchronous memory updates
- Memory decisions made per-turn

### New Architecture (Separated)
- `ConversationalAgent` for dialogue
- `BackgroundMemoryProcessor` for memory
- Asynchronous/batch memory updates
- Memory decisions made with full context

### Key Changes
1. **No more tool calls during conversation** - Agent focuses on dialogue
2. **Memory updates are batched** - Processor analyzes multiple turns together
3. **Background processing option** - Can run memory updates in separate thread
4. **Better context awareness** - Processor sees entire conversation history

## Best Practices

1. **Conversation Interval**: Default to 1 (process after each round) for immediate updates
2. **Context Window**: Include enough turns for meaningful analysis (8-12 turns)
3. **Update Threshold**: Balance between capturing information (0.6) and avoiding noise (0.8)
4. **Manual Triggers**: Use for important conversations or before session end
5. **Operation Review**: Monitor the operation list to understand memory changes

## Future Enhancements

- [ ] Memory conflict resolution strategies
- [ ] User-specific processing preferences
- [ ] Memory importance scoring
- [ ] Conversation summarization before processing
- [ ] Multi-model memory validation
- [ ] Memory versioning and rollback
