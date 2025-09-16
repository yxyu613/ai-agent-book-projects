# User Memory System with Separated Architecture

A sophisticated memory management system for AI agents with **separated conversation and memory processing**, enabling more flexible and intelligent memory management across sessions.

## 🆕 New Architecture (v2.0)

The system now features a **separated architecture** that decouples conversation handling from memory management:

- **ConversationalAgent**: Focuses purely on natural dialogue with users
- **BackgroundMemoryProcessor**: Analyzes full conversation context to intelligently update memories
- **Flexible Processing**: Supports manual, automatic, or background memory updates
- **Better Context Awareness**: Memory decisions based on entire conversations, not single messages

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Features

### 🏗️ Separated Processing Architecture
- Conversation and memory management are completely decoupled
- Background thread option for asynchronous memory updates
- Manual trigger option for controlled memory processing
- Full conversation context analysis for better memory decisions

### 🧠 Dual Memory Mechanisms

1. **Notes-Based Memory**
   - Maintains a list of textual notes with session references
   - Each note includes content, tags, and source session ID
   - Automatic memory consolidation when limits are reached
   - Simple and interpretable memory structure

2. **JSON Cards Memory**
   - Hierarchical two-level JSON structure
   - Organized by categories and subcategories
   - Each memory card contains value and session reference
   - Structured for complex information organization

### 🔍 Conversation History Search
- Optional integration with Dify API for vector-based search
- Fallback to keyword-based search when Dify is unavailable
- Session-aware conversation retrieval
- Contextual memory augmentation

### 📊 LOCOMO Benchmark Integration
- Comprehensive evaluation of memory system effectiveness
- Tests for:
  - Memory retention
  - Preference tracking
  - Context switching
  - Memory updates
  - Multi-session continuity
  - Complex reasoning with memory
  - Temporal awareness
  - Conflict resolution

### 🚀 Kimi K2 Model Integration
- Powered by Moonshot's Kimi K2 model (kimi-k2-0905-preview)
- Streaming response support for real-time interaction
- Optimized prompting for memory management
- Separate LLM calls for memory updates

## Installation

1. Clone the repository:
```bash
cd projects/week2/user-memory
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your API keys
```

4. Get your Moonshot API key:
   - Visit https://platform.moonshot.cn/
   - Create an account and generate an API key
   - Add it to your `.env` file

## Quick Start

### Run the quickstart demo:
```bash
python quickstart.py
```

This will:
- Create a demo user with notes-based memory
- Run sample conversations
- Demonstrate memory persistence across sessions
- Display the current memory state

## Usage

### Interactive Session
Start an interactive chat session with the new separated architecture:

```bash
# With background memory processing (default)
python main.py --mode interactive --user john_doe

# Process memory every 2 conversations (default is 1)
python main.py --mode interactive --user john_doe --conversation-interval 2

# Without background processing (manual mode)
python main.py --mode interactive --user john_doe --background-processing False

# With JSON cards memory
python main.py --mode interactive --user jane_doe --memory-mode json_cards
```

Commands during interactive session:
- Type messages normally to chat
- `memory` - Display current memory state
- `process` - Manually trigger memory processing (when background is disabled)
- `reset` - Start a new conversation session
- `quit` - Exit the session

### Demo Mode
Run a comprehensive demonstration:

```bash
python main.py demo
```

This demonstrates both memory modes with sample conversations.

### Benchmark Mode
Run the LOCOMO benchmark to evaluate memory effectiveness:

```bash
# Run all tests
python main.py benchmark

# Run specific tests
python main.py benchmark personal_info_retention preference_tracking
```

## Architecture

### Core Components

1. **`agent.py`** - Main agent implementation with Kimi K2 integration
   - Streaming response handling
   - Memory-augmented prompting
   - Session management

2. **`memory_manager.py`** - Memory management implementations
   - `NotesMemoryManager` - List-based memory
   - `JSONMemoryManager` - Hierarchical JSON memory
   - Base abstract class for extensibility

3. **`conversation_history.py`** - Conversation history management
   - Session-based conversation storage
   - Optional Dify integration for vector search
   - Fallback keyword search

4. **`locomo_benchmark.py`** - LOCOMO benchmark implementation
   - Comprehensive test suite
   - Performance metrics
   - Results analysis

5. **`config.py`** - Configuration management
   - Environment variable handling
   - Default settings
   - Validation

## Memory Update Process

1. **Conversation Analysis**: After each user-assistant interaction, the system analyzes the conversation for memorable information.

2. **LLM-Based Extraction**: A separate LLM call determines what information should be:
   - Created as new memory
   - Updated in existing memory
   - Deleted from memory

3. **Memory Persistence**: Updates are immediately saved to disk for persistence across sessions.

4. **Context Integration**: Memory is automatically included in the system prompt for future interactions.

## Configuration

Key configuration options in `.env`:

```env
# Memory Mode Selection
MEMORY_MODE=notes  # or json_cards

# Memory Limits
MAX_MEMORY_ITEMS=100
MEMORY_UPDATE_TEMPERATURE=0.2

# Dify Integration (Optional)
ENABLE_HISTORY_SEARCH=true
DIFY_API_KEY=your_dify_key
DIFY_DATASET_ID=your_dataset_id

# Model Settings
MODEL_TEMPERATURE=0.3
MODEL_MAX_TOKENS=2000
```

## Benchmark Results

The LOCOMO benchmark evaluates:

- **Memory Retention**: 
  - Ability to remember user information
  - Persistence across sessions

- **Preference Tracking**:
  - Recording user preferences
  - Using preferences in recommendations

- **Context Management**:
  - Handling topic switches
  - Maintaining relevant context

- **Memory Updates**:
  - Correcting outdated information
  - Resolving conflicts

Results are saved to `results/locomo/` with detailed metrics:
- Success rate per test category
- Average response times
- Memory operation counts
- Detailed test breakdowns

## API Reference

### Creating an Agent

```python
from agent import UserMemoryAgent
from config import MemoryMode

agent = UserMemoryAgent(
    user_id="unique_user_id",
    memory_mode=MemoryMode.NOTES,
    enable_streaming=True,
    verbose=False
)
```

### Streaming Responses

```python
for chunk in agent.chat_stream("Hello, I'm Alice"):
    if chunk['type'] == 'content':
        print(chunk['content'], end='', flush=True)
```

### Memory Operations

```python
# Get memory summary
summary = agent.get_memory_summary()

# Search memories
results = agent.search_memories("programming languages")

# Start new session
session_id = agent.start_session()
```

## Extending the System

### Adding New Memory Modes

1. Create a new class inheriting from `BaseMemoryManager`
2. Implement required methods:
   - `load_memory()`
   - `save_memory()`
   - `add_memory()`
   - `update_memory()`
   - `delete_memory()`
   - `get_context_string()`
   - `search_memories()`

3. Register in `create_memory_manager()` factory function

### Custom Embedding Search

Replace `SimpleEmbeddingSearch` in `conversation_history.py` with:
- Sentence transformers
- Custom embedding models
- External vector databases

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure `MOONSHOT_API_KEY` is set in `.env`
   - Check file is named `.env` (not `env.example`)

2. **Memory Not Persisting**
   - Check write permissions for `data/` directory
   - Verify `MEMORY_STORAGE_DIR` path exists

3. **Streaming Not Working**
   - Ensure `enable_streaming=True` when creating agent
   - Check terminal supports flush output

4. **Dify Search Not Working**
   - Verify `DIFY_API_KEY` and `DIFY_DATASET_ID`
   - Set `ENABLE_HISTORY_SEARCH=true`

## Performance Considerations

- **Memory Size**: Keep `MAX_MEMORY_ITEMS` reasonable (50-200)
- **Update Frequency**: Memory updates happen after each turn
- **Context Length**: Monitor `MAX_CONTEXT_LENGTH` for token limits
- **Streaming**: Enable for better user experience
- **Search**: Dify integration improves search quality but adds latency

## License

This project is part of the AI Agent practical training course.

## Acknowledgments

- Moonshot AI for the Kimi K2 model
- LOCOMO benchmark for evaluation framework
- Dify for vector search capabilities
