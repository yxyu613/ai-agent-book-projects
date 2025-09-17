# User Memory System with Separated Architecture

A sophisticated memory management system for AI agents with **separated conversation and memory processing**, enabling flexible and intelligent memory management across sessions.

## 🏗️ Architecture Overview

The system features a **separated architecture** that decouples conversation handling from memory management:

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
- Configurable conversation-based processing intervals

### 🧠 Multiple Memory Modes

1. **Notes-Based Memory** (`notes`)
   - Maintains a list of textual notes with session references
   - Each note includes content, tags, and source session ID
   - Automatic memory consolidation when limits are reached
   - Simple and interpretable memory structure

2. **Enhanced Notes** (`enhanced_notes`)
   - Paragraph-based memory with full context
   - More detailed information retention
   - Better for complex user information

3. **JSON Cards Memory** (`json_cards`)
   - Hierarchical two-level JSON structure
   - Organized by categories and subcategories
   - Each memory card contains value and session reference
   - Structured for complex information organization

4. **Advanced JSON Cards** (`advanced_json_cards`)
   - Complete card objects with full metadata
   - Enhanced structure for complex scenarios
   - Better conflict resolution and updates

### 🔍 Conversation History Management
- Session-based conversation storage and retrieval
- Optional integration with Dify API for vector-based search
- Fallback to keyword-based search when Dify is unavailable
- Contextual memory augmentation from past conversations

### 📊 Evaluation Framework Integration
- Comprehensive evaluation mode with structured test cases
- Three-layer test hierarchy (basic recall, contextual reasoning, cross-session synthesis)
- Automated scoring and evaluation metrics
- Integration with user-memory-evaluation framework
- See [EVALUATION_MODE.md](EVALUATION_MODE.md) for details

### 📈 LOCOMO Benchmark Support
- Comprehensive benchmark for memory system effectiveness
- Tests for:
  - Memory retention across sessions
  - Preference tracking and usage
  - Context switching capabilities
  - Memory update accuracy
  - Multi-session continuity
  - Complex reasoning with memory
  - Temporal awareness
  - Conflict resolution

### 🚀 Powered by Kimi K2 Model
- Uses Moonshot's Kimi K2 model (kimi-k2-0905-preview)
- Streaming response support for real-time interaction
- Optimized prompting for memory management
- Separate LLM calls for memory analysis

### 🛠️ Memory Operations
- **Add**: Create new memories
- **Update**: Modify existing memories based on new information
- **Delete**: Remove outdated or incorrect memories
- **Search**: Find relevant memories using keywords or semantic search

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

This demonstrates:
- Separated architecture with conversation and memory processing
- Sample conversations with memory extraction
- Memory persistence across sessions
- Clear display of memory operations

## Usage

### Interactive Mode (Recommended)
Start an interactive chat session with the separated architecture:

```bash
# With background memory processing (default)
python main.py --mode interactive --user john_doe

# Process memory every 2 conversations (default is 1)
python main.py --mode interactive --user john_doe --conversation-interval 2

# Without background processing (manual mode)
python main.py --mode interactive --user john_doe --background-processing False

# With different memory modes
python main.py --mode interactive --user jane_doe --memory-mode enhanced_notes
python main.py --mode interactive --user bob_smith --memory-mode json_cards
python main.py --mode interactive --user alice_wong --memory-mode advanced_json_cards
```

Commands during interactive session:
- Type messages normally to chat
- `memory` - Display current memory state
- `process` - Manually trigger memory processing (when background is disabled)
- `reset` - Start a new conversation session
- `quit` - Exit the session

### Single Message Mode
Send a single message and get a response:

```bash
python main.py --mode single --user alice --message "I work as a data scientist at TechCorp"
```

### Demo Mode
Run a comprehensive demonstration of all features:

```bash
python main.py --mode demo
```

### Evaluation Mode
Run structured tests from the evaluation framework:

```bash
# Interactive evaluation menu
python evaluation_main.py

# Run specific test
python evaluation_main.py --mode single --test-id layer1_01_bank_account

# Run all tests in a category
python evaluation_main.py --mode batch --category layer1
```

### Benchmark Mode
Run the LOCOMO benchmark:

```bash
# Run all benchmark tests
python main.py --mode benchmark

# Run specific test categories
python main.py --mode benchmark personal_info_retention preference_tracking
```

## Architecture Components

### Core Components

1. **`conversational_agent.py`** - Pure conversation handling
   - Natural dialogue management
   - Read-only memory access
   - Session management
   - Context integration

2. **`background_memory_processor.py`** - Memory analysis and updates
   - Full conversation context analysis
   - Batch processing capabilities
   - Thread-safe operation

3. **`agent.py`** - Legacy React-pattern agent (kept for compatibility)
   - Tool-based memory management
   - Direct memory updates during conversation
   - Single-turn decision making

4. **`memory_manager.py`** - Memory storage implementations
   - `NotesMemoryManager` - List-based memory
   - `EnhancedNotesMemoryManager` - Paragraph-based memory
   - `JSONMemoryManager` - Hierarchical JSON memory
   - `AdvancedJSONMemoryManager` - Full card object memory
   - Base abstract class for extensibility

5. **`conversation_history.py`** - Conversation management
   - Session-based conversation storage
   - Optional Dify integration for vector search
   - Fallback keyword search
   - Turn-by-turn tracking

6. **`locomo_benchmark.py`** - LOCOMO benchmark implementation
   - Comprehensive test suite
   - Performance metrics
   - Results analysis and reporting

7. **`memory_operation_formatter.py`** - Operation display utilities
   - Consistent formatting for memory operations
   - Operation summaries

8. **`config.py`** - Configuration management
   - Environment variable handling
   - Multiple memory modes
   - Default settings and validation

## Memory Processing Flow

### Separated Architecture Flow

1. **Conversation Phase**: User interacts with ConversationalAgent
   - Agent reads existing memories for context
   - Maintains natural conversation flow
   - Tracks all conversation turns

2. **Processing Trigger**: Based on configuration
   - After N conversation rounds (configurable)
   - Manual trigger via command
   - Automatic background processing

3. **Analysis Phase**: BackgroundMemoryProcessor analyzes conversation
   - Reviews entire conversation context
   - Identifies memorable information
   - Determines required operations (add/update/delete)

4. **Update Phase**: Memory operations are applied
   - Operations executed
   - Memory persistence to disk
   - Clear operation list output

5. **Integration**: Updated memories available for next conversation

## Configuration

Key configuration options in `.env`:

```env
# API Configuration
MOONSHOT_API_KEY=your_api_key_here

# Memory Mode Selection
MEMORY_MODE=notes  # Options: notes, enhanced_notes, json_cards, advanced_json_cards

# Memory Settings
MAX_MEMORY_ITEMS=100
MEMORY_UPDATE_TEMPERATURE=0.2

# Processing Configuration
CONVERSATION_INTERVAL=1  # Process after N conversation rounds
MIN_CONVERSATION_TURNS=1

# Dify Integration (Optional)
ENABLE_HISTORY_SEARCH=false
DIFY_API_KEY=your_dify_key
DIFY_DATASET_ID=your_dataset_id

# Model Settings
MODEL_TEMPERATURE=0.3
MODEL_MAX_TOKENS=4096
```

## Testing

The project includes comprehensive test suites:

```bash
# Test separated architecture flow
python test_separated_architecture.py

# Test all four memory modes
python test_four_modes.py

# Test memory operations
python test_memory_ops.py

# Test evaluation integration
python test_evaluation_integration.py

# Test React pattern (legacy)
python test_react_pattern.py

# Test advanced memory modes
python test_advanced_mode.py
```

## API Reference

### Creating a Conversational Agent

```python
from conversational_agent import ConversationalAgent, ConversationConfig
from config import MemoryMode

config = ConversationConfig(
    enable_memory_context=True,
    temperature=0.7,
    max_tokens=4096
)

agent = ConversationalAgent(
    user_id="unique_user_id",
    memory_mode=MemoryMode.NOTES,
    config=config,
    verbose=True  # Now defaults to True for streaming output
)
```

### Creating a Memory Processor

```python
from background_memory_processor import BackgroundMemoryProcessor, MemoryProcessorConfig

config = MemoryProcessorConfig(
    conversation_interval=1,
    update_threshold=0.7,
    enable_auto_processing=True
)

processor = BackgroundMemoryProcessor(
    user_id="unique_user_id",
    memory_mode=MemoryMode.NOTES,
    config=config
)
```

### Streaming Responses

```python
# Get streaming response
for chunk in agent.chat_stream("Hello, I'm Alice"):
    print(chunk, end='', flush=True)
```

### Memory Operations

```python
# Get memory summary
summary = agent.get_memory_summary()

# Process conversation manually
results = processor.process_conversation()
print(f"Operations: {results['operations']}")
print(f"Summary: {results['summary']}")

# Search memories
results = processor.search_relevant_memories("programming languages")
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
   - `get_summary()`

3. Register in `create_memory_manager()` factory function
4. Add to MemoryMode enum in config.py

### Custom Embedding Search

Replace `SimpleEmbeddingSearch` in `conversation_history.py` with:
- Sentence transformers
- Custom embedding models
- External vector databases (Pinecone, Weaviate, etc.)

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure `MOONSHOT_API_KEY` is set in `.env`
   - Check file is named `.env` (not `env.example`)

2. **Memory Not Persisting**
   - Check write permissions for `data/` directory
   - Verify `MEMORY_STORAGE_DIR` path exists

3. **Processing Not Triggering**
   - Check `conversation_interval` setting
   - Use `process` command to manually trigger
   - Verify background processing is enabled

4. **Dify Search Not Working**
   - Verify `DIFY_API_KEY` and `DIFY_DATASET_ID`
   - Set `ENABLE_HISTORY_SEARCH=true`

## Performance Considerations

- **Memory Size**: Keep `MAX_MEMORY_ITEMS` reasonable (50-200)
- **Update Frequency**: Balance between immediacy and efficiency
- **Context Length**: Monitor `MAX_CONTEXT_LENGTH` for token limits
- **Processing Interval**: Adjust based on conversation patterns

## Project Structure

```
user-memory/
├── main.py                          # Main entry point
├── quickstart.py                    # Quick demonstration
├── conversational_agent.py         # Conversation handling
├── background_memory_processor.py  # Memory processing
├── agent.py                        # Legacy React agent
├── memory_manager.py               # Memory storage
├── conversation_history.py         # History management
├── memory_operation_formatter.py   # Operation formatting
├── config.py                       # Configuration
├── locomo_benchmark.py             # Benchmark implementation
├── evaluation_main.py              # Evaluation mode entry
├── integrated_evaluation.py        # Evaluation integration
├── test_*.py                       # Test suites
├── ARCHITECTURE.md                 # Architecture details
├── EVALUATION_MODE.md              # Evaluation documentation
├── requirements.txt                # Dependencies
├── env.example                     # Environment template
└── data/                          # Storage directories
    ├── memories/                  # Memory files
    └── conversations/             # Conversation histories
```

## License

This project is part of the AI Agent practical training course.

## Acknowledgments

- Moonshot AI for the Kimi K2 model
- LOCOMO benchmark for evaluation framework
- Dify for vector search capabilities
- User Memory Evaluation Framework for structured testing