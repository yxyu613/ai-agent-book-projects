# Attention Visualization

Interactive visualization tool for exploring attention mechanisms in language models. Each agent run creates a unique trajectory that can be viewed and compared in the frontend.

## Overview

This project provides an interactive way to understand how language models allocate attention when processing different types of queries. Each run of the agent creates a new trajectory file that captures:
- The input query and model response
- Token-by-token attention weights
- Attention patterns across layers and heads
- Statistical analysis of attention distribution

## Architecture

The system follows a simple architecture similar to the attention-hallucination-detection project:
1. **Agent generates trajectories**: Run `agent.py` or `main.py` to generate new trajectories
2. **JSON storage**: Each trajectory is saved as a unique JSON file in `frontend/public/trajectories/`
3. **Frontend visualization**: React app loads and displays all trajectories with tab navigation

## Quick Start

### Option 1: Using the Start Script (Recommended)

```bash
./start.sh
```

This script will:
1. Check for existing trajectories
2. Run a demo if no trajectories exist
3. Install frontend dependencies (if needed)
4. Start the development server

Then visit http://localhost:3000 to view the visualizations.

### Option 2: Manual Setup

1. **Generate trajectories:**
```bash
# run the ReAct agent with tool use
python main.py
```

Each run creates a new trajectory file with a unique timestamp.

2. **Start the frontend:**
```bash
cd frontend
npm install  # First time only
npm run dev
```

3. **Open your browser:**
Navigate to http://localhost:3000

## Project Structure

```
attention_visualization/
├── agent.py               # Core attention tracking agent
├── main.py               # ReAct agent with tool calling
├── tools.py              # Tool implementations
├── visualization.py      # Visualization utilities
├── frontend/             # Next.js frontend
│   ├── pages/           # React pages
│   ├── components/      # Visualization components
│   └── public/
│       └── trajectories/  # Stored trajectory JSONs
│           ├── trajectory_YYYYMMDD_HHMMSS.json
│           └── manifest.json  # Index of all trajectories
└── start.sh             # Quick start script
```

## How It Works

### 1. Trajectory Generation
When you run `main.py`:
- The agent processes various test queries
- Attention weights are captured at each generation step
- Results are saved to `frontend/public/trajectories/` with unique timestamps
- A manifest file tracks all available trajectories

### 2. Data Format
Each trajectory JSON contains:
```json
{
  "id": "20250914_123456",
  "timestamp": "2025-09-14 12:34:56",
  "test_case": {
    "category": "Math",
    "query": "What is 25 * 37?",
    "description": "Agent trajectory from..."
  },
  "response": "The answer is...",
  "tokens": ["What", "is", "25", ...],
  "attention_data": {
    "tokens": [...],
    "attention_matrix": [[...]],
    "num_layers": 1,
    "num_heads": 16
  },
  "metadata": {...}
}
```

### 3. Frontend Visualization
The React frontend:
- Loads all trajectories from the manifest
- Provides tabs to switch between different runs
- Displays attention heatmaps, token analysis, and statistics
- Updates automatically when new trajectories are generated

## Features

- **Multiple Trajectories**: Each agent run creates a new trajectory file
- **Tab Navigation**: Easy switching between different agent runs
- **Attention Heatmap**: Interactive visualization of token-to-token attention
- **Token Analysis**: View individual tokens and their attention patterns
- **Statistical Metrics**: Average attention, maximum attention, and entropy
- **Category Support**: Queries are categorized (Math, Knowledge, Reasoning, Code, Creative)
- **Persistent Storage**: All trajectories are saved and can be revisited

## Generating Custom Trajectories

### Using agent.py
Edit the `demonstrate_attention_tracking()` function to add custom queries:

```python
test_prompts = [
    ("Your custom query here", "Category"),
    # Add more queries...
]
```

### Using main.py
The ReAct agent demonstrates tool use and multi-step reasoning. Edit the test queries in `demonstrate_react_agent()`.

### Manual Generation
You can also use the agent programmatically:

```python
from agent import AttentionVisualizationAgent

agent = AttentionVisualizationAgent()
result = agent.generate_with_attention(
    "Your query here",
    max_new_tokens=100,
    temperature=0.3,
    save_trajectory=True,
    category="Custom"
)
```

## Requirements

### Python
- Python 3.10+
- PyTorch
- Transformers
- See `requirements.txt` for full list

### Frontend
- Node.js 14+
- npm or yarn
- See `frontend/package.json` for dependencies

## Installation

1. **Clone the repository**

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies:**
```bash
cd frontend
npm install
```

## Tips

- **Generate Multiple Runs**: Run the agent multiple times to compare different trajectories
- **Compare Responses**: Use the tab interface to compare how the model handles similar queries
- **Analyze Patterns**: Look for attention patterns in different query types
- **Performance**: First run downloads the model (~1-2GB). GPU/MPS recommended for speed.

## Notes

- Each trajectory is timestamped to ensure uniqueness
- The manifest keeps track of the last 50 trajectories
- Trajectories persist between sessions
- The frontend automatically discovers new trajectories via the manifest