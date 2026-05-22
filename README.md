# Multi-Agent Research System

An autonomous research pipeline built with **LangChain**, **LangGraph**, **FastAPI**, and **MLflow**. Three specialized AI agents collaborate in a stateful graph to research any topic, fact-check findings, and produce a structured report.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│       LangGraph State Machine       │
│                                     │
│  ┌────────────┐                     │
│  │ Researcher │◄──────────────┐     │
│  │   Agent    │               │     │
│  └─────┬──────┘               │     │
│        │ findings              │     │
│        ▼                       │     │
│  ┌────────────┐   insufficient │     │
│  │   Critic   │────────────────┘     │
│  │   Agent    │                      │
│  └─────┬──────┘                      │
│        │ sufficient                  │
│        ▼                            │
│  ┌────────────┐                     │
│  │   Writer   │                     │
│  │   Agent    │                     │
│  └─────┬──────┘                     │
└────────┼────────────────────────────┘
         │
         ▼
   Final Report (Markdown)
         │
         ▼
   FastAPI /research endpoint
         │
         ▼
   MLflow experiment tracking
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Researcher** | Searches the web, synthesizes findings | Tavily Search API |
| **Critic** | Fact-checks findings, flags gaps | LLM reasoning |
| **Writer** | Produces structured markdown report | LLM generation |

### LangGraph flow

- Researcher → Critic is always sequential
- Critic → Writer if findings are sufficient OR max iterations reached
- Critic → Researcher if gaps remain and iterations < max (feedback loop)

---

## Project Structure

```
multi_agent_research_system/
├── agents/
│   ├── researcher.py      # Web search + LLM synthesis
│   ├── critic.py          # JSON-structured evaluation
│   └── writer.py          # Final report generation
├── state/
│   └── research_state.py  # Shared TypedDict state schema
├── tools/
│   └── search_tool.py     # Tavily API wrapper + LangChain tool
├── tracking/
│   └── mlflow_tracker.py  # MLflow run logging
├── api/
│   └── app.py             # FastAPI application
├── tests/
│   └── test_pipeline.py   # Pytest test suite
├── pipeline.py            # LangGraph graph builder + CLI
├── main.py                # Uvicorn server entrypoint
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (for containerized run)
- An **OpenAI API key** — [platform.openai.com](https://platform.openai.com)
- A **Tavily API key** — [tavily.com](https://tavily.com) (free tier available)

---

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/yourusername/multi-agent-research-system.git
cd multi_agent_research_system
```

### 2. Create your environment file

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=gpt-4o-mini
MAX_RESEARCH_ITERATIONS=3
```

---

## Running the Project

### Option A — Local Python (Recommended for development)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start MLflow tracking server (in a separate terminal)
mlflow server --host 0.0.0.0 --port 5000

# Run the FastAPI server
python main.py
```

API will be live at: `http://localhost:8000`
Swagger docs at: `http://localhost:8000/docs`
MLflow UI at: `http://localhost:5000`

---

### Option B — Docker Compose (Recommended for production)

```bash
# Build and start all services (API + MLflow)
docker-compose up --build

# Run in background
docker-compose up --build -d

# View logs
docker-compose logs -f research-api

# Stop everything
docker-compose down
```

---

## Usage

### CLI — Run from terminal directly

```bash
# Activate venv first
source venv/bin/activate

# Run with a topic
python pipeline.py "The impact of AI on healthcare in 2024"

# Or any topic
python pipeline.py "Quantum computing breakthroughs"
```

---

### REST API

#### Check health
```bash
curl http://localhost:8000/health
```

#### Run a research task
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "What are the latest advancements in quantum computing?",
    "max_iterations": 2
  }'
```

#### Example response
```json
{
  "topic": "What are the latest advancements in quantum computing?",
  "final_report": "# Quantum Computing: Latest Advancements\n\n## Executive Summary\n...",
  "sources": [
    "https://example.com/quantum-article-1",
    "https://example.com/quantum-article-2"
  ],
  "iterations_completed": 2,
  "sources_count": 8,
  "status": "done",
  "duration_seconds": 42.5,
  "mlflow_run_id": "abc123def456"
}
```

#### Interactive Swagger UI
Open `http://localhost:8000/docs` in your browser to explore and test all endpoints interactively.

---

## Running Tests

```bash
# Activate venv
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short
```

Tests cover:
- State schema field validation
- LangGraph routing logic (no API calls)
- Graph compilation and node presence
- FastAPI endpoint responses
- Input validation

---

## MLflow Experiment Tracking

Every pipeline run is logged to MLflow automatically with:

| What's logged | Type |
|---------------|------|
| Research topic | Parameter |
| Model name | Parameter |
| Max iterations configured | Parameter |
| Iterations completed | Metric (per step) |
| Sources found | Metric (per step) |
| Findings length | Metric (per step) |
| Critic's assessment | Text artifact |
| Final report | Text artifact (`.md`) |
| All sources | Text artifact |

View all runs at `http://localhost:5000` after starting the MLflow server.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `TAVILY_API_KEY` | required | Tavily search API key |
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI model to use |
| `MAX_RESEARCH_ITERATIONS` | `3` | Max researcher-critic loops |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph (StateGraph) |
| LLM chaining | LangChain |
| Web search | Tavily API |
| LLM provider | OpenAI (GPT-4o-mini) |
| API layer | FastAPI + Uvicorn |
| Experiment tracking | MLflow |
| Containerization | Docker + Docker Compose |
| Testing | Pytest |

---

## Extending the System

### Add a new agent
1. Create `agents/your_agent.py` with a function `your_agent(state: ResearchState) -> ResearchState`
2. Register it in `pipeline.py`: `graph.add_node("your_agent", your_agent)`
3. Add edges connecting it in the graph

### Add a new tool
1. Create the tool in `tools/your_tool.py` using `@tool` decorator from LangChain
2. Import and use it inside any agent

### Switch LLM provider
Replace `ChatOpenAI` with any LangChain-compatible chat model:
```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-5")
```
"# multi-agent-research-system" 

## Output
Post/Research
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/2abec41b-9616-47ff-ad6d-4413404d2603" />

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/eda7bb15-5727-4dd5-aaa0-2168843a7133" />


## Author
   Pavithra Nagineni

