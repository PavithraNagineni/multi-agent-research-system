"""
Tests for the Multi-Agent Research System.
Run with: pytest tests/ -v
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
# State schema tests
# ─────────────────────────────────────────────

def test_research_state_fields():
    """ResearchState TypedDict has all required fields."""
    from state.research_state import ResearchState
    required_fields = [
        "topic", "raw_findings", "sources", "critique",
        "is_sufficient", "flagged_gaps", "final_report",
        "iteration", "max_iterations", "status", "mlflow_run_id"
    ]
    hints = ResearchState.__annotations__
    for field in required_fields:
        assert field in hints, f"Missing field: {field}"


# ─────────────────────────────────────────────
# Pipeline routing tests (no API calls)
# ─────────────────────────────────────────────

def test_route_after_critic_sufficient():
    """Router sends to writer when findings are sufficient."""
    # Import router function directly
    from pipeline import route_after_critic
    state = {
        "topic": "test",
        "is_sufficient": True,
        "iteration": 1,
        "max_iterations": 3,
        "raw_findings": [],
        "sources": [],
        "critique": "",
        "flagged_gaps": [],
        "final_report": "",
        "status": "critiquing",
        "mlflow_run_id": None,
    }
    assert route_after_critic(state) == "writer"


def test_route_after_critic_insufficient():
    """Router sends back to researcher when findings are insufficient."""
    from pipeline import route_after_critic
    os.environ["MAX_RESEARCH_ITERATIONS"] = "3"
    state = {
        "topic": "test",
        "is_sufficient": False,
        "iteration": 1,
        "max_iterations": 3,
        "raw_findings": [],
        "sources": [],
        "critique": "",
        "flagged_gaps": [],
        "final_report": "",
        "status": "critiquing",
        "mlflow_run_id": None,
    }
    assert route_after_critic(state) == "researcher"


def test_route_after_critic_max_iterations():
    """Router sends to writer when max iterations is reached regardless of sufficiency."""
    from pipeline import route_after_critic
    os.environ["MAX_RESEARCH_ITERATIONS"] = "3"
    state = {
        "topic": "test",
        "is_sufficient": False,
        "iteration": 3,  # At max
        "max_iterations": 3,
        "raw_findings": [],
        "sources": [],
        "critique": "",
        "flagged_gaps": [],
        "final_report": "",
        "status": "critiquing",
        "mlflow_run_id": None,
    }
    assert route_after_critic(state) == "writer"


# ─────────────────────────────────────────────
# Graph structure tests
# ─────────────────────────────────────────────

def test_graph_builds_without_error():
    """LangGraph compiles successfully."""
    from pipeline import build_graph
    app = build_graph()
    assert app is not None


def test_graph_has_correct_nodes():
    """Graph contains all three agent nodes."""
    from pipeline import build_graph
    app = build_graph()
    node_names = list(app.nodes.keys())
    assert "researcher" in node_names
    assert "critic" in node_names
    assert "writer" in node_names


# ─────────────────────────────────────────────
# FastAPI tests (no real LLM calls)
# ─────────────────────────────────────────────

def test_api_health_endpoint():
    """Health endpoint returns 200."""
    from fastapi.testclient import TestClient
    from api.app import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_api_root_endpoint():
    """Root endpoint returns 200 with expected keys."""
    from fastapi.testclient import TestClient
    from api.app import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs" in data
    assert "research_endpoint" in data


def test_api_research_requires_api_keys():
    """Research endpoint returns 500 when API keys are missing."""
    from fastapi.testclient import TestClient
    from api.app import app

    # Remove keys temporarily
    openai_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        client = TestClient(app)
        response = client.post("/research", json={"topic": "test topic for validation"})
        assert response.status_code == 500
    finally:
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key


def test_api_research_validates_topic_length():
    """Research endpoint rejects topics that are too short."""
    from fastapi.testclient import TestClient
    from api.app import app
    client = TestClient(app)
    response = client.post("/research", json={"topic": "hi"})
    assert response.status_code == 422  # Pydantic validation error
