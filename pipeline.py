import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state.research_state import ResearchState
from agents.researcher import researcher_agent
from agents.critic import critic_agent
from agents.writer import writer_agent
from tracking.mlflow_tracker import start_run, log_iteration, log_final_report, end_run_on_error

load_dotenv()


# ─────────────────────────────────────────────
# Routing logic — this is the LangGraph "edge"
# ─────────────────────────────────────────────

def route_after_critic(state: ResearchState) -> str:
    """
    Conditional edge function.
    After the Critic runs, decide next node:
      - If findings are sufficient → go to Writer
      - If max iterations reached  → go to Writer anyway (best-effort)
      - Otherwise                  → loop back to Researcher
    """
    max_iter = int(os.getenv("MAX_RESEARCH_ITERATIONS", 3))

    if state.get("is_sufficient"):
        print(f"\n[Router] Findings sufficient → routing to Writer")
        return "writer"

    if state.get("iteration", 0) >= max_iter:
        print(f"\n[Router] Max iterations ({max_iter}) reached → routing to Writer (best-effort)")
        return "writer"

    print(f"\n[Router] Findings insufficient → routing back to Researcher (iter {state['iteration']})")
    return "researcher"


# ─────────────────────────────────────────────
# Wrapper nodes — inject MLflow logging
# ─────────────────────────────────────────────

def researcher_node(state: ResearchState) -> ResearchState:
    return researcher_agent(state)


def critic_node(state: ResearchState) -> ResearchState:
    new_state = critic_agent(state)
    try:
        log_iteration(new_state)
    except Exception as e:
        print(f"[MLflow] Logging warning: {e}")
    return new_state


def writer_node(state: ResearchState) -> ResearchState:
    new_state = writer_agent(state)
    try:
        log_final_report(new_state)
    except Exception as e:
        print(f"[MLflow] Logging warning: {e}")
    return new_state


# ─────────────────────────────────────────────
# Build the LangGraph StateGraph
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs the LangGraph state machine.

    Graph structure:
        START → researcher → critic ─┬→ writer → END
                    ↑                │
                    └────────────────┘  (if not sufficient and iter < max)
    """
    graph = StateGraph(ResearchState)

    # Register agent nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    # Entry point
    graph.set_entry_point("researcher")

    # Linear edge: researcher always goes to critic
    graph.add_edge("researcher", "critic")

    # Conditional edge: critic decides to loop or proceed
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "researcher",
            "writer": "writer",
        }
    )

    # Terminal edge
    graph.add_edge("writer", END)

    return graph.compile()


# ─────────────────────────────────────────────
# Public API — run a full research pipeline
# ─────────────────────────────────────────────

def run_research_pipeline(topic: str) -> ResearchState:
    """
    Execute the full multi-agent research pipeline for a given topic.
    Returns the final ResearchState containing the report, sources, etc.
    """
    print(f"\n{'='*60}")
    print(f"  Starting Research Pipeline")
    print(f"  Topic: {topic}")
    print(f"{'='*60}")

    # Start MLflow run
    try:
        run_id = start_run(topic)
    except Exception as e:
        print(f"[MLflow] Could not start run: {e}")
        run_id = None

    # Initial state
    initial_state: ResearchState = {
        "topic": topic,
        "raw_findings": [],
        "sources": [],
        "critique": "",
        "is_sufficient": False,
        "flagged_gaps": [],
        "final_report": "",
        "iteration": 0,
        "max_iterations": int(os.getenv("MAX_RESEARCH_ITERATIONS", 3)),
        "status": "researching",
        "mlflow_run_id": run_id,
    }

    # Build and run the graph
    app = build_graph()

    try:
        final_state = app.invoke(initial_state)
        print(f"\n{'='*60}")
        print(f"  Pipeline Complete!")
        print(f"  Iterations: {final_state.get('iteration')}")
        print(f"  Sources: {len(final_state.get('sources', []))}")
        print(f"  Report length: {len(final_state.get('final_report', ''))} chars")
        print(f"{'='*60}\n")
        return final_state

    except Exception as e:
        print(f"\n[Pipeline] ERROR: {e}")
        try:
            end_run_on_error(str(e))
        except Exception:
            pass
        raise


# ─────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The impact of large language models on software engineering"

    result = run_research_pipeline(topic)

    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(result["final_report"])

    print("\n" + "="*60)
    print("SOURCES")
    print("="*60)
    for i, src in enumerate(result["sources"], 1):
        print(f"{i}. {src}")
