from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """
    Shared state passed between all agents in the LangGraph pipeline.
    Each field is updated by a specific agent and read by others.
    """
    # Input
    topic: str                          # The research query from the user

    # Researcher output
    raw_findings: List[str]             # List of search result snippets
    sources: List[str]                  # URLs of sources found

    # Critic output
    critique: str                       # Critic's assessment of findings
    is_sufficient: bool                 # Whether findings are good enough to write
    flagged_gaps: List[str]             # Topics the researcher should dig deeper on

    # Writer output
    final_report: str                   # The synthesized markdown report

    # Orchestration
    iteration: int                      # How many research loops have happened
    max_iterations: int                 # Cap to prevent infinite loops
    status: str                         # "researching" | "critiquing" | "writing" | "done"

    # MLflow run tracking
    mlflow_run_id: Optional[str]        # MLflow run ID for logging
