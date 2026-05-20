import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pipeline import run_research_pipeline

load_dotenv()


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The research topic or question to investigate",
        example="What are the latest advancements in quantum computing?"
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum research-critique loops (1-5)"
    )


class ResearchResponse(BaseModel):
    topic: str
    final_report: str
    sources: list[str]
    iterations_completed: int
    sources_count: int
    status: str
    duration_seconds: float
    mlflow_run_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    max_iterations: int
    mlflow_tracking_uri: str


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Multi-Agent Research System starting up...")
    yield
    print("[API] Shutting down...")


app = FastAPI(
    title="Multi-Agent Research System",
    description="""
    An autonomous multi-agent research pipeline built with LangChain and LangGraph.
    
    ## How it works
    1. **Researcher Agent** searches the web using Tavily API
    2. **Critic Agent** evaluates findings and flags gaps
    3. **Writer Agent** synthesizes a structured markdown report
    4. All runs are tracked in **MLflow**
    
    The system iterates until findings are sufficient or max_iterations is reached.
    """,
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check API health and configuration."""
    return HealthResponse(
        status="healthy",
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        max_iterations=int(os.getenv("MAX_RESEARCH_ITERATIONS", 3)),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
def run_research(request: ResearchRequest):
    """
    Run the full multi-agent research pipeline on a topic.
    
    This endpoint is synchronous — it waits for the full pipeline to complete
    before returning. Research typically takes 30-90 seconds depending on
    the topic complexity and number of iterations.
    """
    # Validate API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server.")
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY not configured on server.")

    # Override max iterations from request
    os.environ["MAX_RESEARCH_ITERATIONS"] = str(request.max_iterations)

    start_time = time.time()

    try:
        result = run_research_pipeline(request.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    duration = round(time.time() - start_time, 2)

    return ResearchResponse(
        topic=result["topic"],
        final_report=result["final_report"],
        sources=result["sources"],
        iterations_completed=result["iteration"],
        sources_count=len(result["sources"]),
        status=result["status"],
        duration_seconds=duration,
        mlflow_run_id=result.get("mlflow_run_id"),
    )


@app.get("/", tags=["System"])
def root():
    """Redirect info to docs."""
    return {
        "message": "Multi-Agent Research System API",
        "docs": "/docs",
        "health": "/health",
        "research_endpoint": "POST /research",
    }
