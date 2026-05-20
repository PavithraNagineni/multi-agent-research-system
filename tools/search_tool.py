import os
from typing import List, Dict
from tavily import TavilyClient
from langchain.tools import tool


def get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set in environment variables")
    return TavilyClient(api_key=api_key)


def search_web(query: str, max_results: int = 5) -> Dict:
    """
    Calls Tavily API to search the web for a given query.
    Returns a dict with 'results' (list of snippets) and 'sources' (list of URLs).
    """
    client = get_tavily_client()
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
    )

    results = []
    sources = []

    # Tavily answer (synthesized summary if available)
    if response.get("answer"):
        results.append(f"[Summary]: {response['answer']}")

    # Individual result snippets
    for r in response.get("results", []):
        snippet = f"[{r.get('title', 'Untitled')}]: {r.get('content', '')}"
        results.append(snippet)
        if r.get("url"):
            sources.append(r["url"])

    return {"results": results, "sources": sources}


@tool
def tavily_search_tool(query: str) -> str:
    """
    Search the web using Tavily API. Use this to find current, factual information
    on a research topic. Returns a formatted string of findings.
    """
    data = search_web(query)
    formatted = "\n\n".join(data["results"])
    return formatted
