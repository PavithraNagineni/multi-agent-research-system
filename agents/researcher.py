import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.research_state import ResearchState
from tools.search_tool import search_web


RESEARCHER_SYSTEM_PROMPT = """You are an expert research agent. Your job is to gather comprehensive, 
accurate information on a given topic using web search results.

Given a research topic and any previously flagged gaps, your responsibilities are:
1. Identify the most important sub-questions to answer
2. Synthesize the provided search results into clear, structured findings
3. Ensure you cover factual data, recent developments, and multiple perspectives

Always be objective and cite the key points from the search results.
Format your findings as clear bullet points."""


RESEARCHER_USER_PROMPT = """Research Topic: {topic}

Previously flagged gaps to address (if any): {gaps}

Here are the web search results gathered for this topic:

{search_results}

Based on these results, provide comprehensive research findings as structured bullet points.
Focus on facts, data, and key insights."""


def researcher_agent(state: ResearchState) -> ResearchState:
    """
    Researcher agent node.
    - Calls Tavily to search the web on the topic (and any flagged gaps)
    - Uses an LLM to synthesize raw search results into structured findings
    - Updates state with findings and sources
    """
    print(f"\n[Researcher] Iteration {state['iteration'] + 1} — researching: {state['topic']}")

    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Build queries: main topic + any flagged gaps from previous critic pass
    queries = [state["topic"]]
    if state.get("flagged_gaps"):
        queries.extend(state["flagged_gaps"][:2])  # Max 2 gap queries to control cost

    all_results = []
    all_sources = []

    for query in queries:
        print(f"  [Researcher] Searching: '{query}'")
        data = search_web(query, max_results=4)
        all_results.extend(data["results"])
        all_sources.extend(data["sources"])

    # Deduplicate sources
    all_sources = list(dict.fromkeys(all_sources))

    # LLM synthesizes raw search results into structured findings
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCHER_SYSTEM_PROMPT),
        ("human", RESEARCHER_USER_PROMPT),
    ])

    chain = prompt | llm

    response = chain.invoke({
        "topic": state["topic"],
        "gaps": ", ".join(state.get("flagged_gaps", [])) or "None",
        "search_results": "\n\n".join(all_results[:15]),  # Cap to avoid token overflow
    })

    findings_text = response.content

    # Merge with any prior findings (accumulate across iterations)
    existing = state.get("raw_findings", [])
    updated_findings = existing + [findings_text]

    print(f"  [Researcher] Found {len(all_sources)} sources.")

    return {
        **state,
        "raw_findings": updated_findings,
        "sources": all_sources,
        "status": "critiquing",
    }
