import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.research_state import ResearchState


CRITIC_SYSTEM_PROMPT = """You are a rigorous research critic and fact-checker. Your role is to:
1. Evaluate whether the research findings are comprehensive and factually sound
2. Identify any contradictions, gaps, or unsupported claims
3. Decide if the findings are sufficient to write a high-quality research report

You must return a JSON object with exactly these fields:
{{
  "is_sufficient": true or false,
  "critique": "Your detailed assessment as a string",
  "flagged_gaps": ["gap1", "gap2"]
}}

Be strict but fair. Only mark as sufficient if the findings are genuinely comprehensive."""

CRITIC_USER_PROMPT = """Research Topic: {topic}

Research Findings Gathered So Far:
{findings}

Evaluate these findings. Are they sufficient to write a comprehensive, accurate report?
Return your response as a valid JSON object only."""


def critic_agent(state: ResearchState) -> ResearchState:
    """
    Critic agent node.
    - Reviews the researcher's findings
    - Decides if they are sufficient or need another research iteration
    - Flags specific gaps if more research is needed
    """
    print(f"\n[Critic] Reviewing findings for: {state['topic']}")

    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIC_SYSTEM_PROMPT),
        ("human", CRITIC_USER_PROMPT),
    ])

    chain = prompt | llm

    combined_findings = "\n\n---\n\n".join(state.get("raw_findings", []))

    response = chain.invoke({
        "topic": state["topic"],
        "findings": combined_findings,
    })

    # Parse JSON response from critic
    try:
        # Strip markdown code fences if present
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())

        is_sufficient = bool(parsed.get("is_sufficient", False))
        critique = str(parsed.get("critique", "No critique provided."))
        flagged_gaps = list(parsed.get("flagged_gaps", []))

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [Critic] JSON parse error: {e}. Defaulting to sufficient.")
        is_sufficient = True
        critique = response.content
        flagged_gaps = []

    print(f"  [Critic] Sufficient: {is_sufficient}")
    if flagged_gaps:
        print(f"  [Critic] Gaps flagged: {flagged_gaps}")

    return {
        **state,
        "critique": critique,
        "is_sufficient": is_sufficient,
        "flagged_gaps": flagged_gaps,
        "iteration": state["iteration"] + 1,
        "status": "writing" if is_sufficient else "researching",
    }
