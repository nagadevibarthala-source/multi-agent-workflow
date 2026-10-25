import time
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# Initialize local Ollama model
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

# Shared data used by all agents

class AgentState(TypedDict):
    topic: str
    research: str
    fact_check: str
    final_report: str
    review_status: str
    review_count: int
    feedback: str



# Agent 1: Researcher
def researcher(state: AgentState):
    print("\n[Researcher] Local AI research started...")

    topic = state["topic"]
    feedback = state.get("feedback", "")

    prompt = f"""
    Research the following topic: {topic}

    Previous reviewer feedback:
    {feedback}

    Explain in simple language:
    1. Introduction
    2. Important facts and key points
    3. Real-world applications
    4. Benefits
    5. Risks and limitations

    If reviewer feedback is provided, improve the research
    by addressing that feedback.

    Do not invent facts. Clearly mention anything uncertain.
    """

    try:
        response = llm.invoke(prompt)
        research_text = response.content or ""

        print("[Researcher] Research completed successfully.")

        return {"research": research_text}

    except Exception as error:
        print(f"[Researcher] Failed: {error}")
        raise

   
# Agent 2: Reviewer
def fact_checker(state: AgentState):
    print("\n[Reviewer] Reviewing research...")

    
    prompt = f"""
    Review this research about: {state['topic']}

    Research:
    {state['research']}

    Previous feedback:
    {state['feedback']}

    Check:
    1. Is the report clear and complete?
    2. Are important claims supported by evidence?
    3. Are risks and limitations included?

    If the research is acceptable, respond exactly:
    APPROVED
    Feedback: None

    Otherwise, respond exactly:
    REJECTED
    Feedback: explain what needs improvement.

    Do not claim that you independently verified facts.
    """


    response = llm.invoke(prompt)
    review_text = response.content or ""

    print("\n[Reviewer] Decision:")
    print(review_text)

    if "APPROVED" in review_text.upper():
        status = "approved"
    else:
        status = "rejected"

    return {
        "fact_check": review_text,
        "review_status": status,
        "review_count": state["review_count"] + 1,
        "feedback": review_text
    }

# Agent 3: Report Writer
def report_writer(state: AgentState):
    print("\n[Report Writer] Creating final report with local Ollama...")

    prompt = f"""
    {state['topic']}

    Use the research below:
    {state['research']}

    Fact-check review:
    {state['fact_check']}

    Include these sections:
    1. Introduction
    2. Key Findings
    3. Real-World Applications
    4. Benefits
    5. Risks and Limitations
    6. Conclusion

    Instructions:
    - Use clear headings and simple language.
    - Include useful details in every section.
    - Consider the fact-check review.
    - Do not invent statistics or claim unsupported facts.
    - Return the complete report.
    """

    try:
        response = llm.invoke(prompt)
        final_report = response.content or "No final report generated."
        print("[Report Writer] Final report created successfully.")

        return {"final_report": final_report}

    except Exception as error:
        print(f"[Report Writer] Failed: {error}")
        raise


# Handle repeated review rejection
def review_failed(state: AgentState):
    print("\n[Workflow] Research rejected twice.")
    print("[Workflow] Stopping without generating the final report.")

    return {
        "final_report": (
            "Report not approved after two reviews.\n"
            f"Last reviewer feedback: {state['feedback']}"
        )
    }

# Decide the next step after review
def review_router(state: AgentState):
    if state["review_status"] == "approved":
        return "approved"

    if state["review_count"] >= 2:
        return "failed"

    return "revise"

# Build the workflow
builder = StateGraph(AgentState)

builder.add_node("researcher", researcher)
builder.add_node("fact_checker", fact_checker)
builder.add_node("report_writer", report_writer)
builder.add_node("review_failed", review_failed)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "fact_checker")

builder.add_conditional_edges(
    "fact_checker",
    review_router,
    {
        "approved": "report_writer",
        "revise": "researcher",
        "failed": "review_failed",
    },
)

builder.add_edge("report_writer", END)
builder.add_edge("review_failed", END)

workflow = builder.compile()


if __name__ == "__main__":
    topic = input("\nWhat do you want to research? ")

    if not topic.strip():
        print("Please enter a research topic.")
    else:
        result = workflow.invoke({
            "topic": topic,
            "research": "",
            "fact_check": "",
            "final_report": "",
            "review_status": "",
            "review_count": 0,
            "feedback": "",
        })

        print("\n========== RESULT ==========")
        print(result["final_report"])
