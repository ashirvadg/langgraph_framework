# =================================================================================================
# Interview Preparation Agent with quick and long planning options -- A LangGraph Learning Project
# =======================================================================================
#
# This project teaches you how LangGraph works by building a Interview Preparation assistant that suggests personalized calming practices.
#
# WHAT THIS DOES:
# A user enters the role they are interviewing for and how prepared they feel
# (e.g. "I have a senior backend engineer interview tomorrow and I feel
# underprepared"). The graph runs three specialist suggestion nodes in parallel:
# technical topics to review, behavioral story prompts, and confidence-building
# habits. A decision node reads all three outputs plus the user's urgency level,
# then routes to either a QUICK PREP path (1-hour focused drill on the top three
# gaps) or a DEEP PREP path (a structured 3-hour study plan with timed blocks for
# each area). The state tracks job_role, urgency_level, technical_suggestion,
# behavioral_suggestion, confidence_suggestion, needs_deep_prep, final_plan, and
# messages.
#
# LANGGRAPH CONCEPTS COVERED:
# 1. State Management (Pydantic) -- user feeling flows through the graph
# 2. Nodes -- each function does one job (technical, behavioral, and confidence-building, etc.)
# 3. Parallel Execution -- 3 suggestion nodes run at the same time
# 4. Fan-in -- waiting for all 3 suggestions before picking the best
# 5. Conditional Edges -- routing to quick vs deep based on severity
# 6. Graph Compilation -- turning the graph definition into a runnable app
#
# GRAPH STRUCTURE:
#
#   START
#     |
#   understand_role
#     |
#     +---> suggest_technical_topics ----+
#     |                                   |
#     +---> suggest_behavioral_stories ---+---> pick_best_prep_path
#     |                                   |            |
#     +---> suggest_confidence_habits ----+       (conditional)
#                                                 /              \
#                                            quick_prep         deep_prep
#                                                |                  |
#                                               END                END
#
# HOW TO RUN:
#   python interview_prep_graph.py
#
# DEPENDENCIES (same as requirements.txt):
#   langgraph, langchain-openai, python-dotenv, pydantic
#
# =============================================================================

import sys
import operator
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


class InterviewPrepState(BaseModel):
    understand_role: str = ""
    technical_suggestion: str = ""
    behavioral_suggestion: str = ""
    confidence_suggestion: str = ""
    need_quick_prep: bool = False
    needs_deep_prep: bool = False
    final_plan: str = ""
    messages: Annotated[list, operator.add] = []


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def understand_role(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are an interview preparation assistant. "
        f"A user is preparing for an interview for the role of: '{state.understand_role}'. "
        f"Provide a brief overview of the key skills and knowledge areas they should focus on for this role. "
        f"Keep it under 10 sentences."
    )
    return {
        "messages": [f"[understand_role] {response.content}"]
    }


def suggest_technical_topics(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a technical interview preparation specialist. "
        f"The user is preparing for an interview for the role of: '{state.understand_role}'. "
        f"Suggest THREE specific technical topics they should focus on for this role. "
        f"Include a brief description of each topic and why it's important. "
    )
    return {
        "technical_suggestion": response.content,
        "messages": [f"[suggest_technical_topics] {response.content}"]
    }


def suggest_behavioral_stories(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a behavioral interview preparation specialist. "
        f"The user is preparing for an interview for the role of: '{state.understand_role}'. "
        f"Suggest THREE specific behavioral stories they should prepare for this role. "
        f"Include a brief description of each story and the key skills it demonstrates. "
    )
    return {
        "behavioral_suggestion": response.content,
        "messages": [f"[suggest_behavioral_stories] {response.content}"]
    }


def suggest_confidence_habits(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a confidence-building coach. "
        f"The user is preparing for an interview for the role of: '{state.understand_role}'. "
        f"Suggest THREE specific habits or practices that would help them build confidence for this interview. "
        f"Include a brief description of each habit and how it would help. "
   )
    return {
        "confidence_suggestion": response.content,
        "messages": [f"[suggest_confidence_habits] {response.content}"]
    }
        

def pick_interview_practice(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are an interview preparation assistant. The user is preparing for an interview for the role of: '{state.understand_role}'.\n\n"
        f"Here are three suggestions from specialists:\n\n"
        f"TECHNICAL:\n{state.technical_suggestion}\n\n"
        f"BEHAVIORAL:\n{state.behavioral_suggestion}\n\n"
        f"CONFIDENCE:\n{state.confidence_suggestion}\n\n"
        f"Decide: does this person need a QUICK practice (under 1 hour, for quick preperaton based on the urgency) "
        f"or a DEEP practice (2 hours plan, for high intenese interview preparation)?\n\n"
        f"Reply STRICTLY in this JSON format (no other text):\n"
        f'{{"needs_deep_prep": true/false, "reason": "one sentence explanation"}}'
    )
    try:
        result = json.loads(response.content)
        needs_deep = result["needs_deep_prep"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        needs_deep = False
        reason = "Could not parse decision, defaulting to quick practice."

    return {
        "needs_deep_prep": needs_deep,
        "practice_reason": reason,
        "messages": [f"[pick_interview_practice] deep_session={needs_deep}"]
    }


def quick_interview_practice(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a friendly interview preparation coach. The user is preparing for an interview for the role of: '{state.understand_role}'.\n\n"
        f"Based on these specialist suggestions, create a SHORT practice (under 1 hour) "
        f"that combines the best elements:\n\n"
        f"TECHNICAL: {state.technical_suggestion}\n"
        f"BEHAVIORAL: {state.behavioral_suggestion}\n"
        f"CONFIDENCE: {state.confidence_suggestion}\n\n"
        f"Format it as a simple numbered list of steps. "
        f"Keep it warm, encouraging, and easy to follow. End with a kind closing line."
    )
    return {
        "final_plan": f"QUICK INTERVIEW PRACTICE (under 1 hour)\n{'='*45}\n{response.content}",
        "messages": [f"[quick_interview_practice] Generated quick practice"]
    }


def deep_interview_practice(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a compassionate interview preparation coach. The user is preparing for an interview for the role of: '{state.understand_role}'.\n\n"
        f"Based on these specialist suggestions, create a DEEPER session (3 hours) "
        f"that thoughtfully combines all three approaches:\n\n"
        f"TECHNICAL: {state.technical_suggestion}\n"
        f"BEHAVIORAL: {state.behavioral_suggestion}\n"
        f"CONFIDENCE: {state.confidence_suggestion}\n\n"
        f"Structure it in 3 phases: Technical (review), Behavioral (story telling), Confidence (building). "
        f"Give clear step-by-step instructions for each phase with timing. "
        f"Keep it warm and supportive. End with a kind closing message."
    )
    return {
        "final_plan": f"DEEP INTERVIEW PRACTICE (3 hours)\n{'='*45}\n{response.content}",
        "messages": [f"[deep_practice] Generated deep session"]
    }


def route_after_decision(state: InterviewPrepState) -> str:
    if state.needs_deep_prep:
        return "deep"
    else:
        return "quick"


graph = StateGraph(InterviewPrepState)

graph.add_node("understand_role", understand_role)
graph.add_node("suggest_technical_topics", suggest_technical_topics)
graph.add_node("suggest_behavioral_stories", suggest_behavioral_stories)
graph.add_node("suggest_confidence_habits", suggest_confidence_habits)
graph.add_node("pick_interview_practice", pick_interview_practice)
graph.add_node("quick_interview_practice", quick_interview_practice)
graph.add_node("deep_interview_practice", deep_interview_practice)

graph.add_edge(START, "understand_role")

graph.add_edge("understand_role", "suggest_technical_topics")
graph.add_edge("understand_role", "suggest_behavioral_stories")
graph.add_edge("understand_role", "suggest_confidence_habits")

graph.add_edge("suggest_technical_topics", "pick_interview_practice")
graph.add_edge("suggest_behavioral_stories", "pick_interview_practice")
graph.add_edge("suggest_confidence_habits", "pick_interview_practice")

graph.add_conditional_edges(
    "pick_interview_practice",
    route_after_decision,
    {
        "quick": "quick_interview_practice",
        "deep": "deep_interview_practice",
    }
)

graph.add_edge("quick_interview_practice", END)
graph.add_edge("deep_interview_practice", END)

app = graph.compile()


def run_interview_check(interviewrole: str):
    print("=" * 55)
    print("  INTERVIEW PRACTICE SUGGESTER")
    print(f"  You said: \"{interviewrole}\"")
    print("=" * 55)

    result = app.invoke({
        "understand_role": interviewrole,
        "messages": [],
    })

    print("\n" + "=" * 55)
    print("  YOUR PERSONALIZED PRACTICE")
    print("=" * 55)
    print(f"\n{result['final_plan']}")

    print("\n" + "-" * 55)
    print("  MESSAGE LOG")
    print("-" * 55)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  INTERVIEW PRACTICE SUGGESTER")
    print("=" * 55)
    print("\n  Tell me about the interview role you're preparing and I'll suggest a")
    print("  personalized practice session just for you.")
    print("  Type 'quit' to exit.\n")

    while True:
        interview_role = input("  What role are you preparing for and how much time you have ? > ").strip()

        if interview_role.lower() in ("quit", "exit", "q"):
            print("\n  Take care of yourself. Goodbye!\n")
            break

        if not interview_role:
            continue

        run_interview_check(interview_role)
        print("\n")
