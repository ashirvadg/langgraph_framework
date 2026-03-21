# =============================================================================
# Hospital Medicine Inventory Manager -- A LangGraph Learning Project
# =============================================================================
#
# This project teaches you how LangGraph works by building a real-world
# hospital medicine inventory system.
#
# WHAT THIS DOES:
# A pharmacist enters a medicine name. The system runs 3 checks in PARALLEL
# (stock level, expiry dates, supplier availability), then makes a DECISION
# using a conditional edge -- either reorder the medicine or generate an
# "all good" report.
#
# LANGGRAPH CONCEPTS COVERED:
# 1. State Management (Pydantic) -- how data flows through the graph
# 2. Nodes -- each function that does one job
# 3. Parallel Execution -- multiple nodes run at the same time
# 4. Fan-in -- waiting for all parallel nodes to finish
# 5. Conditional Edges -- routing to different paths based on results
# 6. Graph Compilation -- turning the graph definition into a runnable app
#
# =============================================================================

import operator
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Load the OpenAI API key from .env file
load_dotenv()


# =============================================================================
# STEP 1: DEFINE THE STATE (Pydantic Model)
# =============================================================================
# Think of State as the "patient chart" that travels with the medicine request.
# Every node in the graph can READ from it and WRITE to it.
# Each field tracks one piece of information about the inventory check.
#
# WHY PYDANTIC?
# - Pydantic gives us type safety (catches wrong data types early)
# - Default values mean the state is valid even before all nodes run
# - LangGraph merges each node's output dict into this state automatically

class InventoryState(BaseModel):
    """The state that flows through our inventory check graph."""

    # -- Input: What medicine are we checking?
    medicine_name: str = ""

    # -- Results from parallel checks (each node fills its own field)
    stock_status: str = ""          # Result from check_stock_level node
    expiry_status: str = ""         # Result from check_expiry_dates node
    supplier_status: str = ""       # Result from check_supplier_availability node

    # -- Decision fields (filled by inventory_decision node)
    needs_reorder: bool = False     # Does this medicine need reordering?
    decision_reason: str = ""       # Why did we make this decision?

    # -- Final output
    final_report: str = ""          # The final report or reorder request

    # -- Message log: accumulates messages from all nodes
    # The Annotated[list, operator.add] tells LangGraph to APPEND new messages
    # instead of overwriting. This is important for parallel nodes -- each node
    # adds its own messages and they all get combined.
    messages: Annotated[list, operator.add] = []


# =============================================================================
# STEP 2: INITIALIZE THE LLM
# =============================================================================
# We use gpt-4o-mini because it is cheap, fast, and good enough for this demo.
# The temperature=0.7 adds slight creativity to responses.

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# =============================================================================
# STEP 3: DEFINE NODE FUNCTIONS
# =============================================================================
# Each node is a simple Python function that:
#   - Takes the current state as input
#   - Does ONE job (calls OpenAI with a specific prompt)
#   - Returns a dict with ONLY the fields it wants to update
#
# LangGraph automatically merges the returned dict into the state.
# A node does NOT need to return the entire state -- just the changed parts.


def receive_request(state: InventoryState) -> dict:
    """
    NODE 1: Receive and validate the medicine request.

    This is the entry point. It takes the medicine name from the state
    and uses the LLM to generate a structured medicine profile.

    GRAPH POSITION: START --> receive_request --> [3 parallel nodes]
    """
    print(f"\n--- Receiving request for: {state.medicine_name} ---")

    response = llm.invoke(
        f"You are a hospital pharmacy system. A pharmacist wants to check "
        f"inventory for: {state.medicine_name}. "
        f"Acknowledge the request and provide a brief description of this medicine "
        f"(what it is used for, typical dosage). Keep it to 2-3 sentences."
    )

    print(f"Request acknowledged: {response.content[:80]}...")

    return {
        "messages": [f"[receive_request] Checking inventory for {state.medicine_name}"]
    }


def check_stock_level(state: InventoryState) -> dict:
    """
    NODE 2a: Check how many units are left in the warehouse.

    RUNS IN PARALLEL with check_expiry_dates and check_supplier_availability.

    This is one of 3 parallel nodes. LangGraph runs all 3 at the same time
    because we defined edges from receive_request to all 3 nodes.
    Each parallel node writes to its OWN field (stock_status, expiry_status,
    supplier_status) so there are no conflicts.
    """
    print(f"\n--- [Parallel] Checking stock level for: {state.medicine_name} ---")

    response = llm.invoke(
        f"You are a hospital warehouse management system. "
        f"Check the stock level for: {state.medicine_name}. "
        f"Simulate a realistic stock report. Include: "
        f"- Current quantity in units "
        f"- Minimum required stock level "
        f"- Whether stock is LOW, MODERATE, or SUFFICIENT "
        f"Reply in plain text, keep it short (3-4 lines)."
    )

    print(f"Stock check done: {response.content[:80]}...")

    # Return ONLY the fields this node is responsible for
    return {
        "stock_status": response.content,
        "messages": [f"[check_stock_level] Completed stock check"]
    }


def check_expiry_dates(state: InventoryState) -> dict:
    """
    NODE 2b: Check if any batches are expiring soon.

    RUNS IN PARALLEL with check_stock_level and check_supplier_availability.
    """
    print(f"\n--- [Parallel] Checking expiry dates for: {state.medicine_name} ---")

    response = llm.invoke(
        f"You are a hospital pharmacy expiry tracking system. "
        f"Check expiry dates for: {state.medicine_name}. "
        f"Simulate a realistic expiry report. Include: "
        f"- Number of batches currently in stock "
        f"- Earliest expiry date "
        f"- Whether any batch expires within 30 days (YES/NO) "
        f"Reply in plain text, keep it short (3-4 lines)."
    )

    print(f"Expiry check done: {response.content[:80]}...")

    return {
        "expiry_status": response.content,
        "messages": [f"[check_expiry_dates] Completed expiry check"]
    }


def check_supplier_availability(state: InventoryState) -> dict:
    """
    NODE 2c: Check if suppliers can fulfill a reorder if needed.

    RUNS IN PARALLEL with check_stock_level and check_expiry_dates.
    """
    print(f"\n--- [Parallel] Checking supplier availability for: {state.medicine_name} ---")

    response = llm.invoke(
        f"You are a hospital supplier management system. "
        f"Check supplier availability for: {state.medicine_name}. "
        f"Simulate a realistic supplier report. Include: "
        f"- Primary supplier name and estimated delivery time "
        f"- Backup supplier name and estimated delivery time "
        f"- Current supply chain status (NORMAL / DELAYED / DISRUPTED) "
        f"Reply in plain text, keep it short (3-4 lines)."
    )

    print(f"Supplier check done: {response.content[:80]}...")

    return {
        "supplier_status": response.content,
        "messages": [f"[check_supplier_availability] Completed supplier check"]
    }


def inventory_decision(state: InventoryState) -> dict:
    """
    NODE 3: Make a decision based on ALL three parallel checks.

    This node runs AFTER all 3 parallel nodes finish (fan-in).
    It reads stock_status, expiry_status, and supplier_status,
    then decides: does this medicine need reordering?

    GRAPH POSITION: [3 parallel nodes] --> inventory_decision --> conditional edge
    """
    print(f"\n--- Making inventory decision for: {state.medicine_name} ---")

    response = llm.invoke(
        f"You are a hospital inventory decision system. Based on these three reports, "
        f"decide if the medicine needs reordering.\n\n"
        f"Medicine: {state.medicine_name}\n\n"
        f"STOCK REPORT:\n{state.stock_status}\n\n"
        f"EXPIRY REPORT:\n{state.expiry_status}\n\n"
        f"SUPPLIER REPORT:\n{state.supplier_status}\n\n"
        f"Reply STRICTLY in this JSON format (no other text):\n"
        f'{{"needs_reorder": true/false, "reason": "one sentence explanation"}}'
    )

    # Parse the LLM's JSON response
    try:
        result = json.loads(response.content)
        needs_reorder = result["needs_reorder"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        # If LLM response is not valid JSON, default to reorder as a safety measure
        needs_reorder = True
        reason = "Could not parse decision, defaulting to reorder for safety."

    print(f"Decision: {'REORDER NEEDED' if needs_reorder else 'STOCK OK'} -- {reason}")

    return {
        "needs_reorder": needs_reorder,
        "decision_reason": reason,
        "messages": [f"[inventory_decision] Decision: reorder={needs_reorder}"]
    }


def place_reorder(state: InventoryState) -> dict:
    """
    NODE 4a: Place a reorder request with the supplier.

    This node runs ONLY if the conditional edge routes here
    (when needs_reorder is True).

    GRAPH POSITION: inventory_decision --[needs_reorder=True]--> place_reorder --> END
    """
    print(f"\n--- Placing reorder for: {state.medicine_name} ---")

    response = llm.invoke(
        f"You are a hospital procurement system. Generate a reorder request for:\n"
        f"Medicine: {state.medicine_name}\n"
        f"Reason: {state.decision_reason}\n"
        f"Supplier Info: {state.supplier_status}\n\n"
        f"Create a brief reorder form with: quantity to order, preferred supplier, "
        f"urgency level (URGENT/NORMAL), and expected delivery date. "
        f"Keep it to 4-5 lines."
    )

    print(f"Reorder placed: {response.content[:80]}...")

    return {
        "final_report": f"REORDER REQUEST\n{'='*40}\n{response.content}",
        "messages": [f"[place_reorder] Reorder request generated"]
    }


def generate_report(state: InventoryState) -> dict:
    """
    NODE 4b: Generate an "all clear" inventory report.

    This node runs ONLY if the conditional edge routes here
    (when needs_reorder is False).

    GRAPH POSITION: inventory_decision --[needs_reorder=False]--> generate_report --> END
    """
    print(f"\n--- Generating report for: {state.medicine_name} ---")

    response = llm.invoke(
        f"You are a hospital inventory reporting system. Generate a brief inventory "
        f"status report for:\n"
        f"Medicine: {state.medicine_name}\n"
        f"Stock: {state.stock_status}\n"
        f"Expiry: {state.expiry_status}\n"
        f"Supplier: {state.supplier_status}\n\n"
        f"Summarize that everything is in order. Keep it to 4-5 lines."
    )

    print(f"Report generated: {response.content[:80]}...")

    return {
        "final_report": f"INVENTORY REPORT -- ALL CLEAR\n{'='*40}\n{response.content}",
        "messages": [f"[generate_report] Status report generated"]
    }


# =============================================================================
# STEP 4: DEFINE THE ROUTING FUNCTION (for Conditional Edge)
# =============================================================================
# This function decides which path to take after inventory_decision.
# It looks at the state and returns a string key that maps to a node name.
#
# HOW CONDITIONAL EDGES WORK:
# - LangGraph calls this function with the current state
# - The function returns a string (like "reorder" or "report")
# - LangGraph looks up that string in the mapping we provide
# - It routes to the corresponding node


def route_after_decision(state: InventoryState) -> str:
    """
    Routing function for the conditional edge.

    Returns "reorder" if medicine needs reordering, "report" otherwise.
    These strings are mapped to actual node names in add_conditional_edges().
    """
    if state.needs_reorder:
        return "reorder"
    else:
        return "report"


# =============================================================================
# STEP 5: BUILD THE GRAPH
# =============================================================================
# This is where we wire everything together. Think of it as drawing the
# flowchart -- we add nodes (boxes) and edges (arrows).
#
# THE GRAPH STRUCTURE:
#
#   START
#     |
#   receive_request
#     |
#     +---> check_stock_level -------+
#     |                              |
#     +---> check_expiry_dates ------+---> inventory_decision
#     |                              |         |
#     +---> check_supplier_avail ----+    (conditional)
#                                        /          \
#                                 reorder?         report?
#                                   |                |
#                             place_reorder    generate_report
#                                   |                |
#                                  END              END

# Create the graph with our Pydantic state model
graph = StateGraph(InventoryState)

# Add all nodes to the graph (like placing boxes on the flowchart)
graph.add_node("receive_request", receive_request)
graph.add_node("check_stock_level", check_stock_level)
graph.add_node("check_expiry_dates", check_expiry_dates)
graph.add_node("check_supplier_availability", check_supplier_availability)
graph.add_node("inventory_decision", inventory_decision)
graph.add_node("place_reorder", place_reorder)
graph.add_node("generate_report", generate_report)

# --- EDGES (the arrows connecting the boxes) ---

# START --> receive_request (entry point of the graph)
graph.add_edge(START, "receive_request")

# receive_request --> 3 parallel nodes (FAN-OUT)
# When you add multiple edges FROM the same node, LangGraph runs them in PARALLEL.
# This is like 3 departments checking different things at the same time.
graph.add_edge("receive_request", "check_stock_level")
graph.add_edge("receive_request", "check_expiry_dates")
graph.add_edge("receive_request", "check_supplier_availability")

# 3 parallel nodes --> inventory_decision (FAN-IN)
# All 3 must complete before inventory_decision runs.
# LangGraph automatically waits for all incoming edges.
graph.add_edge("check_stock_level", "inventory_decision")
graph.add_edge("check_expiry_dates", "inventory_decision")
graph.add_edge("check_supplier_availability", "inventory_decision")

# inventory_decision --> CONDITIONAL EDGE
# This is the decision point. Based on the state, route to one of two paths.
# The route_after_decision function returns "reorder" or "report".
# The dict maps those strings to actual node names.
graph.add_conditional_edges(
    "inventory_decision",           # Source node
    route_after_decision,           # Routing function
    {                               # Mapping: function return value --> node name
        "reorder": "place_reorder",
        "report": "generate_report"
    }
)

# Terminal edges --> END
graph.add_edge("place_reorder", END)
graph.add_edge("generate_report", END)


# =============================================================================
# STEP 6: COMPILE AND RUN
# =============================================================================
# Compiling turns the graph definition into a runnable application.
# After compilation, you call app.invoke() with the initial state.

# Compile the graph into a runnable app
app = graph.compile()


def run_inventory_check(medicine_name: str):
    """
    Run the inventory check for a given medicine.

    This is the main entry point. It:
    1. Creates the initial state with the medicine name
    2. Invokes the compiled graph
    3. Prints the final report
    """
    print("=" * 60)
    print(f"  HOSPITAL MEDICINE INVENTORY CHECK")
    print(f"  Medicine: {medicine_name}")
    print("=" * 60)

    # Invoke the graph with initial state
    # We only need to provide the fields we want to set initially.
    # All other fields use their default values from the Pydantic model.
    result = app.invoke({
        "medicine_name": medicine_name,
        "messages": []
    })

    # Print the final result
    print("\n" + "=" * 60)
    print("  FINAL RESULT")
    print("=" * 60)
    print(f"\n{result['final_report']}")

    print("\n" + "-" * 60)
    print("  MESSAGE LOG (shows the order nodes executed)")
    print("-" * 60)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result


# =============================================================================
# MAIN: Run the inventory check
# =============================================================================
if __name__ == "__main__":
    # Try it with a common medicine
    run_inventory_check("Paracetamol 500mg")
