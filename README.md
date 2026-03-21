# Hospital Medicine Inventory Manager -- Learn LangGraph Step by Step

A beginner-friendly project to learn the LangGraph framework by building a hospital medicine inventory check system.

---

## Understanding LangGraph Through the Tiffin Box Analogy

Before diving into code, let us understand how LangGraph works using the **Tiffin Box** (Indian lunch box) analogy.

A tiffin box has multiple compartments. Imagine a kitchen where 3 cooks prepare different compartments at the same time (parallel), a supervisor checks the quality (conditional), and either packs it for delivery or sends it back for fixing.

LangGraph works the same way -- but instead of food, we pass **data (state)** through **nodes (functions)** connected by **edges (arrows)**.

```
    THE TIFFIN BOX ANALOGY FOR LANGGRAPH
    =====================================

    Think of LangGraph as a kitchen assembly line:

    +----------------------------------------------------------+
    |                                                          |
    |   TIFFIN BOX = STATE (Pydantic Model)                   |
    |   The box travels through the kitchen.                   |
    |   Each station fills one compartment.                    |
    |                                                          |
    |   +-------------+  +-------------+  +-------------+     |
    |   | Compartment |  | Compartment |  | Compartment |     |
    |   |   (rice)    |  |   (curry)   |  |   (salad)   |     |
    |   +-------------+  +-------------+  +-------------+     |
    |                                                          |
    +----------------------------------------------------------+

    NODES = Kitchen Stations (each does one job)
    EDGES = Conveyor belts connecting stations
    PARALLEL NODES = Multiple cooks working at the same time
    CONDITIONAL EDGE = Supervisor deciding: "Pack it or fix it?"

    How data flows:

    [Customer Order]                         <-- START node
          |
          v
    +-----------+
    | Take Order|                            <-- Node 1
    +-----------+
          |
     _____|_____________________
    |            |              |
    v            v              v
  +------+   +-------+   +-------+
  | Rice |   | Curry |   | Salad |           <-- Parallel Nodes
  +------+   +-------+   +-------+              (Fan-Out)
    |            |              |
    |____________|______________|
                 |
                 v                               (Fan-In)
         +--------------+
         | Quality Check|                    <-- Decision Node
         +--------------+
                 |
           ______|______
          |             |
     [PASS]         [FAIL]                   <-- Conditional Edge
          |             |
          v             v
    +----------+   +----------+
    |  Pack    |   |   Fix    |--+
    |  Tiffin  |   |   Meal   |  |
    +----------+   +----------+  |
          |             |        |
          v             +--------+           <-- Loop (retry)
        [END]


    KEY LANGGRAPH CONCEPTS:
    -----------------------
    1. STATE    = The tiffin box itself (holds all data)
    2. NODE     = A kitchen station (a function that does one thing)
    3. EDGE     = A conveyor belt (connects one node to the next)
    4. PARALLEL = Multiple stations working at the same time
    5. FAN-IN   = Waiting for all parallel stations to finish
    6. CONDITIONAL EDGE = A supervisor deciding the next step
```

---

## Our Project: Hospital Medicine Inventory Check

Now we apply the same pattern to a real use case. A pharmacist enters a medicine name, and the system runs parallel checks, then makes a decision.

```
    HOSPITAL INVENTORY CHECK -- GRAPH ARCHITECTURE
    ================================================

              +------------------+
              |      START       |
              +--------+---------+
                       |
              +--------v---------+
              | receive_request   |     Pharmacist enters medicine name.
              +--------+---------+     System acknowledges the request.
                       |
          _____________|_______________
         |             |               |
         v             v               v
  +-----------+  +-----------+  +----------------+
  | check     |  | check     |  | check          |    3 PARALLEL NODES
  | stock     |  | expiry    |  | supplier       |    (Fan-Out)
  | level     |  | dates     |  | availability   |
  +-----------+  +-----------+  +----------------+    Each node writes to
         |             |               |              its own state field.
         |_____________|_______________|              No conflicts.
                       |
              +--------v---------+
              | inventory        |                    FAN-IN:
              | decision         |                    Waits for all 3 checks.
              +--------+---------+                    Reads all results.
                       |                              Decides: reorder or not?
                 ______|______
                |             |
          [REORDER]       [ALL OK]                    CONDITIONAL EDGE:
                |             |                       route_after_decision()
                v             v                       returns "reorder" or "report"
         +------------+ +---------------+
         | place      | | generate      |
         | reorder    | | report        |
         +-----+------+ +-------+------+
               |                 |
               v                 v
             [END]             [END]


    STATE FIELDS (Pydantic Model):
    ================================
    medicine_name      --> Input: what medicine to check
    stock_status       --> Filled by: check_stock_level
    expiry_status      --> Filled by: check_expiry_dates
    supplier_status    --> Filled by: check_supplier_availability
    needs_reorder      --> Filled by: inventory_decision
    decision_reason    --> Filled by: inventory_decision
    final_report       --> Filled by: place_reorder OR generate_report
    messages           --> Accumulated by ALL nodes (uses operator.add)
```

---

## How LangGraph State Works

```
    HOW STATE FLOWS THROUGH THE GRAPH
    ===================================

    Initial State (before graph runs):
    +-----------------------------------+
    | medicine_name: "Paracetamol 500mg"|
    | stock_status: ""                  |
    | expiry_status: ""                 |
    | supplier_status: ""               |
    | needs_reorder: False              |
    | final_report: ""                  |
    | messages: []                      |
    +-----------------------------------+
                    |
                    v
        [receive_request runs]
                    |
                    v
    After receive_request:
    +-----------------------------------+
    | medicine_name: "Paracetamol 500mg"|  <-- unchanged
    | stock_status: ""                  |  <-- not yet filled
    | expiry_status: ""                 |  <-- not yet filled
    | supplier_status: ""               |  <-- not yet filled
    | messages: ["[receive_request]..."]|  <-- appended
    +-----------------------------------+
                    |
        ____________|____________
       |            |            |
       v            v            v
    [3 parallel nodes run, each fills ONE field]
                    |
                    v
    After parallel nodes:
    +-----------------------------------+
    | stock_status: "Current: 150..."   |  <-- filled by check_stock_level
    | expiry_status: "Batch A: 2025..." |  <-- filled by check_expiry_dates
    | supplier_status: "MedCorp: 3..."  |  <-- filled by check_supplier_availability
    | messages: [..., stock, expiry,    |  <-- all 3 appended (operator.add)
    |            supplier messages]     |
    +-----------------------------------+
                    |
                    v
    After inventory_decision:
    +-----------------------------------+
    | needs_reorder: True               |  <-- decision made
    | decision_reason: "Stock is low"   |  <-- reason captured
    +-----------------------------------+
                    |
           [conditional edge]
           needs_reorder = True
                    |
                    v
    After place_reorder:
    +-----------------------------------+
    | final_report: "REORDER REQUEST.."|  <-- final output
    +-----------------------------------+
```

---

## Setup and Run

### Prerequisites
- Python 3.10 or higher
- An OpenAI API key

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/NisargKadam/langgraph_framework.git
cd langgraph_framework

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Run the inventory check
python hospital_inventory_graph.py
```

### Expected Output

```
============================================================
  HOSPITAL MEDICINE INVENTORY CHECK
  Medicine: Paracetamol 500mg
============================================================

--- Receiving request for: Paracetamol 500mg ---
Request acknowledged: Paracetamol 500mg is a commonly used analgesic and...

--- [Parallel] Checking stock level for: Paracetamol 500mg ---
--- [Parallel] Checking expiry dates for: Paracetamol 500mg ---
--- [Parallel] Checking supplier availability for: Paracetamol 500mg ---
Stock check done: Current quantity: 150 units...
Expiry check done: Batch A expires: 2025-08-15...
Supplier check done: Primary supplier: MedCorp...

--- Making inventory decision for: Paracetamol 500mg ---
Decision: STOCK OK -- All stock levels are sufficient...

--- Generating report for: Paracetamol 500mg ---

============================================================
  FINAL RESULT
============================================================

INVENTORY REPORT -- ALL CLEAR
========================================
Paracetamol 500mg inventory is in good standing...

------------------------------------------------------------
  MESSAGE LOG (shows the order nodes executed)
------------------------------------------------------------
  [receive_request] Checking inventory for Paracetamol 500mg
  [check_stock_level] Completed stock check
  [check_expiry_dates] Completed expiry check
  [check_supplier_availability] Completed supplier check
  [inventory_decision] Decision: reorder=False
  [generate_report] Status report generated
```

---

## Code Walkthrough

| Step | What Happens | File Location |
|------|-------------|---------------|
| 1 | Define `InventoryState` with Pydantic | `hospital_inventory_graph.py` line 50 |
| 2 | Initialize OpenAI LLM | `hospital_inventory_graph.py` line 72 |
| 3 | Define 6 node functions | `hospital_inventory_graph.py` lines 88-230 |
| 4 | Define routing function | `hospital_inventory_graph.py` line 244 |
| 5 | Build graph (add nodes + edges) | `hospital_inventory_graph.py` lines 260-310 |
| 6 | Compile and run | `hospital_inventory_graph.py` lines 320-360 |

---

## Key Takeaways for Beginners

1. **State is just a data class** -- Define what data your graph needs using Pydantic fields with defaults.

2. **Nodes are just functions** -- Each function takes state, does one thing, returns a dict of updated fields.

3. **Parallel is automatic** -- Add multiple edges from one node to many, and LangGraph runs them in parallel.

4. **Conditional edges need a routing function** -- Write a function that returns a string key, map keys to node names.

5. **The graph is just a flowchart** -- You define it in code the same way you would draw it on paper.
