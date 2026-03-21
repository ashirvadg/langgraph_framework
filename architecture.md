# Mental Wellness Practice Suggester -- Architecture

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│              "I feel stressed and overwhelmed"                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     understand_mood                              │
│                                                                  │
│  • Acknowledges the user's feeling warmly                        │
│  • Classifies severity: MILD / MODERATE / HIGH                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   suggest    │ │   suggest    │ │   suggest    │
│  breathing   │ │ mindfulness  │ │  movement    │
│              │ │              │ │              │
│ 4-7-8, box  │ │ 5-4-3-2-1,  │ │ Child's pose │
│ breathing,   │ │ body scan,  │ │ neck rolls,  │
│ etc.         │ │ etc.        │ │ etc.         │
└──────┬───────┘ └──────┬──────┘ └──────┬───────┘
       │                │               │
       │      PARALLEL EXECUTION        │
       │     (LangGraph Fan-Out)        │
       │                │               │
       └────────────────┼───────────────┘
                        │
                        ▼  (Fan-In: waits for all 3)
┌──────────────────────────────────────────────────────────────────┐
│                    pick_best_practice                             │
│                                                                  │
│  • Reads all 3 suggestions                                       │
│  • Evaluates user severity                                       │
│  • Decides: QUICK (≤5 min) or DEEP (10-15 min)                  │
│  • Returns JSON: {needs_deep_session, reason}                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                    CONDITIONAL EDGE
                    (route_after_decision)
                           │
              ┌────────────┴────────────┐
              │                         │
     needs_deep = false        needs_deep = true
              │                         │
              ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│   quick_practice    │   │   deep_practice     │
│                     │   │                     │
│ • Under 5 minutes   │   │ • 10-15 minutes     │
│ • Best single       │   │ • 3 phases:         │
│   technique from    │   │   1. Settle (breath) │
│   the 3 suggestions │   │   2. Ground (mind)   │
│ • Numbered steps    │   │   3. Release (move)  │
│                     │   │ • Full guided session │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
│                                                                  │
│  • Personalized wellness practice with step-by-step instructions │
│  • Message log showing node execution order                      │
└──────────────────────────────────────────────────────────────────┘
```

## LangGraph Concepts Map

```
CONCEPT                 WHERE IN CODE
─────────────────────── ──────────────────────────────────────
State (Pydantic)        WellnessState class
Nodes                   understand_mood, suggest_*, pick_*, quick/deep
Parallel Execution      understand_mood → 3 suggest nodes (fan-out)
Fan-In                  3 suggest nodes → pick_best_practice
Conditional Edge        pick_best_practice → quick OR deep
Routing Function        route_after_decision()
Graph Compilation       graph.compile() → app
Invocation              app.invoke({user_feeling: "..."})
```

## State Flow

```
WellnessState
├── user_feeling ─────────────── Set by user at start
├── breathing_suggestion ─────── Written by suggest_breathing
├── mindfulness_suggestion ───── Written by suggest_mindfulness
├── movement_suggestion ──────── Written by suggest_movement
├── needs_deep_session ───────── Written by pick_best_practice
├── practice_reason ──────────── Written by pick_best_practice
├── final_suggestion ─────────── Written by quick_practice OR deep_practice
└── messages ─────────────────── Appended by ALL nodes (operator.add)
```

## Tech Stack

```
LangGraph ─────── Graph orchestration (nodes, edges, parallel, conditional)
LangChain ─────── OpenAI LLM wrapper (ChatOpenAI)
OpenAI ────────── gpt-4o-mini (cheap, fast, good enough)
Pydantic ──────── State validation and type safety
python-dotenv ─── Load API key from .env
```
