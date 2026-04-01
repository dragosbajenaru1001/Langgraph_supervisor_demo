# LangGraph Supervisor Demo

A multi-agent pipeline built with **LangGraph** where a **supervisor** orchestrates three specialized workers — `researcher`, `coder`, and `writer`.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
4. [Prerequisites](#prerequisites)
5. [Step-by-Step: Building `main.py`](#step-by-step-building-mainpy)
   - [Step 1 — Configure the LLM](#step-1--configure-the-llm)
   - [Step 2 — Define the Routing Schema](#step-2--define-the-routing-schema)
   - [Step 3 — Define the Graph State](#step-3--define-the-graph-state)
   - [Step 4 — Build the Supervisor Node](#step-4--build-the-supervisor-node)
   - [Step 5 — Build the Worker Nodes](#step-5--build-the-worker-nodes)
   - [Step 6 — Assemble the Graph](#step-6--assemble-the-graph)
   - [Step 7 — Run the Pipeline](#step-7--run-the-pipeline)
6. [Prompts Used and Why They Work](#prompts-used-and-why-they-work)

---

## Overview

This project demonstrates how to implement a **supervisor multi-agent system** using LangGraph. A single LLM call (the supervisor) decides which specialized agent acts next, routing work sequentially: `researcher → coder → writer → FINISH`.

Each agent receives the original user request plus the most recent context in the conversation, performs its specific task, then hands control back to the supervisor, which decides the next step.

---

## Architecture

```
                 ┌─────────────────────────────────────────┐
                 │              StateGraph                  │
                 │                                          │
   START ───────►│           [supervisor]                   │
                 │          /     |      \                  │
                 │   [researcher] [coder] [writer]          │
                 │          \     |      /                  │
                 │           └────┴──────┘                  │
                 │                │                         │
                 │              END                         │
                 └─────────────────────────────────────────┘
```

- **`GraphState`** — shared state carrying `messages` (full conversation history) and `acted` (list of agents that have already contributed)
- **`supervisor_node`** — reads `acted`, sends a minimal routing prompt, returns a `Command` pointing to the next node
- **`researcher_node` / `coder_node` / `writer_node`** — each reads the first message (original request) + last two messages (recent context), invokes the LLM, appends the result, and routes back to `supervisor`

---

## Prerequisites

### Install dependencies

```bash
pip install langgraph langchain-core langchain-groq python-dotenv pydantic
```

### Environment variables

Create a `.env` file at the project root:

```env
GROQ_API_KEY=your_key_here
```

Get your API key at [https://console.groq.com/keys](https://console.groq.com/keys).

---

## Step-by-Step: Building `main.py`

### Step 1 — Configure the LLM

**Prompt used to understand this step:**

> "I want to use Groq as my LLM provider inside a LangGraph application. Show me how to initialize a ChatGroq instance with a fixed temperature and load the API key from an environment variable using python-dotenv."

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)
```

**Why `temperature=0`?**
The supervisor must make deterministic routing decisions. Any randomness in the router output could cause it to jump steps or loop indefinitely. Workers can use a slightly higher temperature if creativity is desired, but for routing, determinism is critical.

---

### Step 2 — Define the Routing Schema

**Prompt used to understand this step:**

> "I need the LLM to always return a structured routing decision with a `next` field constrained to exactly the values `researcher`, `coder`, `writer`, or `FINISH`, and a `reason` field. Use Pydantic and LangChain's `with_structured_output`."

```python
from pydantic import BaseModel
from typing import Literal

class RouteDecision(BaseModel):
    next: Literal["researcher", "coder", "writer", "FINISH"]
    reason: str

llm_router = llm.with_structured_output(RouteDecision)
```

**Why structured output?**
Without it the LLM might return `"the researcher should go next"` as free text, which cannot be used as a graph edge. `with_structured_output` forces the model to emit valid JSON matching the Pydantic schema — making the routing machine-readable and safe to pass directly into LangGraph's `Command`.

**Supervisor system prompt:**

```python
SYSTEM_PROMPT = (
    "Supervisor pt echipa: researcher, coder, writer.\n"
    "Routing în ordine: researcher → coder → writer → FINISH.\n"
    "Mesajul tău va indica cine a acționat deja. Alege următorul."
)
```

The prompt is intentionally minimal. It tells the supervisor:
1. What team members exist
2. What the expected ordering is
3. That the user message will provide the current progress

---

### Step 3 — Define the Graph State

**Prompt used to understand this step:**

> "Using LangGraph's TypedDict pattern, define a shared state that holds the full message history as a list of LangChain BaseMessages, plus a list of strings tracking which agents have already acted."

```python
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    messages: List[BaseMessage]
    acted: List[str]
```

**Why `acted`?**
The supervisor only receives a short routing message, not the entire conversation. The `acted` list gives it enough context to know where in the sequence it is, without sending unnecessary tokens — each node only receives what it needs.

---

### Step 4 — Build the Supervisor Node

**Prompt used to understand this step:**

> "Write a LangGraph node function called `supervisor_node` that reads the `acted` list from state, sends a short routing message to the LLM router, and returns a `Command` that either points to the next worker or to END if the decision is FINISH."

```python
from langgraph.graph import END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage

def supervisor_node(state: GraphState) -> Command[Literal["researcher", "coder", "writer", END]]:
    acted = state.get("acted", [])
    routing_msg = HumanMessage(content=f"Au acționat deja: {acted or 'nimeni'}. Cine urmează?")
    decision = llm_router.invoke([SystemMessage(content=SYSTEM_PROMPT), routing_msg])

    goto = END if decision.next == "FINISH" else decision.next
    return Command(goto=goto, update={"acted": acted})
```

**Key design decisions:**
- The routing message is only ~10 tokens — the supervisor does not need to re-read the full conversation to decide routing
- `Command(goto=..., update=...)` is LangGraph's way of combining a state update with a navigation decision in a single return value
- The supervisor does **not** modify `acted` — workers are responsible for appending their own name

---

### Step 5 — Build the Worker Nodes

**Prompt used to understand this step:**

> "Write a factory function `make_worker(name, description)` that returns a LangGraph node. The node should: take only the first message (original request) and the last two messages (recent context) to minimize token usage, invoke the LLM with a system prompt identifying the worker's role, append the result to messages, add the worker's name to `acted`, and route back to the supervisor."

```python
def make_worker(name: str, description: str):
    def worker_node(state: GraphState) -> Command[Literal["supervisor"]]:
        # Each worker sees only the original request + recent context
        context = state["messages"][:1] + state["messages"][-2:]
        result = llm.invoke(
            [SystemMessage(content=f"Ești {name}. {description}")]
            + context
        )
        acted = state.get("acted", []) + [name]
        return Command(
            goto="supervisor",
            update={"messages": state["messages"] + [result], "acted": acted},
        )

    worker_node.__name__ = f"{name}_node"
    return worker_node
```

**Worker prompts:**

```python
researcher_node = make_worker(
    "researcher",
    "Caută și sintetizează informații. Răspunde concis cu datele găsite.",
)
coder_node = make_worker(
    "coder",
    "Scrie cod Python curat și funcțional. Include comentarii.",
)
writer_node = make_worker(
    "writer",
    "Redactează texte clare și coerente pe baza informațiilor primite.",
)
```

**Why `messages[:1] + messages[-2:]`?**
This is an explicit token-reduction strategy. The worker always sees:
- `messages[0]` — the original user request (the task definition)
- `messages[-2:]` — the two most recent messages (the latest context from previous agents)

This avoids sending the entire history to every worker, which would multiply token costs by 3x at minimum.

---

### Step 6 — Assemble the Graph

**Prompt used to understand this step:**

> "Using LangGraph's StateGraph builder, add the supervisor and three worker nodes, connect START to the supervisor, compile the graph, and return it."

```python
from langgraph.graph import StateGraph, START

def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "supervisor")

    return builder.compile()
```

**Why no explicit edges between workers and supervisor?**
LangGraph's `Command` object returned by each node carries the routing decision internally. When a node returns `Command(goto="supervisor", ...)`, LangGraph uses that as the edge. This means the graph topology is partially defined at runtime, which is more flexible than static `add_edge` calls for all transitions.

---

### Step 7 — Run the Pipeline

**Prompt used to understand this step:**

> "Invoke the compiled graph with an initial state containing a HumanMessage asking the agents to research LangGraph and write a minimal Python StateGraph example. Print the last message in the result."

```python
if __name__ == "__main__":
    graph = build_graph()

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Cercetează ce este LangGraph, "
                        "apoi scrie un exemplu minimal de StateGraph în Python."
                    )
                )
            ],
            "acted": [],
        }
    )

    print("\n=== Au acționat ===")
    print(result["acted"])
    print("\n=== Răspuns final ===")
    print(result["messages"][-1].content)
```

The initial state seeds the pipeline:
- `messages` — contains only the user's original request
- `acted` — empty list, signaling to the supervisor that no one has acted yet

---

## Prompts Used and Why They Work

### Design principles applied

| Principle | Where applied |
|---|---|
| **Single responsibility** | Each worker system prompt has exactly one job description |
| **Minimal context per step** | Workers receive `messages[:1] + messages[-2:]`, not the full history |
| **Structured handoff** | The supervisor emits a `RouteDecision` Pydantic object, not free text |
| **Explicit progress tracking** | The `acted` list is passed as a single line to the supervisor, not inferred from the full conversation |
| **Deterministic routing** | `temperature=0` on the router ensures the sequence always progresses forward |

### Prompts summary table

| Node | System prompt intent | Key constraint |
|---|---|---|
| Supervisor | Route to next unvisited worker | Must output valid `RouteDecision` enum |
| Researcher | Find and synthesize information | Respond concisely with found data |
| Coder | Write clean, functional Python | Include comments |
| Writer | Draft clear, coherent text | Based on information received from prior steps |

### Prompt chaining

**Prompt chaining** was the technique used to build this README itself. Rather than asking for the entire implementation in one shot, each step was prompted individually — and the output of each prompt informed the next one.

The chain started broad and narrowed progressively:

1. **Step 1 prompt** established the foundation — initialize the LLM with Groq and load credentials. No prior context needed.
2. **Step 2 prompt** built on that — now that the LLM exists, constrain its output to a typed routing schema using `with_structured_output`.
3. **Step 3 prompt** built on step 2 — now that routing decisions exist, define the state that will carry those decisions through the graph.
4. **Step 4 prompt** built on steps 2 and 3 — now that the schema and state exist, write the supervisor that reads state and returns a `Command`.
5. **Step 5 prompt** built on step 4 — now that the supervisor is defined, write the workers that the supervisor routes to, and have them return back to it.
6. **Step 6 prompt** built on steps 4 and 5 — now that all nodes exist, assemble them into a compiled graph.
7. **Step 7 prompt** built on step 6 — now that the graph is compiled, invoke it with an initial state and print the result.

Each prompt assumed the previous step was already working. This kept each prompt short and specific, and produced code that fit together cleanly because every step was written with the previous step's output as its input context.

---
