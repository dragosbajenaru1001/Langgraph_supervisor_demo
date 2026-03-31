import os
from typing import Literal, TypedDict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pydantic import BaseModel

load_dotenv()

# ──────────────────────────────────────────────
# 1. Modelul LLM
# ──────────────────────────────────────────────
# GROQ_API_KEY poate fi luată accesând link-ul https://console.groq.com/keys
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

# ──────────────────────────────────────────────
# 2. Schema de routing (structured output)
# ──────────────────────────────────────────────
class RouteDecision(BaseModel):
    next: Literal["researcher", "coder", "writer", "FINISH"]
    reason: str


llm_router = llm.with_structured_output(RouteDecision)

MEMBERS = ["researcher", "coder", "writer"]
SYSTEM_PROMPT = (
    "Supervisor pt echipa: researcher, coder, writer.\n"
    "Routing în ordine: researcher → coder → writer → FINISH.\n"
    "Mesajul tău va indica cine a acționat deja. Alege următorul."
)


# ──────────────────────────────────────────────
# 3. Starea grafului
# ──────────────────────────────────────────────
class GraphState(TypedDict):
    messages: List[BaseMessage]
    acted: List[str]  # agenți care au contribuit deja


# ──────────────────────────────────────────────
# 4. Nodul supervisor
# ──────────────────────────────────────────────
def supervisor_node(
    state: GraphState,
) -> Command[Literal["researcher", "coder", "writer", END]]:
    acted = state.get("acted", [])
    # Trimite doar un mesaj scurt în loc de tot istoricul
    routing_msg = HumanMessage(content=f"Au acționat deja: {acted or 'nimeni'}. Cine urmează?")
    decision = llm_router.invoke([SystemMessage(content=SYSTEM_PROMPT), routing_msg])

    goto = END if decision.next == "FINISH" else decision.next
    return Command(goto=goto, update={"acted": acted})


# ──────────────────────────────────────────────
# 5. Noduri worker
# ──────────────────────────────────────────────
def make_worker(name: str, description: str):
    def worker_node(state: GraphState) -> Command[Literal["supervisor"]]:
        # Workerul primește doar primul mesaj (cererea originală) + ultimul context relevant
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


# ──────────────────────────────────────────────
# 6. Construirea grafului
# ──────────────────────────────────────────────
def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "supervisor")

    return builder.compile()


# ──────────────────────────────────────────────
# 7. Rulare
# ──────────────────────────────────────────────
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

    print("\n=== Răspuns final ===")
    print(result["messages"][-1].content)
