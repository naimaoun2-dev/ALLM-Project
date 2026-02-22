"""
Multi-Agent LangGraph: agents with distinct roles, different toolsets,
Pydantic state, MemorySaver checkpointer, and conditional routing.
"""
import json
import os
import logging
import threading
from typing import List, Dict, Optional, Any, Literal
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


_rag_system_local = threading.local()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver



logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured output from the Router agent (parsed/validated by Pydantic). At least one agent uses SOM."""
    route: Literal["db", "internet", "time", "restaurant", "direct", "refuse"] = Field(
        description="Route to take: 'db' = use RAG database, 'internet' = search internet, 'time' = get current time in a city, 'restaurant' = find restaurants near a city, 'direct' = answer from knowledge without tools, 'refuse' = question is about illegal/harmful/unethical content (drugs, violence, etc.) - do not answer"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score that this route is correct (0.0 to 1.0)"
    )
    reason: str = Field(description="Brief reason for this routing decision")
    legality_score: int = Field(
        ge=-5, le=5,
        description="Legality of the question: -5 = very illegal/harmful, 0 = neutral/uncertain, 5 = very legal/benign. Use negative for illegal, harmful, or unethical topics."
    )



class ToolCallRecord(BaseModel):
    """Nested BaseModel for state: record of a single tool invocation."""
    tool_name: str
    arguments: Dict[str, Any]
    status: str = "success"
    source_info: Optional[Dict[str, Any]] = None


class GraphState(BaseModel):
    """State model : multiple field types (str, int, float, bool, List, Dict, nested BaseModel) and Optional fields that become populated during execution."""
    # Message history
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    # Current user question
    current_question: str = ""
    # Router output (populated by router node)
    route: Optional[str] = None
    confidence_score: Optional[float] = None
    route_reason: Optional[str] = None
    # Tool execution (populated by tool agents)
    tool_result: Optional[str] = None
    source_info: Optional[Dict[str, Any]] = None
    source_documents: List[Any] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    tool_name_used: Optional[str] = None
    # Control flow
    need_critic: bool = False
    step_count: int = 0
    # Final answer (populated by synthesizer)
    final_answer: Optional[str] = None
    question_status: Optional[str] = None
    legality_score: Optional[int] = None
    # Allow extra fields for LangGraph compatibility
    model_config = {"arbitrary_types_allowed": True}



def _get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=api_key,
    )


def _get_router_llm_structured():
    """LLM bound to structured output (Pydantic) for Router agent."""
    llm = _get_llm()
    return llm.with_structured_output(RouteDecision)


def router_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """
    Agent 1: Router. Uses Structured Output Mode (Pydantic) to decide route.
    No tool — only classification.
    """

    logger.info("ROUTER NODE ACTIVATED")

    question = state.current_question
    step = state.step_count
    new_step = step + 1

    structured_llm = _get_router_llm_structured()
    prompt = (
        "You are a router. Given the user question, decide which single path to take.\n"
        "Routes:\n"
        "- 'db': Need to search the internal RAG database (countries, capitals, user documents).\n"
        "- 'internet': Need to search the internet (e.g. GeoNames) for country/capital info not in DB.\n"
        "- 'time': User is asking for current time in a city (e.g. 'What time is it in Tokyo?').\n"
        "- 'restaurant': User wants to find restaurants near a city or location.\n"
        "- 'direct': You can answer from your own knowledge with high confidence; no tool needed.\n"
        "- 'refuse': The question is about illegal, harmful, or unethical content. Choose this to decline.\n\n"
        f"User question: {question}\n\n"
        "Respond with your route, confidence (0.0-1.0), brief reason, and legality_score from -5 (very illegal) to 5 (very legal)."
    )

    decision = structured_llm.invoke([HumanMessage(content=prompt)])

    

    route = decision.route
    confidence = decision.confidence
    reason = decision.reason
    legality_score = getattr(decision, "legality_score", 5 if route != "refuse" else -5)

    logger.info(f"Route: {route} ")
    logger.info(f"Confidence: {confidence} ")
    logger.info(f"Reason: {reason} ")
    logger.info(f"Legality Score: {legality_score} ")


    need_critic = confidence < 0.5
    question_status = "unacceptable" if route == "refuse" else "acceptable"

    # Optional: track router as a tool call
    router_call = ToolCallRecord(
        tool_name="router_node",
        arguments={"question": question},
        status="success",
        source_info={"route": route, "confidence": confidence, "reason": reason},
    )

    return {
        "route": route,
        "confidence_score": confidence,
        "route_reason": reason,
        "need_critic": need_critic,
        "step_count": new_step,
        "question_status": question_status,
        "legality_score": legality_score,
        "tool_calls": state.tool_calls + [router_call],
    }


def _get_rag_system_from_context(state: GraphState, config: RunnableConfig | None = None) -> Any:
    """Get RAG system from thread-local (set in run_graph), then config, then state."""
    rag_system = getattr(_rag_system_local, "value", None)
    if rag_system is None and config:
        rag_system = config.get("configurable", {}).get("rag_system")
    if rag_system is None:
        rag_system = getattr(state, "_rag_system", None)
    return rag_system


def db_agent_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """
    Agent 2: Database agent. Uses exactly one tool: query_rag_tool (RAG database).
    """
    logger.info("DB AGENT ACTIVATED")

    question = state.current_question
    new_step = state.step_count + 1

    rag_system = _get_rag_system_from_context(state, config)

    if not rag_system:
        tool_record = ToolCallRecord(
            tool_name="query_rag_tool",
            arguments={"question": question},
            status="error",
            source_info=None,
        )

        return {
            "tool_result": "Error: RAG system not available.",
            "source_info": {"source": "error", "info_type": "error"},
            "tool_name_used": "query_rag_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
        }

    try:
        result = rag_system.qa_chain.invoke({"query": question})

        answer = result.get("result", "")
        source_documents = result.get("source_documents", [])

        # Default source info
        source_info = {"source": "database", "info_type": "db", "metadata": ""}

        if source_documents:
            meta = source_documents[0].metadata
            source_info = {
                "source": meta.get("source", "database"),
                "info_type": meta.get("info_type", "db"),
                "metadata": json.dumps(meta),
            }

        tool_record = ToolCallRecord(
            tool_name="query_rag_tool",
            arguments={"question": question},
            status="success",
            source_info=source_info,
        )

        return {
            "tool_result": answer,
            "source_info": source_info,
            "source_documents": source_documents,
            "tool_name_used": "query_rag_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
            "need_critic": False,
        }

    except Exception as e:
        logger.exception("db_agent_node failed")

        tool_record = ToolCallRecord(
            tool_name="query_rag_tool",
            arguments={"question": question},
            status="error",
            source_info=None,
        )

        return {
            "tool_result": f"Error querying database: {e}",
            "source_info": {"source": "error", "info_type": "error"},
            "tool_name_used": "query_rag_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
        }


def internet_agent_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """
    Agent 3: Internet agent. Uses exactly one tool: internet_search_api_call_rag_tool.
    """
    logger.info("INTERNET AGENT ACTIVATED")

    question = state.current_question
    new_step = state.step_count + 1

    rag_system = _get_rag_system_from_context(state, config)

    if not rag_system:
        tool_record = ToolCallRecord(
            tool_name="internet_search_api_call_rag_tool",
            arguments={"question": question},
            status="error",
            source_info=None,
        )
        return {
            "tool_result": "Error: RAG system not available.",
            "source_info": {"source": "error", "info_type": "error"},
            "tool_name_used": "internet_search_api_call_rag_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
        }

    try:
        answer = rag_system.internet_search_api_call(question)

        source_info = {"source": "internet", "info_type": "internet"}

        tool_record = ToolCallRecord(
            tool_name="internet_search_api_call_rag_tool",
            arguments={"question": question},
            status="success",
            source_info=source_info,
        )

        return {
            "tool_result": answer,
            "source_info": source_info,
            "source_documents": [],
            "tool_name_used": "internet_search_api_call_rag_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
            "need_critic": False,
        }

    except Exception as e:
        logger.exception("internet_agent_node failed")

        tool_record = ToolCallRecord(
            tool_name="internet_search_api_call_rag_tool",
            arguments={"question": question},
            status="error",
            source_info=None,
        )

        return {
            "tool_result": f"Error searching internet: {e}",
            "source_info": {"source": "error", "info_type": "error"},
            "tool_name_used": "internet_search_api_call_rag_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
        }


def time_agent_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """Agent 4: Agent that calls only the time tool (get_time_in_city)."""
    
    logger.info("TIME AGENT ACTIVATED")

    question = state.current_question
    step = state.step_count

    rag_system = _get_rag_system_from_context(state, config)

    new_step = step + 1

    if not rag_system:
        tool_record = ToolCallRecord(
            tool_name="get_time_tool",
            arguments={"question": question},
            status="error",
            source_info=None,
        )

        return {
            "tool_result": "Error: RAG system not available.",
            "source_info": {"source": "error", "info_type": "error"},
            "tool_name_used": "get_time_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
        }

    try:
        answer = rag_system.get_time_in_city(question)

        source_info = {"source": "internet", "info_type": "time"}

        tool_record = ToolCallRecord(
            tool_name="get_time_tool",
            arguments={"question": question},
            status="success",
            source_info=source_info,
        )

        return {
            "tool_result": answer,
            "source_info": source_info,
            "source_documents": [],
            "tool_name_used": "get_time_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
            "need_critic": False,  # example explicit control
        }

    except Exception as e:
        logger.exception("time_agent_node failed")

        tool_record = ToolCallRecord(
            tool_name="get_time_tool",
            arguments={"question": question},
            status="error",
            source_info=None,
        )

        return {
            "tool_result": f"Error getting time: {e}",
            "source_info": {"source": "error", "info_type": "error"},
            "tool_name_used": "get_time_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
        }


def restaurant_agent_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """Agent 5: Agent that calls only the restaurant tool (get_restaurants_near_city)."""
    logger.info("RESTAURANT AGENT ACTIVATED")

    question = state.current_question
    step = state.step_count
    new_step = step + 1

    rag_system = _get_rag_system_from_context(state, config)

    if not rag_system:
        tool_record = ToolCallRecord(
            tool_name="find_restaurants_tool",
            arguments={"question": question},
            status="error",
            source_info=None,
        )

        return {
            "tool_result": "Error: RAG system not available.",
            "source_info": {"source": "error", "info_type": "error"},
            "tool_name_used": "find_restaurants_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
            "need_critic": False,
        }

    try:
        answer = rag_system.get_restaurants_near_city(question)

        source_info = {"source": "internet", "info_type": "internet"}

        tool_record = ToolCallRecord(
            tool_name="find_restaurants_tool",
            arguments={"question": question},
            status="success",
            source_info=source_info,
        )

        return {
            "tool_result": answer,
            "source_info": source_info,
            "source_documents": [],
            "tool_name_used": "find_restaurants_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
            "need_critic": False,
        }

    except Exception as e:
        logger.exception("restaurant_agent_node failed")

        tool_record = ToolCallRecord(
            tool_name="find_restaurants_tool",
            arguments={"question": question},
            status="error",
            source_info=None,
        )

        return {
            "tool_result": f"Error finding restaurants: {e}",
            "source_info": {"source": "error", "info_type": "error"},
            "tool_name_used": "find_restaurants_tool",
            "tool_calls": state.tool_calls + [tool_record],
            "step_count": new_step,
        }


def synthesizer_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """Agent 6: Synthesize final answer from tool result or generate direct answer."""
    logger.info("SYNTHESIZER ACTIVATED")

    question = state.current_question
    route = state.route
    messages = state.messages
    step = state.step_count
    new_step = step + 1

    if route == "direct":
        llm = _get_llm()

        prompt = (
            "Answer the following question concisely and accurately using your own knowledge. "
            "Do not use any tools.\n\n"
            f"Question: {question}"
        )

        response = llm.invoke(messages + [HumanMessage(content=prompt)])
        final = getattr(response, "content", str(response))

    else:
        final = state.tool_result or "I couldn't generate an answer."

    return {
        "final_answer": final,
        "step_count": new_step,
        "need_critic": False,          # synthesis completed
        "question_status": "completed"
    }



def critic_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """Review low-confidence route and optionally adjust."""
    logger.info("CRITIC ACTIVATED")

    new_step = state.step_count + 1

    return {
        "need_critic": False,
        "step_count": new_step,
    }



REFUSAL_MESSAGE = "I do not answer these types of questions."


def refusal_node(state: GraphState, config: RunnableConfig | None = None) -> dict:
    """Agent 7: When the router chose 'refuse', return a polite refusal and log it in tool_calls."""
    logger.info("REFUSAL NODE ACTIVATED")

    question = state.current_question
    new_step = state.step_count + 1

    refusal_record = ToolCallRecord(
        tool_name="refusal_node",
        arguments={
            "question": question,
            "reason": "illegal/harmful/unethical",
        },
        status="refused",
        source_info={"source": "agent", "info_type": "refusal"},
    )

    return {
        "tool_result": REFUSAL_MESSAGE,
        "final_answer": REFUSAL_MESSAGE,  # optionally finalize immediately
        "source_info": {"source": "agent", "info_type": "refusal"},
        "tool_name_used": "refusal_node",
        "tool_calls": state.tool_calls + [refusal_record],
        "question_status": "refused",
        "need_critic": False,
        "step_count": new_step,
    }



def route_after_router(state: GraphState) -> str:
    """Conditional edge: from Router to db_agent, internet_agent, time/restaurant agent, synthesizer, or critic."""
    route = state.route or "direct"
    if state.need_critic:
        return "critic"
    return route



def build_graph(rag_system: Any = None) -> StateGraph:
    """
    Build the LangGraph with multi agents, conditional edges, MemorySaver.
    Agents: Router (SOM), DB agent (query_rag_tool), Internet agent (internet_search_tool), Time agent (get_time_in_city), Restaurant agent (get_restaurants_near_city), Refusal agent (refusal_node), Critic agent (critic_node), Synthesizer agent (synthesizer_node).
    rag_system is passed at invoke time via config.configurable["rag_system"].
    """
 

    builder = StateGraph(GraphState)

    # Agents: Router (SOM), DB, Internet, Time, Restaurant, Refusal (no tools), Critic, Synthesizer
    builder.add_node("router", router_node)
    builder.add_node("db_agent", db_agent_node)
    builder.add_node("internet_agent", internet_agent_node)
    builder.add_node("time_agent", time_agent_node)
    builder.add_node("restaurant_agent", restaurant_agent_node)
    builder.add_node("refusal", refusal_node)
    builder.add_node("critic", critic_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "db": "db_agent",
            "internet": "internet_agent",
            "time": "time_agent",
            "restaurant": "restaurant_agent",
            "refuse": "refusal",
            "direct": "synthesizer",
            "critic": "critic",
        },
    )
    builder.add_edge("db_agent", "synthesizer")
    builder.add_edge("internet_agent", "synthesizer")
    builder.add_edge("time_agent", "synthesizer")
    builder.add_edge("restaurant_agent", "synthesizer")
    builder.add_edge("refusal", "synthesizer")
    builder.add_edge("critic", "synthesizer")
    builder.add_edge("synthesizer", END)

    memory = MemorySaver()
    compiled = builder.compile(checkpointer=memory)
    return compiled


def run_graph(compiled_graph, question: str, rag_system: Any, thread_id: str = "default") -> dict:
    """
    Run the graph with initial state. Injects rag_system via thread-local and RunnableConfig so tool nodes can use it.
    """

    initial = {
        "current_question": question,
        "messages": [],
        "step_count": 0,
    }

    config = RunnableConfig(configurable={"thread_id": thread_id, "rag_system": rag_system})

    try:
        _rag_system_local.value = rag_system
        result = compiled_graph.invoke(initial, config=config)
        return result
    finally:
        _rag_system_local.value = None
