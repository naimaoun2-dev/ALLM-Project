from langchain.tools import tool
from typing import Optional, Dict
from datetime import datetime
import streamlit as st



@tool
def query_rag_tool(question: str) -> dict:
    """
    Direct RAG retrieval tool.
    Uses the internal qa_chain, NOT rag.query() to avoid recursion.
    Returns structured metadata.
    """
    rag = st.session_state.rag_system

    result = rag.qa_chain.invoke({"query": question})
    answer = result["result"]
    source_documents = result.get("source_documents", [])

    # Build metadata for UI
    source_info = {"source": "database", "info_type": "db"}

    if source_documents:
        metadata = source_documents[0].metadata
        source_info = {
            "source": metadata.get("source", "database"),
            "info_type": metadata.get("info_type", "db"),
            "metadata": metadata
        }

    return {
        "answer": answer,
        "source_info": source_info,
        "source_documents": source_documents
    }


@tool
def internet_search_api_call_rag_tool(question: str) -> dict:
    """
    Internet fallback tool using GeoNames.
    Use this tool when the answer is **not available in the local knowledge base (RAG)** 
    or when the question requires **general world knowledge, current events, geography, 
    or facts about people, places, or things**. 
    For example, questions like "Beirut is the capital of what country?" 
    should use this tool.
    
    Returns a structured result containing:
        - answer: the answer string from the internet
        - source_info: metadata indicating the source is internet
        - source_documents: an empty list (can be extended if using sources)
    """
    rag = st.session_state.rag_system

    answer = rag.internet_search_api_call(question)

    return {
        "answer": answer,
        "source_info": {"source": "internet", "info_type": "internet"},
        "source_documents": []
    }


@tool
def get_time_tool(question: str) -> dict:
    """
    Get local time in a city.
    Returns structured result.
    """
    # Extract city
    lower_q = question.lower()
    if "time in" in lower_q:
        city = question.lower().split("time in")[-1].strip().title()
    else:
        city = question.strip().title()

    rag = st.session_state.rag_system
    answer = rag.get_time_in_city(city)

    return {
        "answer": answer,
        "source_info": {"source": "internet", "info_type": "time"},
        "source_documents": []
    }


    

