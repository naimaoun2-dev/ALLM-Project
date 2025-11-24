from langchain.tools import tool
from typing import Optional, Dict
from datetime import datetime
import streamlit as st



@tool
def query_rag_tool(question: str) -> str:
    """
    Query the RAG knowledge base via session state.
    """
    rag = st.session_state.rag_system
    answer, info = rag.query(question)
    return f"{answer}\n[Source: {info.get('source')}, Type: {info.get('info_type')}]"

@tool
def internet_search_api_call_rag_tool(question: str) -> str:
    """
    Tool wrapper for the RAG system's internet fallback search.
    Uses GeoNames API to fetch country information.
    """
    rag = st.session_state.rag_system

    answer = rag.internet_search_api_call(question)

    return f"{answer}\n[Source: internet, Type: internet]"


@tool
def get_time_tool(question: str) -> str:
    """
    Tool wrapper to get the local time in a city using timeanddate.com
    Expects question like 'Time in London' or 'What is the time in New York?'
    """
    # Extract city from the question (simple approach)
    lower_q = question.lower()
    if "time in" in lower_q:
        city = question.lower().split("time in")[-1].strip().title()
    else:
        city = question.strip().title()
    
    rag = st.session_state.rag_system

    answer = rag.get_time_in_city(city)
    return f"{answer}\n[Source: internet, Type: time]"    


    

