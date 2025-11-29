from langchain.tools import tool
from typing import Optional, Dict
from datetime import datetime
import streamlit as st
import logging
import time


# Setup logging
logger = logging.getLogger(__name__)


@tool
def query_rag_tool(question: str) -> str:
    """
    Search the internal RAG database for information about countries, capitals, regions, and user-provided documents.
    
    Use this tool when:
    - You need to look up information that might be stored in the database (e.g., capital cities, country descriptions, user notes)
    - The user asks about a country and you want to check if there's stored information
    - You need the capital city name before using other tools (like get_time_tool or find_restaurants_tool)
    
    IMPORTANT: Only use this tool if you cannot answer from your own knowledge or if you need to verify/retrieve stored information.
    The database contains user-populated documents and country information. If the database returns no results or incomplete information,
    you should then use internet_search_api_call_rag_tool to find the answer.
    
    Examples:
    
    Example 1:
    User: "What is the capital of Japan?"
    Agent reasoning: I know Tokyo is the capital of Japan, but let me check the database to see if there's additional stored information.
    Tool call: query_rag_tool("What is the capital of Japan?")
    Tool response: {"answer": "The capital of Japan is Tokyo.", "source_info": {"source": "database", "info_type": "db"}, ...}
    Agent: "The capital of Japan is Tokyo."
    
    Example 2:
    User: "Tell me about Lebanon"
    Agent reasoning: I should check the database first to see if there's detailed information about Lebanon stored there.
    Tool call: query_rag_tool("Tell me about Lebanon")
    Tool response: {"answer": "Lebanon is a country in the Middle East. The capital is Beirut...", "source_info": {"source": "user", "info_type": "user"}, ...}
    Agent: "According to information provided by a user, Lebanon is a country in the Middle East. The capital is Beirut..."
    
    Returns a dictionary with:
    - answer: The answer string from the database
    - source_info: Metadata indicating the source (database/user) and info_type
    - source_documents: List of retrieved document chunks
    """
    try:
        
        logger.info(f"[TOOL: query_rag_tool] Starting RAG database query for: {question}")
        rag = st.session_state.rag_system

        logger.debug(f"[TOOL: query_rag_tool] Invoking qa_chain with query: {question}")
        start_time = time.time()
        result = rag.qa_chain.invoke({"query": question})
        end_time = time.time()
        answer = result["result"]
        source_documents = result.get("source_documents", [])
        
        logger.info(f"[TOOL: query_rag_tool] Retrieved {len(source_documents)} document chunks")

        # Build metadata for UI
        source_info = {"source": "SELF", "info_type": "SELF"}

        if source_documents:
            metadata = source_documents[0].metadata
            source_info = {
                "source": metadata.get("source", "database"),
                "info_type": metadata.get("info_type", "db"),
                "metadata": metadata
            }
            logger.debug(f"[TOOL: query_rag_tool] Source info: {source_info}")

        # Store metadata in session state for later retrieval
        if "last_tool_metadata" not in st.session_state:
            st.session_state.last_tool_metadata = {}
        st.session_state.last_tool_metadata["query_rag_tool"] = {
            "source_info": source_info,
            "source_documents": source_documents,
            "tool_call": {"tool_name": "query_rag_tool", "arguments": {"question": question}},
            "tool_details" : {
                "execution_time": end_time - start_time,
                "num_results_returned": len(source_documents),
                "raw_query": question,
                "answer": answer,
                "status": "success" if result else "no_results",
                "source_info": source_info
            }
        }
        
        logger.info(f"[TOOL: query_rag_tool] Successfully completed. Answer length: {len(answer)} chars")
        # Return just the answer string
        return answer
    except Exception as e:
        logger.error(f"[TOOL: query_rag_tool] ERROR: {str(e)}", exc_info=True)
        error_msg = f"Error querying database: {str(e)}"
        if "last_tool_metadata" not in st.session_state:
            st.session_state.last_tool_metadata = {}
        st.session_state.last_tool_metadata["query_rag_tool"] = {
            "source_info": {"source": "error", "info_type": "error"},
            "source_documents": [],
            "tool_call": {"tool_name": "query_rag_tool", "arguments": {"question": question}},
            "tool_details" : {
                "execution_time": 0,
                "num_results_returned": 0,
                "raw_query": question,
                "answer": None,
                "status": "no_results",
                "source_info": {"source": "error", "info_type": "error"}
            }
        }
        return error_msg


@tool
def internet_search_api_call_rag_tool(question: str) -> str:
    """
    Search the internet using GeoNames API for country information, capitals, and geography facts.
    
    Use this tool when:
    - You cannot answer the question from your own knowledge
    - The query_rag_tool returned no results or incomplete information
    - The user asks about a country or capital that you don't know or need to verify
    - You need current, up-to-date information that might not be in the database
    
    This tool searches GeoNames, a comprehensive geographical database. It's particularly useful for:
    - Finding capital cities of countries
    - Getting country information not in your knowledge base
    - Verifying geographical facts
    
    Examples:
    
    Example 1:
    User: "What is the capital of Bhutan?"
    Agent reasoning: I'm not certain about Bhutan's capital. Let me search the internet.
    Tool call: internet_search_api_call_rag_tool("What is the capital of Bhutan?")
    Tool response: {"answer": "The capital of Bhutan is Thimphu.", "source_info": {"source": "internet", "info_type": "internet"}, ...}
    Agent: "The capital of Bhutan is Thimphu."
    
    Example 2:
    User: "Beirut is the capital of what country?"
    Agent reasoning: I know Beirut is a city, but I should verify which country it's the capital of.
    Tool call: internet_search_api_call_rag_tool("Beirut is the capital of what country?")
    Tool response: {"answer": "The capital of Lebanon is Beirut.", "source_info": {"source": "internet", "info_type": "internet"}, ...}
    Agent: "Beirut is the capital of Lebanon."
    
    Returns a dictionary with:
    - answer: The answer string from GeoNames
    - source_info: Metadata indicating the source is internet
    - source_documents: Empty list (internet search doesn't return document chunks)
    """
    try:
        logger.info(f"[TOOL: internet_search_api_call_rag_tool] Starting internet search for: {question}")
        rag = st.session_state.rag_system

        logger.debug(f"[TOOL: internet_search_api_call_rag_tool] Calling internet_search_api_call method")
        start_time = time.time()
        answer = rag.internet_search_api_call(question)
        end_time = time.time()
        logger.info(f"[TOOL: internet_search_api_call_rag_tool] Successfully retrieved answer. Length: {len(answer)} chars")

        # Store metadata in session state for later retrieval
        if "last_tool_metadata" not in st.session_state:
            st.session_state.last_tool_metadata = {}
        st.session_state.last_tool_metadata["internet_search_api_call_rag_tool"] = {
            "source_info": {"source": "internet", "info_type": "internet"},
            "source_documents": [],
            "tool_call": {"tool_name": "internet_search_tool", "arguments": {"question": question}},
            "tool_details" : {
                "execution_time": end_time - start_time,
                "num_results_returned": 0,
                "raw_query": question,
                "answer": answer,
                "status": "success" if answer else "no_results",
                "source_info": {"source": "internet", "info_type": "internet"}
            }
        }
        
        return answer
    except Exception as e:
        logger.error(f"[TOOL: internet_search_api_call_rag_tool] ERROR: {str(e)}", exc_info=True)
        error_msg = f"Error searching internet: {str(e)}"
        if "last_tool_metadata" not in st.session_state:
            st.session_state.last_tool_metadata = {}
        st.session_state.last_tool_metadata["internet_search_api_call_rag_tool"] = {
            "source_info": {"source": "error", "info_type": "error"},
            "source_documents": [],
            "tool_call": {"tool_name": "internet_search_tool", "arguments": {"question": question}},
            "tool_details" : {
                "execution_time": 0,
                "num_results_returned": 0,
                "raw_query": question,
                "answer": None,
                "status": "no_results",
                "source_info": {"source": "error", "info_type": "error"}
            }
            
        }
        return error_msg


@tool
def get_time_tool(question: str) -> str:
    """
    Get the current local time and timezone for a specific city or location.
    
    Use this tool when:
    - The user asks "What time is it in [city]?" or "What's the time in [city]?"
    - The user asks about the time in a capital city (you may need to get the capital name first using query_rag_tool)
    - The user wants to know the current time in any city worldwide
    
    IMPORTANT: If the user asks "What time is it in the capital of [country]?", you should:
    1. First use query_rag_tool to find the capital city name
    2. Then use get_time_tool with that capital city name
    
    Examples:
    
    Example 1:
    User: "What time is it in Tokyo?"
    Agent reasoning: The user wants the current time in Tokyo. I can directly use get_time_tool.
    Tool call: get_time_tool("What time is it in Tokyo?")
    Tool response: {"answer": "The current time in Tokyo is 15:30 PM (Timezone: Asia/Tokyo).", "source_info": {"source": "internet", "info_type": "time"}, ...}
    Agent: "The current time in Tokyo is 15:30 PM (Timezone: Asia/Tokyo)."
    
    Example 2:
    User: "What time is it in the capital of Argentina?"
    Agent reasoning: I need the capital first, then get the time.
    Step 1 - Tool call: query_rag_tool("What is the capital of Argentina?")
    Step 1 - Response: {"answer": "The capital of Argentina is Buenos Aires.", ...}
    Step 2 - Tool call: get_time_tool("What time is it in Buenos Aires?")
    Step 2 - Response: {"answer": "The current time in Buenos Aires is 12:45 PM (Timezone: America/Argentina/Buenos_Aires).", ...}
    Agent: "The capital of Argentina is Buenos Aires. The current time there is 12:45 PM (Timezone: America/Argentina/Buenos_Aires)."
    
    Returns a dictionary with:
    - answer: The current time and timezone information
    - source_info: Metadata indicating the source is internet and info_type is time
    - source_documents: Empty list
    """
    try:
        logger.info(f"[TOOL: get_time_tool] Getting time for question: {question}")
        

        rag = st.session_state.rag_system
        logger.debug(f"[TOOL: get_time_tool] Calling get_time_in_city method")
        start_time = time.time()
        answer = rag.get_time_in_city(question)
        end_time = time.time()
        logger.info(f"[TOOL: get_time_tool] Successfully retrieved time. Answer: {answer}")

        # Store metadata in session state for later retrieval
        if "last_tool_metadata" not in st.session_state:
            st.session_state.last_tool_metadata = {}
        st.session_state.last_tool_metadata["get_time_tool"] = {
            "source_info": {"source": "internet", "info_type": "internet"},
            "source_documents": [],
            "tool_call": {"tool_name": "get_time_tool", "arguments": {"question": question}},
            "tool_details" : {
                "execution_time": end_time - start_time,
                "num_results_returned": 0,
                "raw_query": question,
                "answer": answer,
                "status": "success" if answer else "no_results",
                "source_info": {"source": "internet", "info_type": "internet"}
            }
        }
        
        return answer
    except Exception as e:
        logger.error(f"[TOOL: get_time_tool] ERROR: {str(e)}", exc_info=True)
        error_msg = f"Error getting time: {str(e)}"
        if "last_tool_metadata" not in st.session_state:
            st.session_state.last_tool_metadata = {}
        st.session_state.last_tool_metadata["get_time_tool"] = {
            "source_info": {"source": "error", "info_type": "error"},
            "source_documents": [],
            "tool_call": {"tool_name": "get_time_tool", "arguments": {"question": question}},
            "tool_details" : {
                "execution_time": 0,
                "num_results_returned": 0,
                "raw_query": question,
                "answer": None,
                "status": "no_results",
                "source_info": {"source": "error", "info_type": "error"}
            }
        }
        return error_msg

@tool
def find_restaurants_tool(question: str) -> str:
    """
    Find restaurants near a specific city or location using OpenStreetMap data.
    
    Use this tool when:
    - The user asks "Find restaurants near [city]" or "Show me restaurants in [city]"
    - The user asks for restaurants near a capital city (you may need to get the capital name first)
    - The user wants restaurant recommendations for a specific location
    
    IMPORTANT: If the user asks "Find restaurants near the capital of [country]?", you should:
    1. First use query_rag_tool to find the capital city name
    2. Then use find_restaurants_tool with that capital city name
    
    This tool searches within approximately 3km of the specified location and returns up to 10 restaurants
    with their names, addresses, and cuisine types when available.
    
    Examples:
    
    Example 1:
    User: "Find restaurants near Paris"
    Agent reasoning: The user wants restaurants in Paris. I can directly use find_restaurants_tool.
    Tool call: find_restaurants_tool("Find restaurants near Paris")
    Tool response: "Here are some restaurants near Paris (within 3.0 km):\n1. Le Comptoir du Relais — 9 Carrefour de l'Odéon (Cuisine: French)\n2. L'As du Fallafel — 34 Rue des Rosiers (Cuisine: Middle Eastern)\n..."
    Agent: "Here are some restaurants near Paris:\n1. Le Comptoir du Relais — 9 Carrefour de l'Odéon (Cuisine: French)\n..."
    
    Example 2:
    User: "Show me restaurants near the capital of Italy"
    Agent reasoning: I need the capital first, then find restaurants.
    Step 1 - Tool call: query_rag_tool("What is the capital of Italy?")
    Step 1 - Response: {"answer": "The capital of Italy is Rome.", ...}
    Step 2 - Tool call: find_restaurants_tool("Find restaurants near Rome")
    Step 2 - Response: "Here are some restaurants near Rome (within 3.0 km):\n1. Trattoria da Enzo — Via dei Vascellari, 29 (Cuisine: Italian)\n..."
    Agent: "The capital of Italy is Rome. Here are some restaurants near Rome:\n1. Trattoria da Enzo — Via dei Vascellari, 29 (Cuisine: Italian)\n..."
    
    Returns a string with a list of restaurants, their addresses, and cuisine types.
    """
    try:
        logger.info(f"[TOOL: find_restaurants_tool] Finding restaurants for: {question}")
        rag = st.session_state.rag_system
        
        logger.debug(f"[TOOL: find_restaurants_tool] Calling get_restaurants_near_city method")
        start_time = time.time()
        answer = rag.get_restaurants_near_city(question)
        end_time = time.time()
        logger.info(f"[TOOL: find_restaurants_tool] Successfully found restaurants. Response length: {len(answer)} chars")
        
        # Store metadata in session state for later retrieval
        if "last_tool_metadata" not in st.session_state:
            st.session_state.last_tool_metadata = {}
        st.session_state.last_tool_metadata["find_restaurants_tool"] = {
            "source_info": {"source": "internet", "info_type": "internet"},
            "source_documents": [],
            "tool_call": {"tool_name": "find_restaurants_tool", "arguments": {"question": question}},
            "tool_details" : {
                "execution_time": end_time - start_time,
                "num_results_returned": 0,
                "raw_query": question,
                "answer": answer,
                "status": "success" if answer else "no_results",
                "source_info": {"source": "internet", "info_type": "internet"}
            }
        }
        
        return f"{answer}\n[Source: overpass, Type: internet]"
    except Exception as e:
        logger.error(f"[TOOL: find_restaurants_tool] ERROR: {str(e)}", exc_info=True)
        error_msg = f"Error finding restaurants: {str(e)}"
        if "last_tool_metadata" not in st.session_state:
            st.session_state.last_tool_metadata = {}
        st.session_state.last_tool_metadata["find_restaurants_tool"] = {
            "source_info": {"source": "error", "info_type": "error"},
            "source_documents": [],
            "tool_call": {"tool_name": "find_restaurants_tool", "arguments": {"question": question}},
            "tool_details" : {
                "execution_time": 0,
                "num_results_returned": 0,
                "raw_query": question,
                "answer": None,
                "status": "no_results",
                "source_info": {"source": "error", "info_type": "error"}
            }
        }
        return f"{error_msg}\n[Source: error, Type: error]"