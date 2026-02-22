"""
RAG System using LangGraph (multi-agent), Chroma DB, and LangChain for retrieval.
"""
import os
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import requests
import string
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph_agents import build_graph, run_graph
import streamlit as st


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)
logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG System for querying Chroma DB and handling user documents"""
    
    def __init__(self):
        """Initialize the RAG system"""
        logger.info("[RAGSystem.__init__] ===== INITIALIZING RAG SYSTEM =====")
        
        try:
            logger.debug("[RAGSystem.__init__] Loading environment variables")
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not self.gemini_api_key:
                logger.error("[RAGSystem.__init__] GEMINI_API_KEY not found in environment variables")
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            logger.info("[RAGSystem.__init__] GEMINI_API_KEY loaded successfully")
            
            self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
            self.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            
            logger.info(f"[RAGSystem.__init__] Model: {self.model_name}")
            logger.info(f"[RAGSystem.__init__] Embedding model: {self.embedding_model}")
            logger.info(f"[RAGSystem.__init__] Use local embeddings: {self.use_local_embeddings}")
            logger.info(f"[RAGSystem.__init__] Persist directory: {self.persist_directory}")
            
            # Initialize embeddings - use local embeddings by default to avoid API quota issues
            logger.info("[RAGSystem.__init__] Initializing embeddings")
            if self.use_local_embeddings:
                # Use local sentence-transformers model (no API calls needed)
                logger.debug("[RAGSystem.__init__] Using local HuggingFace embeddings")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': 'cpu'}
                )
                logger.info("[RAGSystem.__init__] Local embeddings initialized")
            else:
                # Use Gemini embeddings (requires API key and quota)
                logger.debug("[RAGSystem.__init__] Using Gemini embeddings")
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.embedding_model,
                    google_api_key=self.gemini_api_key
                )
                logger.info("[RAGSystem.__init__] Gemini embeddings initialized")
            
            # Initialize LLM
            logger.info("[RAGSystem.__init__] Initializing LLM")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.7,
                google_api_key=self.gemini_api_key
            )
            logger.info("[RAGSystem.__init__] LLM initialized successfully")
        except Exception as e:
            logger.error(f"[RAGSystem.__init__] ERROR during initialization: {str(e)}", exc_info=True)
            raise

        # Multi-agent LangGraph (3 agents: Router with SOM, DB agent, Internet agent) with MemorySaver
        try:
            logger.info("[RAGSystem.__init__] Building LangGraph (3 agents, conditional edges, MemorySaver)")
            self.graph = build_graph()
            logger.info("[RAGSystem.__init__] LangGraph built successfully")
        except Exception as e:
            logger.error(f"[RAGSystem.__init__] Error building LangGraph: {str(e)}", exc_info=True)
            raise

        # Initialize vector store (create directory if it doesn't exist)
        try:
            logger.info("[RAGSystem.__init__] Setting up vector store")
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.debug(f"[RAGSystem.__init__] Created/verified directory: {self.persist_directory}")
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("[RAGSystem.__init__] Vector store initialized")
            
            # Text splitter
            logger.debug("[RAGSystem.__init__] Setting up text splitter")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            logger.info("[RAGSystem.__init__] Text splitter configured")
            
            # Initialize retrieval chain
            logger.info("[RAGSystem.__init__] Setting up retrieval chain")
            self._setup_retrieval_chain()
            logger.info("[RAGSystem.__init__] ===== RAG SYSTEM INITIALIZED SUCCESSFULLY =====")
        except Exception as e:
            logger.error(f"[RAGSystem.__init__] ERROR setting up vector store/chain: {str(e)}", exc_info=True)
            raise
    
    def _setup_retrieval_chain(self):
        """Setup the retrieval QA chain"""
        logger.info("[RAGSystem._setup_retrieval_chain] Setting up retrieval QA chain")
        logger.debug("[RAGSystem._setup_retrieval_chain] Creating prompt template")
        # Custom prompt template (keeping the existing one as requested)
        template = """
<identity>
You are a helpful and friendly assistant that answers questions about countries.
Your main topics are:
- Basic country information (for example: capital city, region, neighbors, population if available in context).
- Time in the capital city (current local time and time zone).
- Restaurants near the capital city or near a specific location in that country.

You should:
- Use simple, clear language that non-experts can understand.
- Be concise but complete. Explain extra details only when helpful.
- Be honest about what you know and what you do not know.
- Never invent facts. If you are not sure, say you are not sure and suggest how the user could check.
</identity>

<rules>
1. Use the provided {context} as your main source of truth when possible.
2. If the context says that the source is "user", clearly mention that this information was provided by a user (for example: "According to information provided by a user, ...").
3. If you do not find the answer in the context, or if the context is incomplete, decide whether to call one or more tools according to the <instructions>.
4. If, even after using tools, you still cannot answer confidently, say:
   - that you do not know, and
   - what extra information or tools would be needed.
5. Do not contradict the context. If tool results conflict with the context, prefer the most reliable and most recent source, and explain this briefly.
6. Always answer in a helpful and polite way. If the user asks multiple questions, try to answer all of them clearly.
</rules>

<instructions>
You have access to these four tools:

1) query_rag_database
   - What it does:
     Looks up information about countries from your internal RAG database (for example: capital city, region, descriptions, stored user notes).
   - When to use it:
     - When the user asks general questions about a country that are likely covered by your internal data.
       Examples:
       • "What is the capital of Japan?"
       • "Tell me about Lebanon."
       • "What is the capital of Brazil and where is it located?"
     - When you need the capital city before using other tools.
       Example:
       • User: "What time is it in the capital of Argentina?"  
         → First use query_rag_database to get the capital (Buenos Aires), then use get_time_tool.
     - When the user refers to information that might be stored in your RAG database (for example, previous user-provided notes about a country).

2) internet_search_api_call
   - What it does:
     Searches the internet for up-to-date or missing information that is not available or is incomplete in the RAG database.
   - When to use it:
     - When query_rag_database does not return enough information or returns nothing.
     - When the user asks for information that changes over time (for example: very recent events, new restaurant openings, recent travel restrictions) and this is not in the RAG database.
     - When the user asks about a country or place that does not appear in the RAG data.
       Examples:
       • "What is the capital of a newly created country X?" (not in the database)
       • "Are there any famous street-food areas near the capital of Thailand right now?"
   - How to behave:
     - Use the search results to support or update your answer.
     - If the search results are unclear or conflicting, say so and answer carefully.

3) get_time_tool
   - What it does:
     Returns the current local time in a given city or location.
   - When to use it:
     - When the user asks about the current time in a capital or a city.
       Examples:
       • "What time is it now in Paris?"
       • "What time is it in the capital of Canada?"
     - When the user asks about time zones related to a capital.
       Example:
       • "What is the time difference between the capital of Japan and London right now?"
   - How to combine it with other tools:
     - If the user asks "What time is it in the capital of X?" and you do NOT know the capital:
       1. Call query_rag_database (or internet_search_api_call, if needed) to find the capital name.
       2. Then call get_time_tool with that capital city.
     - After getting the time, explain it clearly (for example: "It is currently 15:30 in Tokyo").

4) find_restaurants_tool
   - What it does:
     Finds restaurants near a given location (for example: near the capital city or near specific coordinates).
   - When to use it:
     - When the user asks about restaurants near a capital or a place in a country.
       Examples:
       • "Show me restaurants near the capital of Italy."
       • "Find some vegetarian restaurants near the center of Madrid."
     - When the user asks for restaurant suggestions around a place that you can map to a city or coordinates.
   - How to combine it with other tools:
     - If the user says: "Find restaurants near the capital of X":
       1. First call query_rag_database (or internet_search_api_call, if needed) to get the capital name and location.
       2. Then call find_restaurants_tool with the capital's location.
     - If the user gives coordinates directly, you can call find_restaurants_tool with those coordinates.

GENERAL DECISION PROCESS:
1. Read the user's question carefully.
2. Identify the main task:
   - Is it about:
     a) Basic country info / capital? : Use query_rag_database first.
     b) Current time in a capital or city? : Make sure you know the city, then use get_time_tool.
     c) Restaurants near a capital or location? : Make sure you know the location, then use find_restaurants_tool.
     d) Something that seems missing from the RAG context? : Use internet_search_api_call.
3. Use tools in a logical order:
   - For capital-related time or restaurant questions:
     • Step 1: Get the capital from query_rag_database (or internet_search_api_call if needed).
     • Step 2: Use get_time_tool or find_restaurants_tool with that capital.
4. Minimize unnecessary tool calls:
   - If the needed information is already clearly present in {context}, you do not need to call a tool.
   - Only call a tool when it will add useful or required information.
5. After using tools:
   - Combine:
     • the given {context}, and
     • the tool outputs
     into one clear, final answer for the user.
   - If some parts of the answer come from tools, you can mention this briefly (for example: "Based on current data, the time in Tokyo is ...").
6. If you still cannot fully answer:
   - Clearly state what is missing.
   - Give the best partial answer you can without guessing.
</instructions>

<output_format>
Context:
{context}

Question:
{question}

Answer:
- First, briefly restate the user's request in your own words (one sentence).
- Then, give the main answer clearly and directly.
- If relevant, add short extra details (for example: time zone, brief capital description, or a short list of restaurants with names and basic info).
- If the information came from a "user" source in the context, say that this part was provided by a user.
- If you are unsure or some data is approximate, explain this clearly.
</output_format>"""

        
        try:
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            logger.debug("[RAGSystem._setup_retrieval_chain] Prompt template created")
            
            # Create retrieval chain
            logger.debug("[RAGSystem._setup_retrieval_chain] Creating RetrievalQA chain")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 10}
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            logger.info("[RAGSystem._setup_retrieval_chain] Retrieval QA chain created successfully")
        except Exception as e:
            logger.error(f"[RAGSystem._setup_retrieval_chain] ERROR: {str(e)}", exc_info=True)
            raise
    
    def query(self, question: str) -> Tuple[Dict, Dict, list, Dict, Dict]:
        """
        Query using the agent. The agent will first try to answer from its knowledge,
        then use tools (RAG database, internet search, etc.) if needed.

        Returns:
            (answer_dict, source_info, source_documents,tool_call, tool_details)
            answer_dict contains: {"output": str, "content": str} format
        """
        logger.info(f"[RAGSystem.query] ===== STARTING QUERY ===== Question: {question}")
        
        # Check if vectorstore has any documents
        try:
            logger.debug("[RAGSystem.query] Checking vectorstore collection count")
            collection_count = self.vectorstore._collection.count()
            logger.info(f"[RAGSystem.query] Vectorstore contains {collection_count} documents")
        except Exception as e:
            logger.warning(f"[RAGSystem.query] Error checking collection count: {str(e)}")
            collection_count = 0

        if collection_count == 0:
            logger.warning("[RAGSystem.query] Vectorstore is empty, returning empty database message")
            empty_message = (
                "I don't have any information in my database yet. "
                "Please provide the information using the 'Add Document' section in the sidebar."
            )
            return (
                {"output": empty_message, "content": empty_message},
                {"source": "empty", "info_type": "empty"},
                [],
                None,
                None,
                {"agent_used": None, "route": None, "route_reason": None, "question_status": "acceptable", "legality_score": 5},
            )

        # Run LangGraph 
        question_str = str(question).strip()
        if not question_str:
            raise ValueError("Question cannot be empty")

        try:
            logger.info("[RAGSystem.query] Invoking LangGraph with question")
            thread_id = str(id(st.session_state)) if hasattr(st, "session_state") else "default"
            result = run_graph(self.graph, question_str, self, thread_id=thread_id)
        except Exception as e:
            logger.error(f"[RAGSystem.query] ERROR invoking LangGraph: {str(e)}", exc_info=True)
            error_message = f"I encountered an error while processing your question: {str(e)}"
            return (
                {"output": error_message, "content": error_message},
                {"source": "error", "info_type": "error"},
                [],
                None,
                None,
                {"agent_used": None, "route": None, "route_reason": None, "question_status": "acceptable", "legality_score": 5},
            )

        # Map graph state to return format (answer, source_info, source_documents, tool_call, tool_details, agent_info)
        answer_text = result.get("final_answer") or result.get("tool_result") or "I couldn't generate an answer."
        source_info = result.get("source_info") or {"source": "agent", "info_type": "agent"}
        source_documents = result.get("source_documents") or []
        tool_calls_list = result.get("tool_calls") or []
        tool_name_used = result.get("tool_name_used")
        route = result.get("route") or "direct"

        # Human-readable agent that ran (for UI)
        route_to_agent = {
            "db": "Database (RAG) Agent",
            "internet": "Internet Agent",
            "time": "Time Agent",
            "restaurant": "Restaurant Agent",
            "refuse": "Refusal (no answer)",
            "direct": "Synthesizer (direct answer)",
        }
        agent_used = route_to_agent.get(route, "Synthesizer (direct answer)")
        question_status = result.get("question_status", "acceptable")
        legality_score = result.get("legality_score", 5 if question_status == "acceptable" else -5)
        agent_info = {
            "agent_used": agent_used,
            "route": route,
            "route_reason": result.get("route_reason"),
            "question_status": question_status,
            "legality_score": legality_score,
        }

        tool_call = None
        tool_details = None
        if route != "direct" and tool_calls_list:
            last_tc = tool_calls_list[-1] if isinstance(tool_calls_list[-1], dict) else {}
            tool_call = {
                "tool_name": tool_name_used or last_tc.get("tool_name", "unknown"),
                "arguments": last_tc.get("arguments", {}),
            }
            tool_details = {
                "status": last_tc.get("status", "success"),
                "source_info": source_info,
            }

        if route == "direct":
            source_info = {"source": "agent", "info_type": "agent"}
            source_documents = []
            

        logger.info(f"[RAGSystem.query] Final source info: {source_info}")
        logger.info(f"[RAGSystem.query] Agent used: {agent_used}")

        return (
            {"output": answer_text, "content": answer_text},
            source_info,
            source_documents,
            tool_call,
            tool_details,
            agent_info,
        )


    
    def add_user_document(self, text: str, title: str = "User Document", metadata: Optional[Dict] = None):
        """
        Add a user-provided document to the vector store
        
        Args:
            text: Document text
            title: Document title
            metadata: Additional metadata
        """
        logger.info(f"[RAGSystem.add_user_document] Adding document: {title}")
        logger.debug(f"[RAGSystem.add_user_document] Text length: {len(text)} chars")
        
        try:
            if not text.strip():
                logger.error("[RAGSystem.add_user_document] Document text is empty")
                raise ValueError("Document text cannot be empty")
            
            # Prepare metadata
            doc_metadata = {
                "source": "user",
                "info_type": "user",  # Default to "user" if not specified
                "title": title,
                "added_at": datetime.now().isoformat()
            }
            if metadata:
                doc_metadata.update(metadata)
                # Ensure info_type is set
                if "info_type" not in doc_metadata:
                    doc_metadata["info_type"] = "user"
            
            logger.debug(f"[RAGSystem.add_user_document] Metadata: {doc_metadata}")
            
            # Split text into chunks
            logger.debug("[RAGSystem.add_user_document] Splitting text into chunks")
            texts = self.text_splitter.split_text(text)
            logger.info(f"[RAGSystem.add_user_document] Split into {len(texts)} chunks")
            
            # Create documents
            documents = [
                Document(
                    page_content=chunk,
                    metadata={**doc_metadata, "chunk_index": i}
                )
                for i, chunk in enumerate(texts)
            ]
            
            # Add to vector store
            logger.info("[RAGSystem.add_user_document] Adding documents to vector store")
            ids = [f"{title}-{i}-{datetime.now().timestamp()}" for i in range(len(documents))]
            self.vectorstore.add_documents(documents, ids=ids)
            logger.debug(f"[RAGSystem.add_user_document] Added {len(documents)} documents with IDs")

            logger.debug("[RAGSystem.add_user_document] Persisting vector store")
            self.vectorstore.persist()
            
            # Reinitialize retrieval chain to include new documents
            logger.info("[RAGSystem.add_user_document] Reinitializing retrieval chain")
            self._setup_retrieval_chain()
            logger.info(f"[RAGSystem.add_user_document] Successfully added document: {title}")
        except Exception as e:
            logger.error(f"[RAGSystem.add_user_document] ERROR: {str(e)}", exc_info=True)
            raise
    
    def get_user_documents(self) -> List[Dict]:
        """
        Get all user-provided documents
        
        Returns:
            List of document metadata
        """
        try:
            collection = self.vectorstore._collection
            # Get all documents and filter by metadata
            all_results = collection.get(include=["metadatas", "documents"])
            
            # Extract unique documents by title
            documents = {}
            if all_results and "metadatas" in all_results:
                for i, metadata in enumerate(all_results["metadatas"]):
                    info_type = metadata.get("info_type", "db")
                    # Get user-provided documents (info_type == "user")
                    if info_type == "user":
                        title = metadata.get("title", "Untitled")
                        if title not in documents:
                            documents[title] = {
                                "title": title,
                                "added_at": metadata.get("added_at", "Unknown"),
                                "source": metadata.get("source", "user"),
                                "info_type": info_type,
                                "chunk_count": 1
                            }
                        else:
                            documents[title]["chunk_count"] += 1
            
            return list(documents.values())
        except Exception as e:
            print(f"Error getting user documents: {str(e)}")
            return []
    
    def delete_user_documents(self) -> bool:
        """
        Delete all user-provided documents
        
        Returns:
            True if successful
        """
        try:
            collection = self.vectorstore._collection
            
            # Get all documents and filter by metadata
            all_results = collection.get(
                        include=["metadatas"],
                        limit=None
                    )
            
            
            if all_results and "ids" in all_results and "metadatas" in all_results:
                # Find IDs of user documents (info_type == "user")
                user_ids = [
                    all_results["ids"][i]
                    for i, metadata in enumerate(all_results["metadatas"])
                    if metadata.get("info_type") == "user"
                ]
                
                if user_ids:
                    collection.delete(ids=user_ids)
                    self.vectorstore.persist()
                    # Reinitialize retrieval chain
                    self._setup_retrieval_chain()
                    return True
            return False
        except Exception as e:
            print(f"Error deleting user documents: {str(e)}")
            return False
            
    
    def internet_search_api_call(self, question: str) -> str:
        """
        Fetch country information from GeoNames and return relevant answers.

        This API handles questions like:
        - "What is the capital of Lebanon?"
        - "Beirut is the capital of what country?"
        - "List countries"
        - "How many countries are there?"

        Returns:
            str: Answer to the question based on GeoNames data.
        """
        import requests
        import unicodedata
        import string

        def clean_text(s: str) -> str:
            """Normalize, strip, lowercase a string for comparison."""
            return unicodedata.normalize("NFKC", s).strip().lower()

        print("Internet Search Called")
        url = "http://download.geonames.org/export/dump/countryInfo.txt"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.text

            # Split lines and ignore comments
            lines = [line for line in content.splitlines() if not line.startswith("#")]

            # Build country dictionary
            country_data = {}
            for line in lines:
                fields = line.split("\t")
                if len(fields) > 5:
                    country_name_raw = fields[4].strip()
                    capital_raw = fields[5].strip()
                    iso = fields[0].strip()

                    country_data[clean_text(country_name_raw)] = {
                        "capital": clean_text(capital_raw),
                        "iso": iso,
                        "name": country_name_raw  # keep original for display
                    }

            # Normalize the question
            q_lower = clean_text(question)
            import re

            q_tokens = set(re.findall(r"\w+", q_lower))  # split question into words

            for country_name, info in country_data.items():
                capital_clean = info["capital"]
                country_clean = country_name

                # Check if any token in the question matches the capital or country
                if capital_clean in q_tokens:
                    return f"{info['capital'].title()} is the capital of {info['name']}."
                if country_clean in q_tokens:
                    return f"The capital of {info['name']} is {info['capital'].title()}."
            # Handle generic list/count questions
            if "list" in q_lower and "countries" in q_lower:
                top_countries = [v["name"] for i, v in enumerate(country_data.values()) if i < 20]
                return "Countries (top 20 shown):\n" + "\n".join(top_countries)

            if "how many countries" in q_lower:
                return f"There are {len(country_data)} countries in the GeoNames database."

            # Fallback if nothing matches
            return "I fetched data from GeoNames, but I need a more specific question."

        except requests.exceptions.RequestException as e:
            print(f"Error fetching GeoNames data: {e}")
            return "Failed to retrieve data from GeoNames."



    def get_time_in_city(self, question: str) -> str:
        """
        Retrieve the current local time for a specified city.

        This method automatically detects whether the question is about time and
        extracts the city name from a variety of natural language formats. It
        supports questions such as:

            - "What time is it in London?"
            - "What is the time in London?"
            - "Time in Paris?"
            - "Current time for New York"
            - "London, what's the time?"

        Args:
            question (str): A natural language question containing a city and a request for the time.

        Returns:
            str: A human-readable string indicating the current local time in the specified city,
                or an error message if the city cannot be detected or the API request fails.

        Example:
            >>> get_time_in_city("What is the time in Tokyo?")
            "The current time in Tokyo is 14:32 PM (Timezone: Asia/Tokyo)."
        """
        logger.info(f"[RAGSystem.get_time_in_city] Getting time for: {question}")

        # Normalize question
        q_lower = question.lower()

        # Only proceed if the word "time" is in the question
        if "time" not in q_lower:
            logger.warning(f"[RAGSystem.get_time_in_city] No 'time' detected in question: {question}")
            return "Your question does not appear to be about time."

        # Improved regex to extract the city
        match = re.search(r'(?:time (?:in|for)|current time (?:in|for)|time is it in)\s+([\w\s_]+)', q_lower)
        if not match:
            logger.warning(f"[RAGSystem.get_time_in_city] Could not extract city from: {question}")
            return "Sorry, I could not detect a city in your question."

        city_name = match.group(1).strip().replace(" ", "_")
        logger.info(f"[RAGSystem.get_time_in_city] Extracted city: {city_name}")

        url = f"https://www.icalendar37.net/gadgets/timeInTheCity/?q={city_name}"
        logger.debug(f"[RAGSystem.get_time_in_city] Fetching from: {url}")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"[RAGSystem.get_time_in_city] API response received: {data}")

            if "time" in data:
                result = f"The current time in {data['city'].replace('_',' ')} is {data['time']} {data['APM']} (Timezone: {data['timezone']})."
                logger.info(f"[RAGSystem.get_time_in_city] Successfully retrieved time: {result}")
                return result
            else:
                logger.warning(f"[RAGSystem.get_time_in_city] No time data in response for: {city_name}")
                return f"Could not retrieve time for {city_name.replace('_',' ')}."

        except requests.exceptions.RequestException as e:
            logger.error(f"[RAGSystem.get_time_in_city] ERROR: {str(e)}", exc_info=True)
            return f"Failed to retrieve time: {e}"


    def _is_restaurant_query(self, question: str) -> bool:
        """Return True if the question is asking for restaurants near/in a city."""
        q = question.lower()
        if "restaurant" not in q and "restaurants" not in q:
            return False
        return bool(re.search(r"(restaurant|restaurants).*(in|near|around)", q))
    
    def _get_restaurants_via_overpass_geocode(self, city: str, radius_m: int = 3000, max_results: int = 10) -> str:
        """
        Fallback method: Use Overpass API to both geocode the city and find restaurants.
        This avoids Nominatim rate limiting issues.
        """
        user_agent = "LLM_RAG_Chatbot/1.0"
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # First, find the city coordinates using Overpass
        geocode_query = f"""
        [out:json][timeout:60];
        (
          relation["place"="city"]["name"="{city}"];
          relation["place"="town"]["name"="{city}"];
          node["place"="city"]["name"="{city}"];
          node["place"="town"]["name"="{city}"];
        );
        out center;
        """
        
        try:
            geo_resp = requests.post(
                overpass_url,
                data=geocode_query,
                headers={"User-Agent": user_agent, "Content-Type": "text/plain"},
                timeout=60
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
            elements = geo_data.get("elements", [])
            
            if not elements:
                # Try a broader search with "like" pattern
                geocode_query2 = f"""
                [out:json][timeout:60];
                (
                  relation["place"~"^(city|town)$"]["name"~"{city}",i];
                  node["place"~"^(city|town)$"]["name"~"{city}",i];
                );
                out center;
                """
                geo_resp2 = requests.post(
                    overpass_url,
                    data=geocode_query2,
                    headers={"User-Agent": user_agent, "Content-Type": "text/plain"},
                    timeout=60
                )
                geo_resp2.raise_for_status()
                geo_data = geo_resp2.json()
                elements = geo_data.get("elements", [])
            
            if not elements:
                raise ValueError(f"Could not find location data for '{city}' using Overpass API.")
            
            # Get coordinates from the first result
            element = elements[0]
            if "center" in element:
                lat = float(element["center"]["lat"])
                lon = float(element["center"]["lon"])
            elif "lat" in element:
                lat = float(element["lat"])
                lon = float(element["lon"])
            else:
                raise ValueError(f"Could not extract coordinates for '{city}'.")
            
            city_name = element.get("tags", {}).get("name", city)
            
        except Exception as e:
            raise ValueError(f"Error geocoding city via Overpass: {str(e)}")
        
        # Now find restaurants near the city
        overpass_query = f"""
        [out:json][timeout:60];
        (
          node["amenity"="restaurant"](around:{radius_m},{lat},{lon});
          way["amenity"="restaurant"](around:{radius_m},{lat},{lon});
          relation["amenity"="restaurant"](around:{radius_m},{lat},{lon});
        );
        out body {max_results};
        """
        
        try:
            overpass_resp = requests.post(
                overpass_url,
                data=overpass_query,
                headers={"User-Agent": user_agent, "Content-Type": "text/plain"},
                timeout=60
            )
            overpass_resp.raise_for_status()
            overpass_data = overpass_resp.json()
            elements = overpass_data.get("elements", [])
            
            if not elements:
                return f"I couldn't find restaurants within {radius_m/1000:.1f} km of {city_name}."
            
            lines = []
            for idx, element in enumerate(elements[:max_results], start=1):
                tags = element.get("tags", {})
                name = tags.get("name", "Unnamed restaurant")
                street = tags.get("addr:street", "")
                housenumber = tags.get("addr:housenumber", "")
                cuisine = tags.get("cuisine", "")
                address = ", ".join(filter(None, [street, housenumber]))
                description = name
                if address:
                    description += f" — {address}"
                if cuisine:
                    description += f" (Cuisine: {cuisine})"
                lines.append(f"{idx}. {description}")
            
            summary = f"Here are some restaurants near {city_name} (within {radius_m/1000:.1f} km):\n"
            summary += "\n".join(lines)
            summary += "\n\nData via OpenStreetMap/Overpass API."
            return summary
            
        except Exception as e:
            raise ValueError(f"Error fetching restaurants via Overpass: {str(e)}")

    def _extract_city_from_restaurant_query(self, question: str) -> Optional[str]:
        """Extract probable city name from a restaurant-related question."""
        cleaned = question.lower().translate(str.maketrans("", "", string.punctuation))
        patterns = [
            r"restaurants?\s+(?:near|in|around)\s+([a-zA-Z\s]+)",
            r"(?:near|in|around)\s+([a-zA-Z\s]+)\s+restaurants?",
            r"restaurants?\s+in\s+([a-zA-Z\s]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                city = match.group(1).strip()
                if city:
                    return city
        # Fallback: take text after "restaurants near"
        if "restaurants near" in cleaned:
            return cleaned.split("restaurants near", 1)[-1].strip()
        if "restaurants in" in cleaned:
            return cleaned.split("restaurants in", 1)[-1].strip()
        return None

    def get_restaurants_near_city(self, question: str, radius_m: int = 3000, max_results: int = 10) -> str:
        """
        Use Nominatim + Overpass API to find restaurants near the specified city.

        Args:
            question: User question that references restaurants.
            radius_m: Search radius in meters.
            max_results: Maximum number of restaurants to return.
        """
        logger.info(f"[RAGSystem.get_restaurants_near_city] Finding restaurants for: {question}")
        logger.debug(f"[RAGSystem.get_restaurants_near_city] Radius: {radius_m}m, Max results: {max_results}")
        import time
        
        city = self._extract_city_from_restaurant_query(question)
        if not city:
            logger.error(f"[RAGSystem.get_restaurants_near_city] Could not extract city from: {question}")
            raise ValueError("Could not identify a city in the question.")
        
        logger.info(f"[RAGSystem.get_restaurants_near_city] Extracted city: {city}")

        # Nominatim requires a proper User-Agent and has strict rate limiting
        # Use a descriptive User-Agent with application name
        user_agent = "LLM_RAG_Chatbot/1.0"
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        nominatim_params = {
            "format": "json",
            "q": city,
            "limit": 1,
            "addressdetails": 1
        }
        
        # Nominatim requires proper headers and rate limiting (max 1 request per second)
        headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Language": "en"
        }
        
        try:
            # Add a small delay to respect rate limits
            logger.debug("[RAGSystem.get_restaurants_near_city] Waiting 1 second for rate limiting")
            time.sleep(1)
            logger.debug(f"[RAGSystem.get_restaurants_near_city] Calling Nominatim API: {nominatim_url}")
            geo_resp = requests.get(
                nominatim_url,
                params=nominatim_params,
                headers=headers,
                timeout=60
            )
            
            # Check for rate limiting or forbidden errors
            if geo_resp.status_code == 403:
                logger.warning("[RAGSystem.get_restaurants_near_city] Nominatim returned 403, using Overpass fallback")
                # Try alternative: use Overpass API directly to geocode the city
                return self._get_restaurants_via_overpass_geocode(city, radius_m, max_results)
            
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
            logger.debug(f"[RAGSystem.get_restaurants_near_city] Nominatim returned {len(geo_data)} results")
            
            if not geo_data:
                logger.error(f"[RAGSystem.get_restaurants_near_city] No location data found for: {city}")
                raise ValueError(f"Could not find location data for '{city}'.")

            lat = float(geo_data[0]["lat"])
            lon = float(geo_data[0]["lon"])
            city_name = geo_data[0].get("display_name", city).split(",")[0]
            logger.info(f"[RAGSystem.get_restaurants_near_city] Found coordinates: {lat}, {lon} for {city_name}")
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.warning("[RAGSystem.get_restaurants_near_city] HTTP 403 error, using Overpass fallback")
                # Fallback to Overpass-only geocoding
                return self._get_restaurants_via_overpass_geocode(city, radius_m, max_results)
            logger.error(f"[RAGSystem.get_restaurants_near_city] HTTP error: {str(e)}", exc_info=True)
            raise ValueError(f"Error fetching location data: {str(e)}")
        except Exception as e:
            logger.error(f"[RAGSystem.get_restaurants_near_city] ERROR: {str(e)}", exc_info=True)
            raise ValueError(f"Error fetching location data: {str(e)}")

        # Search for restaurants (nodes, ways, and relations)
        overpass_query = f"""
        [out:json][timeout:60];
        (
          node["amenity"="restaurant"](around:{radius_m},{lat},{lon});
          way["amenity"="restaurant"](around:{radius_m},{lat},{lon});
          relation["amenity"="restaurant"](around:{radius_m},{lat},{lon});
        );
        out body {max_results};
        """
        overpass_url = "https://overpass-api.de/api/interpreter"
        logger.debug(f"[RAGSystem.get_restaurants_near_city] Querying Overpass API for restaurants")
        overpass_resp = requests.post(
            overpass_url,
            data=overpass_query,
            headers={"User-Agent": user_agent, "Content-Type": "text/plain"},
            timeout=60
        )
        overpass_resp.raise_for_status()
        overpass_data = overpass_resp.json()
        elements = overpass_data.get("elements", [])
        logger.info(f"[RAGSystem.get_restaurants_near_city] Found {len(elements)} restaurant elements")

        if not elements:
            logger.warning(f"[RAGSystem.get_restaurants_near_city] No restaurants found near {city_name}")
            return f"I couldn't find restaurants within {radius_m/1000:.1f} km of {city_name}."

        lines = []
        for idx, element in enumerate(elements[:max_results], start=1):
            tags = element.get("tags", {})
            name = tags.get("name", "Unnamed restaurant")
            street = tags.get("addr:street", "")
            housenumber = tags.get("addr:housenumber", "")
            cuisine = tags.get("cuisine")
            address = ", ".join(filter(None, [street, housenumber]))
            description = name
            if address:
                description += f" — {address}"
            if cuisine:
                description += f" (Cuisine: {cuisine})"
            lines.append(f"{idx}. {description}")

        summary = f"Here are some restaurants near {city_name} (within {radius_m/1000:.1f} km):\n"
        summary += "\n".join(lines)
        summary += "\n\nData via OpenStreetMap/Overpass API."
        logger.info(f"[RAGSystem.get_restaurants_near_city] Successfully compiled {len(lines)} restaurants")
        return summary
