"""
RAG System using LangChain and Chroma DB
"""
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pydantic.v1 import tools
import requests
import string
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.agents import initialize_agent, AgentType
from rag_utils import query_rag_tool, internet_search_api_call_rag_tool, get_time_tool
import streamlit as st
import unicodedata

class RAGSystem:
    """RAG System for querying Chroma DB and handling user documents"""
    
    def __init__(self):
        """Initialize the RAG system"""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        
        # Initialize embeddings - use local embeddings by default to avoid API quota issues
        if self.use_local_embeddings:
            # Use local sentence-transformers model (no API calls needed)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
        else:
            # Use Gemini embeddings (requires API key and quota)
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=self.gemini_api_key
            )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.7,
            google_api_key=self.gemini_api_key
        )

        tools = [
                    query_rag_tool,
                    internet_search_api_call_rag_tool,
                    get_time_tool
                ]

        self.agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True
            )
        
        # Initialize vector store (create directory if it doesn't exist)
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize retrieval chain
        self._setup_retrieval_chain()
    
    def _setup_retrieval_chain(self):
        """Setup the retrieval QA chain"""
        # Custom prompt template
        template = """<rules>
        Use the following pieces of context to answer the question at the end.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    If the context mentions that the source is "user", indicate that this information was provided by a user.
</rules>
                    
<output_format>
                    Context: {context}

                    Question: {question}

                    Answer: 
</output_format>"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 10}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def query(self, question: str) -> Tuple[str, Dict, list]:
        """
        Query using the agent.
        Tools will handle: 
            - RAG retrieval (query_rag_tool)
            - Internet fallback (internet_search_api_call_rag_tool)
            - Time lookup (get_time_tool)

        Returns:
            (answer, source_info, source_documents)
        """

        # --- 1. Check database first (optional, same as before) ---
        try:
            print("Collectinon Count Try")
            collection_count = self.vectorstore._collection.count()
        except:
            collection_count = 0

        if collection_count == 0:
            return (
                "I don't have any information in my database yet. "
                "Please provide the information using the 'Add Document' section in the sidebar.",
                {"source": "empty", "info_type": "empty"},
                []
            )

        # --- 2. Ask the agent to solve the query ---
        try:
            print("Invoking Agent")
            result = self.agent.invoke({"input": question, "chat_history":  st.session_state.messages})
        except Exception:
            print("Exception Invoking Agent")
            result = self.agent.run(question)

        # --- 3. Tool output is structured if a tool was used ---
        if isinstance(result, dict) and "answer" in result:
            print("In Answer")
            return (
                result["answer"],
                result.get("source_info", {"source": "agent", "info_type": "agent"}),
                result.get("source_documents", [])
            )

        # --- 4. Fallback: agent returned a plain string ---
        return (
            result,
            {"source": "agent", "info_type": "agent"},
            []
        )


    
    def add_user_document(self, text: str, title: str = "User Document", metadata: Optional[Dict] = None):
        """
        Add a user-provided document to the vector store
        
        Args:
            text: Document text
            title: Document title
            metadata: Additional metadata
        """
        if not text.strip():
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
        
        # Split text into chunks
        texts = self.text_splitter.split_text(text)
        
        # Create documents
        documents = [
            Document(
                page_content=chunk,
                metadata={**doc_metadata, "chunk_index": i}
            )
            for i, chunk in enumerate(texts)
        ]
        
        # Add to vector store
        ids = [f"{title}-{i}-{datetime.now().timestamp()}" for i in range(len(documents))]
        self.vectorstore.add_documents(documents, ids=ids)

        self.vectorstore.persist()
        
        # Reinitialize retrieval chain to include new documents
        self._setup_retrieval_chain()
    
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
        Get the current time in a city using the icalendar37.net gadget API.

        Args:
            question: A natural language question like "What is the time in London?"

        Returns:
            A string with the current local time in the requested city.
        """
        import re
        print("Start of Method")
        # Extract city name from the question
        match = re.search(r'time in ([\w\s_]+)', question, re.IGNORECASE)
        print("match", match)
        if not match:
            return "Sorry, I could not extract a city from your question."

        print("After If")
        city_name = match.group(1).strip().replace(" ", "_")


        url = f"https://www.icalendar37.net/gadgets/timeInTheCity/?q={city_name}"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "time" in data:
                return f"The current time in {data['city'].replace('_',' ')} is {data['time']} {data['APM']} (Timezone: {data['timezone']})."
            else:
                return f"Could not retrieve time for {city_name.replace('_',' ')}."

        except requests.exceptions.RequestException as e:
            return f"Failed to retrieve time: {e}"

    
      
