"""
Streamlit Chatbot with RAG (Chroma DB) and Admin Panel
"""
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from rag_system import RAGSystem
from admin_panel import AdminPanel
from download_handler import DownloadHandler



# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LLM Chatbot with RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "lockout_until" not in st.session_state:
    st.session_state.lockout_until = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_tool_metadata" not in st.session_state:
     st.session_state.last_tool_metadata = {}    

def initialize_rag_system():
    """Initialize the RAG system"""
    if st.session_state.rag_system is None:
        try:
            st.session_state.rag_system = RAGSystem()
            return True
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")
            return False
    return True





def main():
    """Main application"""
    st.title("🤖 LLM Chatbot with RAG")
    st.markdown("Ask questions and get answers from the knowledge base or provide new information!")
    
    # Show initialization status (non-blocking)
    if st.session_state.rag_system is None:
        with st.spinner("Initializing RAG system..."):
            if initialize_rag_system():
                st.success("✅ RAG system ready!")
            else:
                st.warning("⚠️ RAG system initialization failed. You can still try to chat, but it may not work properly.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Admin Panel
        st.subheader("🔐 Admin Panel")
        admin_panel = AdminPanel()
        
        if st.session_state.admin_authenticated:
            st.success("✅ Admin Authenticated")
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.session_state.login_attempts = 0
                st.session_state.lockout_until = None
                st.rerun()
            
            # Admin functions
            st.subheader("Admin Functions")
           

            if st.button("View User Documents"):
                admin_panel.view_user_documents()

            if st.button("Delete User Documents"):
                st.session_state.show_delete = True

            if st.session_state.get("show_delete"):
                admin_panel.delete_user_documents()
            
            
        else:
            admin_panel.show_login()
        
        st.divider()
        
        # User Document Input
        st.subheader("📄 Add Document")
        st.markdown("Provide information if the database doesn't have it")
        
        doc_text = st.text_area(
            "Enter document text:",
            height=150,
            placeholder="Paste your document content here..."
        )
        
        doc_title = st.text_input("Document Title (optional):")
        
        if st.button("Save Document"):
            if doc_text.strip():
                if initialize_rag_system():
                    try:
                        st.session_state.rag_system.add_user_document(
                            text=doc_text,
                            title=doc_title or "User Provided Document",
                            metadata={
                                "source": "user",
                                "info_type": "user",
                                "added_at": datetime.now().isoformat()
                            }
                        )
                        st.success("✅ Document saved successfully!")
                        doc_text = ""
                        doc_title = ""
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving document: {str(e)}")
            else:
                st.warning("Please enter some text to save.")
        
        st.divider()
        
        # Download Options
        st.subheader("📥 Download Chat History")
        download_handler = DownloadHandler()
        
        if st.session_state.chat_history:
            col1, col2 = st.columns(2)
            with col1:
                pdf_data = download_handler.generate_pdf(st.session_state.chat_history)
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_data,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            
            with col2:
                excel_data = download_handler.generate_excel(st.session_state.chat_history)
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("No chat history to download yet. Start chatting to generate history!")
    

    # Main chat interface
    for message in st.session_state.messages:
        role = message.get("role")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(message.get("content", ""))

        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message.get("content", ""))

                # Display info badge
                info_type = message.get("info_type")
                if info_type:
                    if info_type == "user":
                        st.caption("ℹ️ Info Type: User-provided")
                    elif info_type == "db":
                        st.caption("📚 Info Type: Database")
                    elif info_type == "internet":
                        st.caption("🌐 Info Type: Internet")
                    elif info_type == "time":
                        st.caption("⏰ Info Type: Time")
                    else:
                        st.caption(f"Info Type: {info_type}")

                # Display tool call and details if present
                tool_call = message.get("tool_call")
                tool_details = message.get("tool_details")
                tool_name = message.get("tool_name")

                if tool_call:
                    st.write(f"🛠️ Tool used: {tool_name or 'Unknown'}")
                    st.json(tool_call)

                    if tool_details:
                        st.write("📄 Tool details:")
                        st.json(tool_details)

        elif role == "tool":
            with st.chat_message("assistant"):
                tool_name = message.get("tool_name", "Unknown Tool")
                with st.status(f"🔧 Tool result: {tool_name}"):
                    st.markdown(message.get("content", ""))

                    # Optionally show tool metadata if available
                    tool_call = message.get("tool_call")
                    tool_details = message.get("tool_details")
                    if tool_call:
                        st.write("🛠️ Tool metadata:")
                        st.json(tool_call)
                    if tool_details:
                        st.write("📄 Tool details:")
                        st.json(tool_details)

    
    # Chat input - always show, even if RAG system isn't initialized
    if prompt := st.chat_input("Ask a question..."):
        # Initialize RAG system if not already done
        if not initialize_rag_system():
            st.error("Failed to initialize RAG system. Please check your API keys in the .env file.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I'm sorry, but I cannot process your request right now. Please check the error message above."
            })
            st.rerun()
            return
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        
       # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query RAG system
                    answer, source_info, source_documents, tool_call, tool_details = st.session_state.rag_system.query(prompt)

                    # Format answer
                    if isinstance(answer, dict):
                        formatted_response = answer.get("output", answer.get("content", "")).strip()
                    else:
                        formatted_response = str(answer).strip()

                    st.markdown(formatted_response, unsafe_allow_html=False)
                    info_type = source_info.get("info_type", "agent")

                    # Determine tool_name
                    if info_type == "db":
                        tool_name = "vectorstore_retrieval"
                    elif info_type == "internet":
                        tool_name = "internet_search"
                        if source_info.get("source") == "overpass":
                            tool_name = "overpass_restaurants"
                    elif info_type == "time":
                        tool_name = "get_time_tool"
                    else:
                        tool_name = None

                    # Store assistant message with tool metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": formatted_response,
                        "source": source_info.get("source", "unknown") if source_info else "unknown",
                        "info_type": info_type,
                        "tool_call": tool_call,
                        "tool_details": tool_details
                    })

                    # Store tool message if tool was used
                    if tool_call:
                        tool_content = ""
                        if info_type == "db":
                            tool_content = f"Retrieved {len(source_documents)} document chunks"
                        elif info_type in ["internet", "time"]:
                            tool_content = answer

                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": tool_content,
                            "tool_call": tool_call,
                            "tool_details": tool_details
                        })

                        # Display tool metadata
                        st.write("🛠️ Tool was used:")
                        st.json(tool_call)
                        if tool_details:
                            st.write("📄 Tool details:")
                            st.json(tool_details)
                    else:
                        st.write("No tool was used.")

                    # Display source information
                    if info_type == "user":
                        st.info("ℹ️ This information was provided by a user")
                    elif info_type == "db":
                        st.success("📚 Retrieved from database")
                    elif info_type in ["internet", "time"]:
                        st.warning("🌐 Retrieved from internet")
                    else:
                        st.caption(f"Info Type: {info_type}")

                    # Display retrieved documents
                    if source_documents:
                        with st.expander("📄 Retrieved documents"):
                            for doc in source_documents:
                                st.markdown(f"**Chunk {doc.metadata.get('chunk_index', 0)}**: {doc.page_content}")
                                if "title" in doc.metadata:
                                    st.caption(f"Title: {doc.metadata['title']}")

                    # Add chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": formatted_response,
                        "source": source_info.get("source", "unknown") if source_info else "unknown",
                        "info_type": info_type,
                        "timestamp": datetime.now().isoformat()
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })





if __name__ == "__main__":
    main()

