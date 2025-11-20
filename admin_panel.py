"""
Admin Panel with Password Protection
"""
import streamlit as st
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class AdminPanel:
    """Admin panel with password protection and lockout mechanism"""
    
    def __init__(self):
        self.admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
        self.max_attempts = 3
        self.lockout_duration = timedelta(seconds=30)
    
    def show_login(self):
        """Display login form"""
        # Check for lockout
        if st.session_state.lockout_until:
            if datetime.now() < st.session_state.lockout_until:
                remaining = (st.session_state.lockout_until - datetime.now()).total_seconds()
                st.error(f"🔒 Account locked. Try again in {int(remaining)} seconds.")
                return
            else:
                # Lockout expired
                st.session_state.lockout_until = None
                st.session_state.login_attempts = 0
        
        password = st.text_input(
            "Admin Password:",
            type="password",
            key="admin_password_input"
        )
        
        if st.button("Login"):
            if password == self.admin_password:
                st.session_state.admin_authenticated = True
                st.session_state.login_attempts = 0
                st.session_state.lockout_until = None
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                remaining_attempts = self.max_attempts - st.session_state.login_attempts
                
                if st.session_state.login_attempts >= self.max_attempts:
                    st.session_state.lockout_until = datetime.now() + self.lockout_duration
                    st.error(f"🔒 Too many failed attempts. Account locked for 30 seconds.")
                    st.session_state.login_attempts = 0
                else:
                    st.error(f"❌ Incorrect password. {remaining_attempts} attempts remaining.")
    
    def view_user_documents(self):
        """View all user-provided documents"""
        if not st.session_state.admin_authenticated:
            st.warning("Please login as admin first.")
            return
        
        if "rag_system" in st.session_state and st.session_state.rag_system:
            documents = st.session_state.rag_system.get_user_documents()
            
            if documents:
                st.subheader("User-Provided Documents")
                for i, doc in enumerate(documents, 1):
                    with st.expander(f"📄 {doc.get('title', 'Untitled')}"):
                        st.write(f"**Added at:** {doc.get('added_at', 'Unknown')}")
                        st.write(f"**Source:** {doc.get('source', 'user')}")
                        st.write(f"**Info Type:** {doc.get('info_type', 'user')}")
            else:
                st.info("No user-provided documents found.")
        else:
            st.warning("RAG system not initialized.")
    
    def delete_user_documents(self): 
        """Delete all user-provided documents""" 
        if not st.session_state.admin_authenticated: 
            st.warning("Please login as admin first.") 
           
            return 
        
        st.warning("⚠️ This will permanently delete all user-provided documents!") 
        st.text_input("Type 'DELETE' to confirm:", key="delete_confirm")

        if st.button("Delete All User Documents", type="primary", key="delete_all_btn"):
            
            if st.session_state.get("delete_confirm", "") == "DELETE":

                if "rag_system" in st.session_state and st.session_state.rag_system:
                    success = st.session_state.rag_system.delete_user_documents()
                    if success:
                        st.success("✅ All user-provided documents deleted successfully!")
                        st.rerun()
                    else:
                        st.info("No user documents to delete.")
                else:
                    st.error("RAG system not initialized.")
            else:
                st.error("Please type 'DELETE' to confirm deletion.")
