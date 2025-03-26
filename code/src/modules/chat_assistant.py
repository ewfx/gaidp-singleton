import pandas as pd
import streamlit as st
import json
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = st.secrets["google"]["api_key"]


chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


def chat_load_data():
    st.title("💬 Compliance Chat Assistant")

    st.markdown("""
        ### 🤖 AI-Powered Financial Compliance Chat  
        This interactive **Compliance Chat Assistant** helps analyze validation issues and refine **regulatory rules**.

        **What You Can Do Here:**
        
        💬 Ask about **validation errors** and why they occurred.  
        📖 Get **rule refinement suggestions** for dataset compliance.  
        ✅ Receive **explanations & recommended fixes** for violations.  

        ### 🚀 Why Use This?
        ✅ **Instant AI-powered compliance guidance**.  
        ✅ **Clarifies errors with structured explanations**.  
        ✅ **Enhances dataset accuracy and compliance understanding**.  
        """)

    validation_rules_file = st.file_uploader("📂 Upload Extracted Rules JSON for the chat assistant", type=["json"])
    extracted_violation_file = st.file_uploader("📂 Upload Extracted violations JSON for the chat assistant", type=["json"])

    if validation_rules_file is not None and extracted_violation_file is not None:
        chat_ui(validation_rules_file,extracted_violation_file)

    return None


def chat_ui(extracted_rules_json, violated_rules_json):

    extracted_rules_json = json.load(extracted_rules_json)
    violated_rules_json = json.load(violated_rules_json)
    # Store conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role, content = message["role"], message["content"]
        align = "flex-end" if role == "user" else "flex-start"
        bg_color = "#1F7A8C" if role == "user" else "#3A3A3A"

        st.markdown(
            f"""
                <div style="display: flex; justify-content: {align}; margin: 5px 0;">
                    <div style="background-color: {bg_color}; 
                                color: white;
                                padding: 15px; border-radius: 20px;
                                max-width: 70%;">
                        {content}
                    </div>
                </div>

            """,
            unsafe_allow_html=True
        )

    # Get user input
    user_input = st.chat_input("💡 Ask about violations or refine rules:")

    if user_input:
        # Store user message immediately & trigger AI processing
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.pending_query = user_input  # Store for processing
        st.rerun()  # Update UI immediately

    # Process AI response only if there's a pending query
    if "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")  # Remove from session state

        # Convert violations into searchable text
        violations_text = [
            f"Row {e['row_no']}, Column {e['column_name']}: {e['error_description']}. Suggested fix: {e.get('suggestion_to_fix', 'No fix available.')}. Severity: {e['severity']}"
            for e in violated_rules_json.get("errors", [])
        ]

        # Initialize embeddings & vector store
        vector_store = FAISS.from_texts(violations_text, embedding=embeddings)

        # Retrieve relevant validation errors
        retrieved_docs = vector_store.similarity_search(query, k=3)
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Construct AI prompt
        prompt = f"""
        You are an expert in financial compliance.
        **User Input:** {query}

        If the input is about a compliance issue or about the violations that happened , provide:
        - Explanation of the issue by mentioning which row and column and what issue
        - Suggested fix for the issue with mentioning the severity
        - Compliance recommendation
        from this information in {retrieved_text}

        If it's a rule refinement request, suggest modifications to:
        {json.dumps(extracted_rules_json, indent=2)}
        """

        # Get AI response
        response = chat_model([HumanMessage(content=prompt)])
        assistant_response = response.content

        # Store assistant response
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.rerun()  # Rerun again to show AI response
