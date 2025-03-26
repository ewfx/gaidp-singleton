import streamlit as st
from modules.rule_extraction import rules_extract_data
from modules.validation import validation_load_data
from modules.chat_assistant import chat_load_data
import asyncio
import sys
from modules.anomaly_detection import main
# from modules.machine_learning.clustering import clustering_main


# ✅ Permanent Fix for AsyncIO on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        /* Change sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #1E1E1E !important;  /* Change to your desired color */
            padding: 20px !important;  /* Adjust padding */
            border-radius: 15px !important;  /* Rounded corners */
            margin: 10px;  /* Add space around the sidebar */
        }

        /* Change sidebar text color */
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply a full-width container
st.markdown(
    """
    <style>
    .block-container {
        padding: 1rem;
        max-width: 95%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display table with full width


st.markdown(
    """
    <style>
        /* Reduce sidebar width */
        [data-testid="stSidebar"] {
            min-width: 290px;
            max-width: 290px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Sidebar Navigation
st.sidebar.title("🔍 Data Profiling")
page = st.sidebar.radio("Select an action", ["📄 Extract Rules", "📊 Validate Dataset", "💬 Chat Assistant", "🤖 Anomaly Detection"])

# File upload logic based on selection
dataset_file, rules_file, pdf_file, violations_file = None, None, None, None

if page == "📄 Extract Rules":
     rules_extract_data()

elif page == "📊 Validate Dataset":
    validation_load_data()

elif page == "💬 Chat Assistant":
    chat_load_data()

elif page == "🤖 Anomaly Detection":
    main()

