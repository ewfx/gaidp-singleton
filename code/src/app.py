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


# Streamlit Page Config

# Sidebar Navigation
st.sidebar.title("🔍 Data Profiling ( Get your results here )")
page = st.sidebar.radio("Select an action", ["📄 Extract Rules", "📊 Validate Dataset", "💬 Compliance Assistant", "🤖 Anomaly Detection"])

# File upload logic based on selection
dataset_file, rules_file, pdf_file, violations_file = None, None, None, None

if page == "📄 Extract Rules":
     rules_extract_data()

elif page == "📊 Validate Dataset":
    validation_load_data()

elif page == "💬 Compliance Assistant":
    chat_load_data()

elif page == "🤖 Anomaly Detection":
    main()

