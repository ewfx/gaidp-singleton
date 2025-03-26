import re

import streamlit as st
import pdfplumber
import pandas as pd
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, Document
# from st_aggrid import AgGrid, GridOptionsBuilder

import os
from dotenv import load_dotenv

# Retrieve the API key
load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = st.secrets["google"]["api_key"]


# Initialize the GenAI model (Google Gemini)
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                          google_api_key=GOOGLE_API_KEY)

def display_styled_dataframe(df):
    # Convert to Streamlit's interactive table
    st.data_editor(
        df,
        hide_index=False,  # Show row numbers
        use_container_width=True,  # Stretch across screen
        column_config={
            col: st.column_config.Column(width="auto") for col in df.columns
        },
    )

    # Apply custom CSS styling
    st.markdown(
        """
        <style>
            /* Make headers bold */
            .stDataFrame th {
                font-weight: bold !important;
                background-color: #f4f4f4 !important;
            }

            /* Alternating row colors */
            .stDataFrame tbody tr:nth-child(odd) {
                background-color: #f9f9f9 !important;
            }

            /* Hover effect */
            .stDataFrame tbody tr:hover {
                background-color: #e6f7ff !important;
            }

            /* Adjust cell padding for better readability */
            .stDataFrame td {
                padding: 8px !important;
            }

            /* Fix column widths */
            .stDataFrame th, .stDataFrame td {
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Circular Progress Bar CSS
    circle_progress_css = """
    <style>
        .progress-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 10px;
        }
        .progress-circle {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: 8px solid #ddd;
            border-top-color: #1f77b4;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
    </style>
    """
    st.markdown(circle_progress_css, unsafe_allow_html=True)



def rules_extract_data():
    st.image("./image/Rule_Extraction.png")

    col1, col2 = st.columns(2)
    with col1:
        regulatory_file = st.file_uploader("📂 Upload Regulatory PDF", type=["pdf"])
    with col2:
        dataset_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    if regulatory_file is not None and dataset_file is not None:
        extract_rules_from_pdf(regulatory_file,dataset_file)
    return None


def fix_truncated_json(response_text):
    """
    Fixes a truncated JSON string by ensuring it starts with '{' and ends with '}'.
    If the last '}' is missing, it appends one at the correct position.
    """
    # Find first `{` and last `}`
    match = re.search(r'{', response_text)  # First occurrence of `{`
    last_match = re.finditer(r'}', response_text)  # All occurrences of `}`
    last_match = list(last_match)  # Convert iterator to list

    if not match or not last_match:
        return None  # No valid JSON structure found

    last_pos = last_match[-1].end()  # Position after last `}`

    # Extract valid JSON portion
    fixed_json = response_text[:last_pos] + "]"   # Add an extra `}` just in case

    return fixed_json

def extract_rules_from_pdf(pdf_file, df):
    """Extract validation rules from the uploaded PDF using LLM & ChromaDB."""

    df = pd.read_csv(df)
    if "extracted_rules" in st.session_state:
        st.success("✅ Rules already extracted. Download below.")
        extracted_rules = st.session_state["extracted_rules"]
        st.download_button("📥 Download Rules JSON", data=json.dumps(extracted_rules, indent=4),
                           file_name="refined_validation_rules.json", mime="application/json")


    else:

        # Load PDF text efficiently
         # Initialize progress bar
        extracted_text = []
        col1, col2 = st.columns(2)
        with col1:
            if "extracted_text" not in st.session_state:
                pdf_progress = st.progress(0)

                extracted_text = []
                with pdfplumber.open(pdf_file) as pdf:
                    total_pages = len(pdf.pages)
                    for i, page in enumerate(pdf.pages):
                        extracted_text.append(page.extract_text() or "")
                        pdf_progress.progress((i + 1) / total_pages)

                pdf_progress.empty()
                st.session_state["extracted_text"] = extracted_text
                st.success("✅ PDF text extraction complete!")
            else:
                extracted_text = st.session_state["extracted_text"]


        full_text = "\n\n".join(extracted_text)

        with col2:
            if "extracted_tables" not in st.session_state:
                table_progress = st.progress(0)

                tables = []
                with pdfplumber.open(pdf_file) as pdf:
                    total_pages = len(pdf.pages)
                    for i, page in enumerate(pdf.pages):
                        for table in page.extract_tables():
                            tables.append(pd.DataFrame(table))
                        table_progress.progress((i + 1) / total_pages)

                table_progress.empty()
                st.session_state["extracted_tables"] = tables
                st.success("✅ Table extraction complete!")
            else:
                tables = st.session_state["extracted_tables"]

        # If tables exist, convert to text and append to full_text
        if tables:
            table_text = "\n\n".join([df.to_string(index=False) for df in tables])
            full_text += "\n\n" + table_text  # Append structured tables to the text

        # Intelligent Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  # Larger but fewer chunks
        chunks = text_splitter.split_text(full_text)

        # Convert document chunks to LangChain format
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Store Chunks in ChromaDB (Persistent Storage)
        persist_dir = "./chroma_db"
        vector_store = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_dir)


        # Retrieve Relevant Sections in Batches
        query = st.text_input("🔎 Enter query for rule extraction:",
                              value="Extract structured validation rules for corporate loans and risk reporting")

        seen_rules = set()
        if st.button("Run Extraction"):
            retrieved_docs = vector_store.similarity_search(query, k=8)
            retrieved_texts = [doc.page_content for doc in retrieved_docs]

            # Batch Processing for Rule Extraction

            extracted_rules = []
            progress_bar = st.progress(0)  # Initialize progress bar
            batch_size = 1  # Process 3 chunks at a time
            batched_chunks = [retrieved_texts[i:i + batch_size] for i in range(0, len(retrieved_texts), batch_size)]
            col1, col2 = st.columns(2)

            with col1:
                success_box = st.empty()  # Placeholder for successful batches

            with col2:
                error_box = st.empty()  # Placeholder for failed batches
                success_count = 0
                error_count = 0
            for i, batch in enumerate(batched_chunks):

                # Construct LLM Prompt
                column_list = df.columns.tolist()  # Use dataset column names for better accuracy
                prompt = f"""
                    Extract and structure the following regulatory rules for all the columns present in the dataset in JSON format and refer the following column names. Each rule should include:
                    - `rule_id`: Unique identifier.
                    - `description`: the exact type or the exact values that are allowed
                    - `applicable_columns`: The exact column names.
                    - `severity`: High, Medium, or Low.
                    - `condition`: A structured validation condition.
                    - `expected_data_type`: Specify whether the column is `string`, `integer`, `float`, `date`, or `boolean`.
                    column names:
                    {column_list}
                    
                    rules:
                    {retrieved_texts}
                    
                    **EXISTING RULES (DO NOT REPEAT THESE RULES):**
                    {json.dumps(extracted_rules, indent=2)}
                    
                    *NEW RULE EXTRACTION:**
                    - Generate only new rules that do not match any descriptions or conditions of the same column name in the existing rules above.
                    - For any flag based column_names , the value "Yes" can be considered the same as "true" and the value "No" can be considered the same as false
                    - The total number of generated rules in the extracted rules should be atleast more than 50 percent of the number of columns in the dataset : {len(df.columns.tolist())} 
                    - The rules can be divided if there are too many columns under the same rule
                    - Ensure uniqueness and do not duplicate existing validation logic.
                    - Date columns must enforce YYYY-MM-DD format and need not have T00:00:00 included in it
                    - Numeric columns must NOT have string-based rules
                    - There should be proper rules generated 
                    - String-based columns must only check textual constraints
    
                    
                """

                # Send query to Gemini
                response = chat_model([HumanMessage(content=prompt)])
                print("THe response is \n")
                try:
                    raw_response = response.content[7:-3].strip()
                    if raw_response.startswith("{") and raw_response.endswith("}"):
                        batch_rules = json.loads(raw_response)  # Parse JSON output
                    else:
                        raw_response = fix_truncated_json(raw_response)
                        batch_rules = json.loads(raw_response)
                    success_count+=1
                    extracted_rules.extend(batch_rules)  # Append to final list
                except json.JSONDecodeError:
                    error_count+=1
                    print(response.content[7:-3])
                    st.error(f"❌ Error parsing response for batch {i + 1}. Skipping.")

                progress_bar.progress((i + 1) / len(batched_chunks))
                success_box.success(f"**Successful Batches**                             {success_count}")
                error_box.error(f"**Failed Batches**                                     {error_count}")
            st.session_state["extracted_rules"] = extracted_rules  # ✅ Store rules in session
            st.success("✅ Rule extraction completed!")
            progress_bar.empty()  # Remove progress bar after completion
            # Step 6: Save Rules Persistently
            save_dir = "./data/output"
            os.makedirs(save_dir, exist_ok=True)
            rules_json_path = os.path.join(save_dir, "extracted_regulatory_rules.json")
            with open(rules_json_path, "w") as f:
                json.dump(extracted_rules, f, indent=4)

            if extracted_rules:
                rules_df = pd.DataFrame(extracted_rules)

                # Display formatted DataFrame in an expandable section

                display_styled_dataframe(rules_df)

                # Allow downloading the extracted rules in CSV format
                csv = rules_df.to_csv(index=False).encode("utf-8")
                st.markdown(
                    """
                    <style>
                        .right-align {
                            display: flex;
                            justify-content: flex-end;
                            padding: 10px;
                        }
                    </style>
                    <div class="right-align">
                    """,
                    unsafe_allow_html=True
                )
                st.download_button("📥 Download Extracted Rules (CSV)", data=csv, file_name="extracted_rules.csv",
                                       mime="text/csv")
                st.markdown("</div>", unsafe_allow_html=True)

                # Provide JSON Download as well
                st.download_button("📥 Download Extracted Rules JSON", data=json.dumps(extracted_rules, indent=4),
                                   file_name="refined_validation_rules.json", mime="application/json")
            else:
                st.warning("⚠️ No rules extracted. Please try again.")

