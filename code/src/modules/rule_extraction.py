import streamlit as st
import pdfplumber
import pandas as pd
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain.schema import HumanMessage, Document

def rules_extract_data():
    st.title("📄 AI-Powered Rule Extraction")

    st.markdown("""
        ### 🔍 Extract Regulatory Validation Rules  
        This module extracts **structured validation rules** from large regulatory PDFs using **Generative AI**.

        **How It Works:**
        1️⃣ Upload a **regulatory PDF**.  
        2️⃣ AI extracts **structured validation rules** from relevant sections.  
        3️⃣ Rules are formatted as **JSON** for easy validation against datasets.  
        4️⃣ Download extracted rules for further processing.  

        ### 🚀 Why Use This?
        ✅ **Automates rule extraction** from complex documents.  
        ✅ **Ensures regulatory compliance** by deriving logical validation rules.  
        ✅ **Saves time** by eliminating manual rule interpretation.  
        """)

    regulatory_file = st.file_uploader("📂 Upload Regulatory PDF", type=["pdf"])
    dataset_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    if regulatory_file is not None and dataset_file is not None:
        extract_rules_from_pdf(regulatory_file,dataset_file)
    return None

def extract_rules_from_pdf(pdf_file, df):
    """Extract validation rules from the uploaded PDF using LLM & ChromaDB."""

    df = pd.read_csv(df)

    chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="")

    # Load PDF text efficiently
     # Initialize progress bar
    extracted_text = []
    if "extracted_text" not in st.session_state:
        st.write("🔍 Extracting text from PDF... Please wait.")
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


    if "extracted_tables" not in st.session_state:
        st.write("🔍 Extracting tables from PDF... Please wait.")
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

    # Step 2: Intelligent Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  # Larger but fewer chunks
    chunks = text_splitter.split_text(full_text)

    # Convert document chunks to LangChain format
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Step 3: Store Chunks in ChromaDB (Persistent Storage)
    persist_dir = "./chroma_db"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="")
    vector_store = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_dir)

    # st.success("✅ Text extraction complete! Storing data for efficient retrieval.")

    # Step 4: Retrieve Relevant Sections in Batches
    # query = "Extract structured validation rules for corporate loans and risk reporting"

    query = st.text_input("🔎 Enter query for rule extraction:",
                          value="Extract structured validation rules for corporate loans and risk reporting")

    if st.button("Run Extraction"):
        retrieved_docs = vector_store.similarity_search(query, k=8)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

    # Step 5: Batch Processing for Rule Extraction

        extracted_rules = []
        progress_bar = st.progress(0)  # Initialize progress bar
        batch_size = 1  # Process 3 chunks at a time

        batched_chunks = [retrieved_texts[i:i + batch_size] for i in range(0, len(retrieved_texts), batch_size)]
        for i, batch in enumerate(batched_chunks):

            # Construct LLM Prompt
            column_list = df.columns.tolist()  # Use dataset column names for better accuracy
            prompt = f"""
                Extract and structure the following regulatory rules in JSON format and refer the following column names. Each rule should include:
                - `rule_id`: Unique identifier.
                - `description`: A clear explanation.
                - `applicable_columns`: The exact column names.
                - `severity`: High, Medium, or Low.
                - `condition`: A structured validation condition.
                - `expected_data_type`: Specify whether the column is `string`, `integer`, `float`, `date`, or `boolean`.
                
                Rules must be logically correct:
                ✔ Numeric columns must NOT have string-based rules
                ✔ Date columns must enforce YYYY-MM-DD format
                ✔ String-based columns must only check textual constraints
                ✔ The rules should not repeated multiple times for the same columns if its already been generated in the JSON

                
                column names:
                {column_list}
                
                rules:
                {retrieved_texts}
            """

            # Send query to Gemini
            response = chat_model([HumanMessage(content=prompt)])
            print("THe response is \n")
            print(type(response))
            print(response)
            try:
                batch_rules = json.loads(response.content[7:-3])  # Parse JSON output
                extracted_rules.extend(batch_rules)  # Append to final list
            except json.JSONDecodeError:
                st.error(f"❌ Error parsing response for batch {i + 1}. Skipping.")

            progress_bar.progress((i + 1) / len(batched_chunks))
        progress_bar.progress((len(batched_chunks)/len(batched_chunks)))

        st.success("✅ Rule extraction completed!")
        progress_bar.empty()  # Remove progress bar after completion
            # Step 6: Save Rules Persistently
        rules_json_path = "./data/refined_validation_rules.json"
        with open(rules_json_path, "w") as f:
            json.dump(extracted_rules, f, indent=4)

        st.success("✅ Rules extracted successfully!")
        st.json(extracted_rules)

        # Provide download button
        st.download_button("📥 Download Rules JSON", data=json.dumps(extracted_rules, indent=4),
                               file_name="refined_validation_rules.json", mime="application/json")
