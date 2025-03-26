import re

import pandas as pd
import json
import time
import streamlit as st
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

# Retrieve the API key
load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = st.secrets["google"]["api_key"]


# Initialize the GenAI model (Google Gemini)
chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Batch size
batch_size = 10

def validation_load_data():
    st.image("./image/data_validation.png")
    col1, col2 = st.columns(2)

    with col1:
        extracted_rules_file = st.file_uploader("📂 Upload the extracted validation rules JSON", type=["json"])
    with col2:
        dataset_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    if extracted_rules_file is not None and dataset_file is not None:
        validate_dataset(dataset_file, extracted_rules_file)
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
    fixed_json = response_text[:last_pos] + "]" + "}"  # Add an extra `}` just in case

    return fixed_json


def validate_dataset(df, validation_rules):
    df = pd.read_csv(df)

    validation_rules = json.load(validation_rules)
    all_results = {"errors": []}
    error_count = 0
    success_count = 0

    total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    if "validation_results" in st.session_state:
        st.success("✅ Validation already completed. Download your results below.")
        with st.expander("🚨 Validation Errors"):
            errors_df = pd.DataFrame(st.session_state.validation_results["errors"])
            st.dataframe(errors_df)
            csv = errors_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Validation Errors", data=csv, file_name="validation_errors.csv",
                               mime="text/csv")
    else:
        progress_bar = st.progress(0)  # Progress bar starts at 0%

        st.write("🔍 Running validation... This may take a few minutes.")

        # Show processing status
        status_box = st.empty()
        col1, col2 = st.columns(2)
        with col1:
            success_box = st.empty()  # Placeholder for successful batches

        with col2:
            error_box = st.empty()  # Placeholder for failed batches

        # Start batch processing
        for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches", unit="batch"):
            batch_df = df.iloc[i:i + batch_size]
            batch_df["row_no"] = batch_df.index + 1
            dataset_text = batch_df.to_json(orient="records")

            prompt = f"""
            You are an expert in financial regulatory compliance.
            Validate the following dataset based on these validation rules.
    
            ### Dataset (JSON format)
            {dataset_text}
    
            ### Validation Rules (JSON format)
            {json.dumps(validation_rules, indent=2)}
    
            Return the validation results in structured JSON format with:
            - "errors": A list of validation errors  (rule_id, row_no, column_name, severity, error_description (need to be very clear) , suggestion_to_fix (a short suggestion to fix the issue))
            

            ⚠️ **IMPORTANT INSTRUCTIONS:**
                - Generate only new Violation that do not match any error_description and column values if its already generated in the response.
                - Ensure the response is **valid JSON** with **NO extra text** before or after.
                - Do NOT reset row numbers in each batch.
                - If the JSON is **too large**, do NOT send an **incomplete JSON**.
                - Instead, return only the **fully completed** JSON **up to the last valid dictionary entry**.
            """

            try:

                start_time = time.time()
                response = chat([HumanMessage(content=prompt)])
                elapsed_time = time.time() - start_time

                if elapsed_time > 60:
                    status_box.error(f"⚠️ Batch {i}-{i + batch_size} took too long. Skipping.")
                    continue

                    # **Fix potential response truncation issue**
                raw_response = response.content[7:-3].strip()
                if raw_response.startswith("{") and raw_response.endswith("}"):
                    parsed_response = json.loads(raw_response)
                else:
                    raw_response = fix_truncated_json(raw_response)
                    parsed_response = json.loads(raw_response)
                if "errors" in parsed_response and isinstance(parsed_response["errors"], list):
                    # st.write(parsed_response["errors"])
                    all_results["errors"].extend(parsed_response["errors"])


                success_count += 1



                # Update UI dynamically
                status_box.success(f"✅ Processed Batch {i}-{i + batch_size} in {elapsed_time:.2f}s")

            except json.JSONDecodeError:
                error_count += 1
                status_box.error(f"❌ Error parsing response for batch {i}-{i + batch_size}. Skipping.")
            except Exception as e:
                status_box.error(f"❌ Unexpected error in batch {i}-{i + batch_size}: {e}")
            success_box.success(f"**Successful Batches**                             {success_count}")
            error_box.error(f"**Failed Batches**                                     {error_count}")
            progress_bar.progress((i + batch_size) / len(df))
        st.session_state["validation_results"] = all_results
        st.session_state["validation_completed"] = True  # ✅ Mark as completed

        st.subheader("📌 Validation Summary")

        save_dir = "./data/output"
        os.makedirs(save_dir, exist_ok=True)
        # Save validation results
        validation_json_path = os.path.join(save_dir, "extracted_violation_results.json")
        with open(validation_json_path, "w") as f:
            json.dump(all_results, f, indent=4)

        if "validation_results" not in st.session_state:
            st.session_state.validation_results = {"errors": [], "suggestions": []}  # Initialize empty

        st.session_state.validation_results = all_results  # Save resuslts persistently

        # 📌 Display Validation Errors
        if st.session_state.validation_results["errors"]:
        # Display results in expandable sections
            errors_df = pd.DataFrame(all_results["errors"])

            # Apply conditional formatting to highlight errors
            def highlight_errors(val):
                return 'background-color: #ffcccc; color: black;' if isinstance(val, str) else ''

            styled_df = errors_df.style.applymap(highlight_errors)

            # Use `st.dataframe()` with optimized size
            st.dataframe(styled_df, height=600, width=1200)

            # Convert to CSV & Provide Download Option
            csv = errors_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Validation Errors", data=csv, file_name="validation_errors.csv",
                                   mime="text/csv")
        else:
            st.success("🎉 No validation errors found!")

