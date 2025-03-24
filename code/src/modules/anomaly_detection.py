import json

import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import google.generativeai as genai

chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyBNtXn6NefK6JG6dx-ZOHzuSVhuxu2fD54")


# Load dataset
def load_data():
    uploaded_file = st.file_uploader("📂 Upload dataset for anomaly detection", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None


# Perform anomaly detection using Isolation Forest
def detect_anomalies(df, contamination):
    # Drop non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Debugging Step 1: Check if dataset has enough variation
    if numeric_df.shape[1] < 2:
        st.error(
            "⚠️ Not enough numeric columns for anomaly detection. Please upload a dataset with at least two numerical columns.")
        return df

    # Normalize data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(numeric_df)

    # Train Isolation Forest
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    predictions = model.fit_predict(normalized_data)

    # Add results to dataframe
    df["Anomaly_Score"] = predictions
    df["Anomaly"] = df["Anomaly_Score"].map({1: "Normal", -1: "Anomaly"})

    # Debugging Step 2: Print Anomaly Distribution
    st.write("🔍 Anomaly Distribution:", df["Anomaly"].value_counts())

    return df


# Get explanations from Generative AI
def generate_anomaly_explanations(anomalies):
    if anomalies.empty:
        return []


    # Convert anomalies to JSON
    anomalies_text = anomalies.to_json(orient="records")

    # Construct prompt
    prompt = f"""
    You are a financial data expert. Analyze the following dataset anomalies and provide explanations.
    Explain why each row is considered an anomaly and suggest possible reasons.

    ### Detected Anomalies (JSON format)
    {anomalies_text}

    Return the output in **structured JSON format**:
    - "explanations": A list where each entry contains:
      - "row_no": The row number of the anomaly.
      - "column": The most affected column (or feature).
      - "reason": Explanation why it's an anomaly.
    """

    response = chat_model([HumanMessage(content=prompt)])

    # Parse the AI response
    try:
        parsed_response = json.loads(response.content[7:-3])  # Extract structured JSON
        return parsed_response.get("explanations", [])
    except json.JSONDecodeError:
        st.error("⚠️ Error parsing GenAI response. Skipping explanations.")
        return []

# Streamlit UI
def main():
    st.title("📉 AI-Powered Anomaly Detection")

    st.markdown("""
    ### 🔎 Detect Outliers in Your Dataset  
    This module applies **Machine Learning (ML) techniques** to detect anomalies in financial data.

    **How It Works:**
    1️⃣ Upload your **dataset (CSV)**.  
    2️⃣ Choose an anomaly detection model (Isolation Forest, LOF, DBSCAN, etc.).  
    3️⃣ AI highlights **unusual data points (outliers)**.  
    4️⃣ View **interactive visualizations** and download anomaly reports.  

    ### 🚀 Why Use This?
    ✅ **Identifies inconsistencies & fraud detection**.  
    ✅ **Enhances data quality & regulatory reporting**.  
    ✅ **Provides clear anomaly explanations via AI insights**.  
    """)
    df = load_data()

    if df is not None:
        st.write("### Preview of Dataset")
        st.dataframe(df.head())

        contamination = st.slider("Set Contamination (Anomaly Percentage)", 0.005, 0.1, 0.02, 0.005)

        if st.button("Run Anomaly Detection"):
            df = detect_anomalies(df, contamination)

            st.write("### Anomaly Detection Results")
            st.dataframe(df)

            # Check if anomalies exist before plotting
            if "Anomaly" in df.columns and df["Anomaly"].nunique() > 1:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color="Anomaly",
                                     title="Anomalies Detected",
                                     labels={"color": "Anomaly"},
                                     color_discrete_map={"Normal": "blue", "Anomaly": "red"})  # 🔴
                    st.plotly_chart(fig)
                    anomalies = df[df["Anomaly_Score"] == -1]
                    anomaly_explanations = generate_anomaly_explanations(anomalies)

                    # Structured expander for explanations
                    with st.expander("📌 View Anomaly Explanations (Structured)"):
                        for explanation in anomaly_explanations:
                            row_no = explanation.get("row_no", "N/A")
                            column = explanation.get("column", "Unknown Column")
                            reason = explanation.get("reason", "No explanation available.")

                            st.write(f"- **Row {row_no} | Column:** {column}")
                            st.write(f"  - 📝 **Reason:** {reason}")
                            st.markdown("---")  # Separator for readability

                    # Convert explanations to DataFrame for download
                    explanation_df = pd.DataFrame(anomaly_explanations)
                    if not explanation_df.empty:
                        csv_data = explanation_df.to_csv(index=False).encode("utf-8")

                        # Provide CSV download button
                        st.download_button("📥 Download Anomaly Explanations", data=csv_data,
                                           file_name="anomaly_explanations.csv", mime="text/csv")



                else:
                    st.warning("⚠️ Not enough numeric columns to visualize anomalies.")
            else:
                st.warning("⚠️ No anomalies detected. Try adjusting the contamination rate.")

            # Download results
            st.download_button("📥 Download Results", df.to_csv(index=False), "anomaly_results.csv", "text/csv")



