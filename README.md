# 📊 AI-Powered Data Profiling & Compliance Assistant

## 📌 Table of Contents
- [🚀 Introduction](#-introduction)
- [💡 Inspiration](#-inspiration)
- [🔍 What It Does](#-what-it-does)
- [🛠️ How I Built It](#-how-I-built-it)
- [⚡ Challenges I Faced](#-challenges-i-faced)
- [📖 How to Use the App](#-how-to-use-the-app)
- [🎥 Demo](#-demo)
- [🏗️ Tech Stack](#-tech-stack)
- [👥 Team](#-team)


## 🚀 Introduction
This project automates **data profiling and regulatory compliance validation** using **Generative AI & Machine Learning**. It extracts **validation rules from regulatory documents**, applies them to datasets, identifies anomalies, and provides an **interactive compliance assistant**.

## 💡 Inspiration
**Financial data validation is complex** due to strict regulatory rules. This project automates rule extraction and dataset validation, reducing **manual effort and improving accuracy.**

## 🔍 What It Does
✅ **Extracts validation rules** from regulatory PDFs using AI.  
✅ **Validates datasets** against extracted rules to detect inconsistencies.  
✅ **Identifies anomalies** using ML techniques like Isolation Forest.  
✅ **Provides an interactive AI-powered chat assistant** for compliance queries.  
✅ **Generates reports** highlighting data violations and anomalies.

---

## 🛠️ How I Built It

### **1️⃣ Rule Extraction (AI-Powered)**
- **Extracts validation rules** from regulatory PDFs using **Google Gemini AI**.
- Uses **ChromaDB** for **semantic search** and rule retrieval.
- Processes documents in **batches** to reduce API costs and improve efficiency.
- **LangChain integration** enables structured rule extraction.

### **2️⃣ Dataset Validation (Batch Processing)**
- Datasets are **validated in batches** to optimize performance.
- Uses **Generative AI (Gemini) to detect violations** based on extracted rules.
- **Ensures JSON response integrity** to prevent parsing failures.
- Tracks **progress dynamically** with Streamlit UI.

### **3️⃣ Anomaly Detection (Unsupervised Machine Learning)**
- Uses **Isolation Forest** for unsupervised anomaly detection.
- Automatically determines the **contamination rate** for anomaly proportion.
- **Visualizes outliers** using **Plotly charts**.
- Anomalies are downloadable as structured reports.

### **4️⃣ Interactive Chat Assistant (Compliance Queries)**
- Uses **FAISS (Vector Search)** to retrieve relevant compliance violations.
- AI-powered responses using **Google Gemini**.
- Supports **rule refinement requests** and **explanations for anomalies**.

---

## ⚡ Challenges I Faced

✅ **Handling Large Regulatory PDFs Efficiently**  
- **Chunked processing & vector search** prevented excessive memory usage.

✅ **Reducing API Usage & Cost Management**  
- **Batch processing** reduced unnecessary API calls.
- Implemented **cache-based retrieval** for rule extraction.

✅ **Ensuring Valid JSON Responses from AI**  
- AI responses sometimes contained **incomplete JSON**.
- Used **error handling, regex fixes**, and **parsing checks**.

✅ **Aligning Rule Extraction with Validation Logic**  
- Ensured extracted rules **matched dataset structures**.
- **Fine-tuned prompts** to improve rule relevance.

✅ **Making UI More Interactive & Engaging**  
- Used **dynamic progress bars, collapsible reports, and color-coded chat bubbles**.

---

## 🏃 How to Run

1️⃣ **Clone this repository**:  
```bash
 git clone https://github.com/your-repo/data-profiling.git
```

2️⃣ **Create the GOOGLE_API_KEY from the Gemini API docs**:  
- 📑 [Gemini API key](https://ai.google.dev/gemini-api/docs)

3️⃣ **Create .streamlit directory with secrets.toml file and paste the API following the below**:  
```
./code/src/.streamlit/secrets.taml 
```
```
[google]
api_key = "YOUR_ACTUAL_API_KEY"
```
4️⃣ **Install dependencies**:  
```bash
 pip install -r requirements.txt
```

5️⃣ **Run the Streamlit app**:  
```bash
 streamlit run app.py
```

6️⃣ **Upload files** and get insights!

---

## 📖 How to Use the App
This AI-powered **Data Profiling & Compliance Assistant** follows a structured **step-by-step workflow**.

### **1️⃣ Extract Validation Rules (AI-Powered)**
- **Upload:** A **regulatory PDF** and a **sample dataset (CSV)**.  
- **AI extracts structured validation rules** from the document.  
- **Download the extracted rules (JSON)** for dataset validation.  

### **2️⃣ Validate Dataset Against Rules**
- **Upload:** The **dataset (CSV)** and the **extracted rules (JSON)** from step 1.
- **validates** each row based on the **regulatory rules**.
- **View validation errors & compliance issues** in a structured format.
- Download detailed **validation reports (CSV)**.

### **3️⃣ Chat with Compliance Assistant**
- Ask queries about **rule violations & compliance issues**.
- **Upload:** The **extracted rules (JSON)** and **validation results (JSON)** from step 1 and 2.
- AI provides **explanations & rule refinement** suggestions.

### #️⃣ **Detect Anomalies in the Dataset**
- **Upload:** The dataset (CSV).
- Uses **Isolation Forest** to detect outliers in financial data.
- Visualizes **anomalies** with interactive graphs.
- Download **flagged anomalies (CSV)** for further review.

## ✐ᝰ. Architecture

  ![data-profiling-arch-diagram](https://github.com/user-attachments/assets/037a1bc4-51d5-47b9-aadf-bbbd92ce1444)

    `

## 🎥 Demo
- 📑 [Part 1 : Extracting Regulation Rules](https://drive.google.com/file/d/1cy0zS77Za4fj_TAqS1BuQTgrN102qgzF/view?usp=sharing)
- 📑 [Part 2 : Validate Dataset For Violations](https://drive.google.com/file/d/1Ve4FH08GHxtqDv9rOTzAFjJ99y7q7Hf7/view?usp=sharing)
- 📑 [Part 3 : Compliance Chat Assistant and Anomaly Detection](https://drive.google.com/file/d/1p2uNhu5UlHvaU26cjABcnVDKVdXdjJZ6/view?usp=drive_link)

## 📸 Screenshots 

## **Rule Extraction (AI-Powered Regulatory Rule Extraction)**

  ![image](https://github.com/user-attachments/assets/2c410a18-90dc-46f8-8f1b-7da471c589c6)
  ![image](https://github.com/user-attachments/assets/29ddeed9-78c0-43f5-8fe3-d9bb9c5d6dd3)

## **Dataset Validation (Automated Compliance Checking)**

  ![image](https://github.com/user-attachments/assets/b78dd3e2-f67d-460b-ad2e-3333ed5f67d3)
  ![image](https://github.com/user-attachments/assets/94d14a03-c6ae-403f-9be5-bf7a9818352f)

## **Compliance Chat Assistant (Interactive AI-Powered Queries)**

  ![image](https://github.com/user-attachments/assets/4d6b9213-85c1-48c6-8c69-db0f9db74bac)
  ![image](https://github.com/user-attachments/assets/bfc6db4d-60bc-4b79-88ff-fc8d9c829e4c)

## **Anomaly Detection (Machine Learning-Based Outlier Detection)**

  ![image](https://github.com/user-attachments/assets/7d40ddae-e296-4589-ad64-51017626cf64)
  ![image](https://github.com/user-attachments/assets/3848708a-09d6-4ef3-8033-147684d1c573)
  ![image](https://github.com/user-attachments/assets/9c121c95-d5d1-4f1d-80c8-2ef5c6e3d525)
  

## 🏗️ Tech Stack

- **Backend:** Python, LangChain, Pandas
- **Frontend:** Streamlit, Plotly, Altair
- **AI Models:** Google Gemini (LLM), Isolation Forest (ML - Z Score, Percentile Cutoff)
- **Database:** ChromaDB, FAISS for Vector Storage

---

## 👥 Team

- **Sharanprasath S** - GEN-AI Implementation, UI/UX: Frontend and Backend Development
- **Links** - [Github](https://github.com/SharanPrasath) | [Linkedin](https://www.linkedin.com/in/sharan-prasath-s-0084b5209/)



