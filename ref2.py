[9:43 AM, 10/14/2026] Haneesh: import os
import httpx
import re
import base64
import streamlit as st
from fpdf import FPDF
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

# --- 1. SETUP & CONFIG ---
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
BASE_INDEX_PATH = "dept_indexes"
RAW_DATA_PATH = "raw_data"
FAQ_PATH = "faqs"

DEPT_MAP = {
    "Passport Services": {"folder": "passport", "faq": "passport_faq.txt"},
    "Aadhaar (UIDAI)": {"folder": "aadhaar", "faq": "aadhaar_faq.txt"},
    "PAN": {"folder": "pan", "faq": "pan_faq.txt"}
}

LANG_OPTIONS = ["Auto-Detect", "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Gujarati", "…
[10:07 AM, 10/14/2026] Haneesh: import streamlit as st
import pandas as pd
import json
import numpy as np
import re
from textblob import TextBlob
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import ssl

# === Fix SSL if blocked by internal network ===
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="🧠 Intelligent Data Quality Agent", layout="wide")
st.title("🧠 Intelligent Data Quality & Governance Agent")

st.write("""
Upload a *dataset (.csv or .json)* and a *metadata schema (.json)* to automatically:
- Validate data quality (types, nulls, ranges, allowed values)
- Detect schema mismatches
- Analyze unstructured text columns (keywords, sentiment)
- Then *chat* with your data interactively!
""")

# ======================================================
# Step 1️⃣ — Metadata Upload & Validation (Mandatory)
# ======================================================
metadata_file = st.file_uploader("📂 Upload metadata JSON file (mandatory)", type=["json"])

def validate_metadata_structure(metadata):
    """Ensure metadata JSON contains mandatory keys and valid structure."""
    required_keys = {"type", "required"}
    issues = []

    if not isinstance(metadata, dict):
        return ["Metadata root must be a JSON object (dictionary)."]

    for field, props in metadata.items():
        if not isinstance(props, dict):
            issues.append(f"❌ Field '{field}' should contain nested JSON with type/required info.")
            continue

        missing = required_keys - props.keys()
        if missing:
            issues.append(f"❌ Field '{field}' missing required metadata keys: {', '.join(missing)}")

        valid_types = {"integer", "float", "string", "boolean", "datetime", "text"}
        if "type" in props and props["type"].lower() not in valid_types:
            issues.append(f"⚠️ Invalid type '{props['type']}' for field '{field}'. Must be one of: {', '.join(valid_types)}")

        if "required" in props and not isinstance(props["required"], bool):
            issues.append(f"⚠️ 'required' must be true/false for field '{field}'")

        if "allowed_values" in props and not isinstance(props["allowed_values"], list):
            issues.append(f"⚠️ 'allowed_values' should be a list for field '{field}'")

        for bound in ["min", "max"]:
            if bound in props and not isinstance(props[bound], (int, float)):
                issues.append(f"⚠️ '{bound}' should be numeric for field '{field}'")

    return issues

if not metadata_file:
    st.error("🚫 Metadata file is *mandatory*. Please upload a valid JSON metadata file before proceeding.")
    st.stop()

try:
    metadata = json.load(metadata_file)
    metadata_issues = validate_metadata_structure(metadata)
    if metadata_issues:
        st.error("❌ Metadata validation failed:")
        for issue in metadata_issues:
            st.write(f"- {issue}")
        st.info("✅ Example of valid metadata structure:")
        st.json({
            "CustomerID": {"type": "integer", "required": True},
            "Name": {"type": "string", "required": True},
            "Age": {"type": "integer", "required": False, "min": 18, "max": 100},
            "City": {"type": "string", "required": False, "allowed_values": ["Kochi", "Bangalore"]},
            "Feedback": {"type": "text", "required": False}
        })
        st.stop()
    else:
        st.success("✅ Metadata file validated successfully.")
except Exception as e:
    st.error(f"❌ Failed to parse metadata JSON: {e}")
    st.stop()

# ======================================================
# Step 2️⃣ — Data Upload & Validation
# ======================================================
uploaded_file = st.file_uploader("📁 Upload data file (CSV or JSON)", type=["csv", "json"])

def validate_with_metadata(df, metadata):
    schema_issues = []
    for col, props in metadata.items():
        # Check for required column presence
        if props.get("required", False) and col not in df.columns:
            schema_issues.append({"column": col, "issue": "Missing Required Column"})
            continue

        if col not in df.columns:
            continue

        # Check nulls in required
        if props.get("required", False):
            nulls = df[col].isna().sum()
            if nulls > 0:
                schema_issues.append({"column": col, "issue": f"{nulls} Nulls in Required Column"})

        # Allowed values
        if "allowed_values" in props:
            invalids = ~df[col].isin(props["allowed_values"])
            if invalids.sum() > 0:
                schema_issues.append({"column": col, "issue": f"{invalids.sum()} Invalid Values"})

        # Numeric min/max
        if np.issubdtype(df[col].dtype, np.number):
            if "min" in props:
                below_min = (df[col] < props["min"]).sum()
                if below_min:
                    schema_issues.append({"column": col, "issue": f"{below_min} Below Minimum"})
            if "max" in props:
                above_max = (df[col] > props["max"]).sum()
                if above_max:
                    schema_issues.append({"column": col, "issue": f"{above_max} Above Maximum"})
    return schema_issues

def analyze_text_column(series):
    """Perform simple NLP checks like sentiment and keyword extraction."""
    sample_texts = series.dropna().astype(str).head(100)
    sentiments = [TextBlob(t).sentiment.polarity for t in sample_texts]
    common_words = pd.Series(" ".join(sample_texts).split()).value_counts().head(10).index.tolist()
    return {
        "avg_sentiment": np.mean(sentiments),
        "common_words": common_words
    }

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    issues = validate_with_metadata(df, metadata)

    if issues:
        st.subheader("⚠️ Data Quality Issues Found")
        st.dataframe(pd.DataFrame(issues))
    else:
        st.success("✅ No schema or data quality issues found.")

    # NLP Analysis for text columns
    text_columns = [col for col, props in metadata.items() if props["type"] == "text" and col in df.columns]
    if text_columns:
        st.subheader("🧠 NLP Insights (Unstructured Text Fields)")
        for col in text_columns:
            result = analyze_text_column(df[col])
            st.write(f"*Column:* {col}")
            st.write(f"- Average Sentiment: {result['avg_sentiment']:.2f}")
            st.write(f"- Top Keywords: {', '.join(result['common_words'])}")

    # ======================================================
    # Step 3️⃣ — Interactive Chat with Data
    # ======================================================
    st.subheader("💬 Ask questions about your data")

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if "qa_chain" not in st.session_state:
        with st.spinner("Initializing AI Agent..."):
            text_content = df.to_csv(index=False)
            embeddings = OpenAIEmbeddings()
            vectordb = FAISS.from_texts([text_content], embeddings)
            retriever = vectordb.as_retriever()
            llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=st.session_state.memory
            )

    query = st.text_input("🔍 Ask your question (e.g., 'List all cities', 'What is average age?')")

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.run(query)
            st.write(f"🤖 *Blitzkrieg Agent:* {response}")