import os
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

LANG_OPTIONS = ["Auto-Detect", "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Gujarati", "Kannada"]

@st.cache_resource
def get_models():
    """Initializes LLM and Embeddings with  SSL Bypass"""
    client = httpx.Client(verify=False, timeout=60.0)
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in", 
        model="azure/genailab-maas-gpt-4o-mini", 
        api_key=API_KEY, 
        http_client=client, 
        temperature=0.1, 
        streaming=True
    )
    embeddings = OpenAIEmbeddings(
        base_url="https://genailab.tcs.in", 
        model="azure/genailab-maas-text-embedding-3-large", 
        api_key=API_KEY, 
        http_client=client, 
        check_embedding_ctx_length=False
    )
    return llm, embeddings

# --- 2. HELPERS ---
def get_download_link(file_path, file_name):
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode()
            return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" style="color: #1e88e5; text-decoration: none; font-weight: bold;">🔗 {file_name}</a>'
    except:
        return ""
    return ""

def sanitize_for_guardrails(text):
    replacements = {
        r"\bdoctor\b": "registered official",
        r"\bmedical\b": "health-department",
        r"\bhospital\b": "health facility",
        r"\bmale\b": "individual (M)", 
        r"\bfemale\b": "individual (F)", 
        r"\bsex\b": "gender category"
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def load_faqs(filename):
    faq_dict = {"Select a Question": ""}
    file_path = os.path.join(FAQ_PATH, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().split("Question:")
            for item in content:
                if "Answer:" in item:
                    parts = item.split("Answer:")
                    faq_dict[parts[0].strip()] = parts[1].strip()
    return faq_dict

def export_chat_to_pdf(messages, summary_text=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="GovPolicy AI - Chat Transcript", ln=True, align='C')
    
    if summary_text:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, txt="Summary:", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 7, txt=summary_text.encode('latin-1', 'ignore').decode('latin-1'))

    pdf.ln(10)
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, txt=f"{role}:", ln=True)
        pdf.set_font("Arial", "", 10)
        clean_text = msg["content"].encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 7, txt=clean_text)
        pdf.ln(2)
    return pdf.output(dest='S').encode('latin-1')

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GovPolicy AI", layout="wide", page_icon="🏛️")
st.title("🏛️ Public Services Multilingual AI Agent")
llm, embeddings = get_models()

if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")
    selected_name = st.selectbox("Department:", list(DEPT_MAP.keys()))
    selected_lang = st.selectbox("Preferred Language:", LANG_OPTIONS, index=1)
    current_dept = DEPT_MAP[selected_name]
    
    st.divider()
    faqs = load_faqs(current_dept["faq"])
    selected_q = st.selectbox("Quick FAQ Dropdown:", list(faqs.keys()))
    if selected_q != "Select a Question":
        st.info(f"*Answer:* {faqs[selected_q]}")

    st.divider()
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.session_state.messages and st.button("📝 Prepare PDF with Summary"):
        full_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        summary_prompt = f"Summarize this conversation regarding {selected_name} in 4 lines in {selected_lang}: {full_history}"
        summary_res = llm.invoke(summary_prompt)
        st.session_state.pdf_ready = export_chat_to_pdf(st.session_state.messages, summary_res.content)
        st.download_button("📥 Download Transcript", st.session_state.pdf_ready, "chat_summary.pdf", "application/pdf")

# --- 4. MAIN AGENT LOGIC ---
index_path = os.path.join(BASE_INDEX_PATH, current_dept["folder"])

if os.path.exists(index_path):
    v_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # Initial Welcome Summary
    if not st.session_state.messages:
        with st.spinner(f"Loading {selected_name} overview..."):
            sum_docs = v_store.similarity_search("General introduction", k=2)
            safe_sum = sanitize_for_guardrails(" ".join([d.page_content for d in sum_docs]))
            summary_content = llm.invoke(f"Provide a 3-bullet point summary of these policies in {selected_lang}: {safe_sum}").content
            st.subheader(f"📋 {selected_name}: Overview")
            st.info(summary_content)

    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                links = [get_download_link(os.path.join(RAW_DATA_PATH, current_dept["folder"], s), s) for s in msg["sources"]]
                if any(links): st.markdown(" | ".join(filter(None, links)), unsafe_allow_html=True)

    # Chat Input
    if query := st.chat_input(f"Ask about {selected_name}..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)

        # --- STEP 1: INTENT CLASSIFICATION ---
        intent_prompt = f"""Identify the intent of this user message: "{query}"
        Respond with ONLY one word: 
        'GREETING' if it is a hello, thanks, or general pleasantry.
        'QUERY' if it is a specific question about {selected_name} procedures/policies."""
        
        intent = llm.invoke(intent_prompt).content.strip().upper()

        with st.chat_message("assistant"):
            # --- STEP 2: ROUTING ---
            if "GREETING" in intent:
                # Direct response for greetings
                response = llm.invoke(f"The user said '{query}'. Give a friendly, helpful greeting as a {selected_name} assistant in {selected_lang}. Do not provide policy details.")
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content, "sources": []})
            
            else:
                # Normal RAG Processing for Queries
                results = v_store.similarity_search(query, k=4)
                safe_context = sanitize_for_guardrails("\n\n".join([d.page_content for d in results]))
                sources = list(set([d.metadata.get('source', 'Document.pdf') for d in results]))

                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are a government administrative expert for {selected_name}. Use context: {safe_context}. Reply in {selected_lang}."),
                    ("user", "{input}")
                ])
                
                try:
                    response = (prompt | llm).invoke({"input": sanitize_for_guardrails(query)})
                    st.markdown(response.content)
                    
                    # Only show links if sources were found and used
                    links = [get_download_link(os.path.join(RAW_DATA_PATH, current_dept["folder"], s), s) for s in sources]
                    links = list(filter(None, links))
                    if links:
                        st.markdown("---")
                        st.markdown("*Resources:* " + " | ".join(links), unsafe_allow_html=True)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.content, "sources": sources})
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.warning("Index not found. Please run the indexer first.")



indexer.py

import os
import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
RAW_DATA_DIR = "raw_data"
OUTPUT_DIR = "dept_indexes"

def get_embeddings():
    # SSL Bypass for TCS GenAI Lab
    client = httpx.Client(verify=False, timeout=60.0)
    return OpenAIEmbeddings(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-text-embedding-3-large", 
        api_key=API_KEY,
        http_client=client,
        check_embedding_ctx_length=False
    )

def process_folders():
    if not API_KEY:
        print("Error: GENAI_API_KEY not found in .env file")
        return

    embeddings = get_embeddings()
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    for dept_name in os.listdir(RAW_DATA_DIR):
        dept_path = os.path.join(RAW_DATA_DIR, dept_name)
        if os.path.isdir(dept_path):
            print(f"📂 Processing: {dept_name}")
            all_docs = []
            for file in os.listdir(dept_path):
                if file.endswith(".pdf"):
                    print(f"  📄 Indexing {file}...")
                    loader = PyPDFLoader(os.path.join(dept_path, file))
                    docs = loader.load()
                    for d in docs: d.metadata["source"] = file 
                    all_docs.extend(docs)
            
            if all_docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(all_docs)
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local(os.path.join(OUTPUT_DIR, dept_name))
                print(f"✅ Saved index to {OUTPUT_DIR}/{dept_name}\n")

if _name_ == "_main_":
    process_folders()