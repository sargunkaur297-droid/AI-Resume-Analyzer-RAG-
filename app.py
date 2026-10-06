import glob
import hashlib
import os
import shutil

import streamlit as st

from rag_pipeline import analyze_resume, build_retriever, create_models

st.set_page_config(page_title="AI Resume Analyzer", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #eef2f7, #dbeafe, #f8fafc); }
h1 { color: #1e3a5f !important; font-weight: 800; }
h2, h3 { color: #2563a8 !important; }
p, label, span { color: #334155 !important; }
[data-testid="stFileUploader"] { background:white; border-radius:18px; padding:20px; border:1px solid #bfdbfe; }
.stButton > button { background:linear-gradient(90deg,#3b82f6,#60a5fa); color:white; border:none; border-radius:14px; padding:12px 26px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Resume Analyzer")
st.write("Upload a resume PDF and get grounded, AI-powered feedback using Retrieval Augmented Generation (RAG).")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "db_path" not in st.session_state:
    st.session_state.db_path = None
if "models" not in st.session_state:
    st.session_state.models = create_models()
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

embeddings, llm = st.session_state.models

st.sidebar.title("📂 Resume")
uploaded_file = st.sidebar.file_uploader(
    "Upload Resume", type=["pdf"], help="Upload a PDF resume to start the analysis."
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    if current_hash != st.session_state.file_hash:
        with st.spinner("Reading and indexing your resume..."):
            retriever, db_path = build_retriever(uploaded_file, embeddings)
            st.session_state.retriever = retriever
            st.session_state.db_path = db_path
            st.session_state.file_hash = current_hash
        st.sidebar.success("✅ Resume indexed successfully!")
    else:
        st.sidebar.success("✅ Resume ready")

if st.session_state.retriever is None:
    st.info("📄 Upload your resume from the sidebar to begin.")
    st.stop()

st.subheader("🔎 Ask about your resume")
question = st.text_area(
    "Question",
    value=("Analyze this resume and provide: resume summary, technical skills, soft skills, "
           "ATS score, missing skills, suitable job roles, resume improvement suggestions, "
           "five HR interview questions, five technical interview questions, and three "
           "project-based interview questions."),
    height=140,
)

if st.button("🚀 Analyze Resume"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        response = None
        with st.spinner("Analyzing your resume..."):
            try:
                response = analyze_resume(st.session_state.retriever, llm, question.strip())
            except Exception as exc:
                if getattr(exc, "response", None) is not None and exc.response.status_code == 429:
                    st.error("⚠️ Mistral is temporarily rate-limited. Please wait a little and try again.")
                else:
                    st.error("⚠️ The AI service could not complete the analysis. Please try again.")
        if response:
            st.subheader("📊 Resume Analysis")
            st.markdown(response)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Options")

if st.sidebar.button("🗑️ Clear Session"):
    db_path = st.session_state.get("db_path")
    if db_path and os.path.isdir(db_path):
        shutil.rmtree(db_path)
    st.session_state.retriever = None
    st.session_state.db_path = None
    st.rerun()

if st.sidebar.button("🧹 Clear All Databases"):
    for folder in glob.glob("chroma_db_*"):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    st.session_state.retriever = None
    st.session_state.db_path = None
    st.success("Database cache cleared.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("""
**🤖 AI Resume Analyzer**

**Architecture**
- Streamlit frontend
- Python RAG backend
- Mistral LLM + embeddings
- ChromaDB vector store
- MMR retrieval
- PyPDF document extraction

The model is instructed to ground answers in retrieved resume content.
""")
