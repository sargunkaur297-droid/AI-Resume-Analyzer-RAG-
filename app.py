import streamlit as st

# ----------------------------
# Page Config (MUST BE FIRST)
# ----------------------------
st.set_page_config(
    page_title="📚 AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

from dotenv import load_dotenv
import tempfile
import os
import shutil

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

load_dotenv()

st.markdown("""
<style>

/* Background */

.stApp{
background:linear-gradient(
135deg,
#eef2f7,
#dbeafe,
#f8fafc
);
}

/* Title */

h1{
color:#1e3a5f !important;
font-weight:800;
}

/* Headings */

h2,h3{
color:#2563a8 !important;
}

/* Text */

p,label,span{
color:#334155 !important;
}

/* Upload */

[data-testid="stFileUploader"]{

background:white;

border-radius:18px;

padding:20px;

border:1px solid #bfdbfe;

box-shadow:
0 8px 25px rgba(30,64,175,.08);

}

/* Input */

.stTextInput>div>div>input{

background:white;

color:#1e293b;

border-radius:12px;

border:2px solid #93c5fd;

}

/* Button */

.stButton>button{

background:linear-gradient(
90deg,
#3b82f6,
#60a5fa
);

color:white;

border:none;

border-radius:14px;

padding:12px 26px;

font-weight:700;

}

.stButton>button:hover{

background:linear-gradient(
90deg,
#2563eb,
#38bdf8
);

}

/* Cards */

div[data-testid="stMarkdownContainer"]{

background: rgba(255,255,255,0.95);

color: #0f172a !important;

border-radius:18px;

padding:12px;

box-shadow:
0 5px 20px rgba(15,23,42,.06);

}

/* Success */

.stSuccess{

border-radius:15px;

}

hr{

border-color:#cbd5e1;

}
.stMarkdown{
    color:#0f172a !important;
}

.stMarkdown p,
.stMarkdown li,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4{
    color:#0f172a !important;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Resume Analyzer")

st.write(
    "Upload your resume and get AI-powered analysis."
)
# ----------------------------
# Initialize Models
# ----------------------------

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

llm = ChatMistralAI(
    model="mistral-small-latest"
)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("📂 Upload Resume")

uploaded_file = st.sidebar.file_uploader(
    "Upload Resume",
    type=["pdf"]
)
if uploaded_file:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp_resume.pdf")

    st.session_state.uploaded_docs = loader.load()

    st.success("Resume uploaded successfully ✅")
# ----------------------------
# Session State
# ----------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ----------------------------
# Process Uploaded PDF
# ----------------------------

if uploaded_file is not None:

    with st.spinner("Reading Resume..."):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        import uuid

        # Create a unique database for every upload
        db_path = f"chroma_db_{uuid.uuid4().hex}"

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=db_path
        )

        st.session_state.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        os.remove(pdf_path)

        st.sidebar.success("✅ Resume indexed successfully!")

# Stop until a resume is uploaded
if st.session_state.retriever is None:
    st.info("📄 Upload your resume to begin.")
    st.stop()
    # ----------------------------
# Prompt Template
# ----------------------------

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an experienced HR recruiter.

Analyze ONLY the uploaded resume.

For each section, provide detailed content using ONLY the resume.

Return exactly these headings:

# Resume Summary
# Technical Skills
# Soft Skills
# ATS Score
# Missing Skills
# Suitable Job Roles
# Resume Improvement Suggestions
# HR Interview Questions
# Technical Interview Questions
# Project-Based Interview Questions

Write bullet points under every heading.

Do not leave any heading blank.

If information is unavailable, write:
"I could not find the answer in the uploaded document."

Do not invent information.
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)
if st.button("🚀 Analyze Resume"):

    uploaded_docs = st.session_state.get("uploaded_docs", [])

    if not uploaded_docs:
        st.warning("Please upload a resume first.")

    else:

        context = "\n\n".join(
            [doc.page_content for doc in uploaded_docs]
        )

        question = """
Analyze this resume.

Return:

1. Resume Summary
2. Technical Skills
3. Soft Skills
4. ATS Score
5. Missing Skills
6. Suitable Job Roles
7. Resume Improvement Suggestions
8. Five HR Interview Questions
9. Five Technical Interview Questions
10. Three Project-Based Interview Questions
"""

        prompt = template.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(prompt)

        st.subheader("📊 Resume Analysis")

        st.markdown(response.content)


                

                # ----------------------------
# Sidebar Options
# ----------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Options")

# Clear Chat
if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.clear()

    st.success("Chat cleared.")

    st.rerun()

# Clear Database
if st.sidebar.button("🧹 Clear Database"):

    import glob
    import shutil

    try:

        folders = glob.glob("chroma_db*")

        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)

        st.session_state.retriever = None

        st.success("✅ Database cleared successfully.")

        st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
# ----------------------------
# Sidebar Information
# ----------------------------

st.sidebar.markdown("---")

st.sidebar.info(
"""
## 🤖 AI Resume Analyzer

### Tech Stack

- **LLM:** Mistral Small Latest
- **Embeddings:** Mistral Embed
- **Vector Database:** ChromaDB
- **Framework:** Streamlit
- **Retriever:** MMR Search

Upload a resume and receive AI-powered resume analysis based only on the uploaded document.
"""
)