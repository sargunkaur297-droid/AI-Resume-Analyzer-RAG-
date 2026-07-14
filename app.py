import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
st.markdown("""
<style>

/* Overall background */
.stApp {
    background: linear-gradient(
        135deg,
        #f5f3ff 0%,
        #ecfeff 50%,
        #f0fdf4 100%
    );
}

/* Main heading */
h1 {
    color: #312e81;
    font-weight: 800;
    letter-spacing: -1px;
}

/* Subheadings */
h2, h3 {
    color: #4338ca;
}

/* Normal text */
p, label {
    color: #374151;
    font-size: 16px;
}


/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.75);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid #ddd6fe;
    box-shadow: 0px 8px 25px rgba(99,102,241,0.08);
}


/* Chat/input boxes */
.stTextInput > div > div > input {
    background-color: white;
    border-radius: 15px;
    border: 2px solid #c4b5fd;
}


/* Buttons */
.stButton > button {
    background: linear-gradient(
        90deg,
        #6366f1,
        #14b8a6
    );
    color: white;
    border-radius: 15px;
    padding: 12px 25px;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}


/* Button hover */
.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(
        90deg,
        #4f46e5,
        #0d9488
    );
}


/* Cards */
div[data-testid="stMarkdownContainer"] {
    background: rgba(255,255,255,0.55);
    border-radius: 18px;
}


/* Success message */
.stSuccess {
    background-color: #dcfce7;
    border-radius: 15px;
}


/* Warning/error boxes */
.stAlert {
    border-radius: 15px;
}


</style>
""", unsafe_allow_html=True)

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="📚 AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

st.title("🤖 AI Resume Analyzer")
st.write("Upload your resume  and get AI-powered analysis.")

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

st.sidebar.title("📂 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload resume",
    type=["pdf"]
)

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

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
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

        st.sidebar.success("✅ PDF indexed successfully!")

if st.session_state.retriever is None:
    st.info("📄 Upload your resume and click **Analyse Resume**. ")
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

Analyze the uploaded resume.

Return:
1. Resume Summary
2. Technical Skills
3. Soft Skills
4. Missing Skills
5. ATS Score out of 100
6. Suggested Improvements
7. Suitable Job Roles

If the answer is not present in the context, reply exactly:

"I could not find the answer in the uploaded document."

Do not make up answers.
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

# ----------------------------
# Display Chat History
# ----------------------------


# ----------------------------
# Chat Input
# ----------------------------

if st.button("🚀 Analyze Resume"):

    question = """
    Analyze this resume.

    Return:

    1. Resume Summary
    2. Technical Skills
    3. Soft Skills
    4. ATS Score out of 100
    5. Missing Skills
    6. Suitable Job Roles
    7. Resume Improvement Suggestions
    8. Five HR Interview Questions
    9. Five Technical Interview Questions based on the candidate's skills and projects.
    10. Three Project-Based Interview Questions based on the projects mentioned in the resume.

    Use only the information available in the resume. Do not invent projects or experience.
    """
    

    
    with st.spinner("Analyzing resume..."):

        docs = st.session_state.retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = template.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = llm.invoke(prompt)

        answer = response.content
        st.subheader("📊 Resume Analysis")
        st.markdown(answer)
    with st.expander("📄 Retrieved Document Chunks"):

        if len(docs) == 0:
            st.warning("No relevant documents found.")

        else:

            for i, doc in enumerate(docs, start=1):

                st.markdown(f"### 📄 Chunk {i}")

                if doc.metadata:
                    st.write("**Metadata**")
                    st.json(doc.metadata)

                st.write("**Content**")
                st.write(doc.page_content)

                st.divider()

   


# ----------------------------
# Sidebar Options
# ----------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Options")

# Clear Chat
if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.success("Chat history cleared.")

    st.rerun()


# Remove Chroma Database
if st.sidebar.button("🧹 Clear Database"):

    import shutil

    try:

        shutil.rmtree("chroma_db")

        st.session_state.retriever = None

        st.success("Vector database deleted successfully.")

        st.rerun()

    except FileNotFoundError:

        st.warning("Database not found.")


# ----------------------------
# Sidebar Information
# ----------------------------

st.sidebar.markdown("---")
st.sidebar.info(
    """
### 🤖 AI Resume Analyzer

**LLM**
- Mistral Small Latest

**Embedding Model**
- Mistral Embed

**Vector Database**
- ChromaDB

Upload a resume and analyse resume  based only on its contents.
"""
)