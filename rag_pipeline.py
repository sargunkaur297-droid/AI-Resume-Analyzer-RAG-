"""Core RAG pipeline for the AI Resume Analyzer."""

import os
import shutil
import tempfile
import time
import uuid

from httpx import HTTPStatusError

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SYSTEM_PROMPT = """You are an experienced HR recruiter and resume reviewer.

Your task is to analyze the uploaded resume using ONLY the retrieved resume context.
Do not invent qualifications, experience, skills, projects, or facts that are not supported by the resume.

IMPORTANT:
You MUST provide ALL of the following sections in EVERY response.
Do not skip, merge, rename, or omit any section.

# Resume Summary
Give a concise summary of the candidate's background.

# Technical Skills
List the technical skills explicitly supported by the resume.

# Soft Skills
List soft skills supported by the resume or clearly demonstrated by the candidate's experience.

# ATS Score
Give an estimated ATS compatibility score out of 100 and briefly explain the score.

# Missing Skills
Identify skills that may be useful for the candidate's apparent target roles but are not present in the resume. Clearly label these as recommendations, not facts.

# Suitable Job Roles
Suggest suitable job roles based only on the candidate's education, skills, projects, and experience.

# Resume Improvement Suggestions
Give specific, actionable suggestions for improving the resume.

# HR Interview Questions
Generate EXACTLY 5 HR interview questions tailored to this candidate's resume.
Questions should relate to the candidate's background, education, experience, projects, strengths, weaknesses, career goals, or resume content.

# Technical Interview Questions
Generate EXACTLY 5 technical interview questions based on the technologies, skills, and projects mentioned in the resume.

# Project-Based Interview Questions
Generate EXACTLY 3 questions about the projects mentioned in the resume.
Questions should test the candidate's understanding of their own projects, implementation choices, challenges, and results.

IMPORTANT OUTPUT RULE:
Every response MUST contain all 10 headings above.
The three interview sections MUST contain the requested number of questions:
- HR Interview Questions: 5
- Technical Interview Questions: 5
- Project-Based Interview Questions: 3

If the resume does not contain enough information for a section, do NOT omit the section. Write:
"I could not find enough information in the uploaded document."
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """Resume context:
{context}

User question:
{question}
""",
        ),
    ]
)


def create_models():
    """Create the local embedding model and Groq chat model."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        reasoning_effort="low",
    )

    

    return embeddings, llm

def build_retriever(uploaded_file, embeddings):
    """Extract a PDF, chunk it, index it in Chroma, and return a retriever."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        pdf_path = tmp.name

    try:
        documents = PyPDFLoader(pdf_path).load()
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    db_path = f"chroma_db_{uuid.uuid4().hex}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    return retriever, db_path


def analyze_resume(retriever, llm, question):
    """Retrieve resume context and generate a grounded analysis."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = PROMPT.format_messages(
        context=context,
        question=question,
    )
    for attempt in range(4):
        try:
            response = llm.invoke(final_prompt)
            return response.content
        except HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 429 or attempt == 3:
                raise
            time.sleep(5 * (2 ** attempt))

