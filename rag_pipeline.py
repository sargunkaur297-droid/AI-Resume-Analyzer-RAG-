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
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SYSTEM_PROMPT = """You are an experienced HR recruiter and resume reviewer.

Analyze only the information contained in the uploaded resume context.
Do not invent qualifications, experience, skills, or facts.

Return these sections:
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

Use concise bullet points. If the context does not contain enough information for a section, say:
"I could not find the answer in the uploaded document."
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
    """Create the embedding model and chat model used by the RAG pipeline."""
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    llm = ChatMistralAI(model="mistral-small-2603", max_retries=0, max_tokens=1200)
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

