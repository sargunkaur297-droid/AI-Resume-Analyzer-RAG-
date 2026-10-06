# 🤖 AI Resume Analyzer — RAG

An AI-powered resume analysis application built with **Retrieval Augmented Generation (RAG)**. Upload a PDF resume, retrieve the most relevant sections, and receive grounded feedback from a Mistral language model.

## ✨ Features
- 📄 Upload a resume in PDF format
- 🔍 Extract and chunk resume text
- 🧠 Generate Mistral embeddings
- 🗄️ Store chunks in ChromaDB
- 🔎 Retrieve relevant context using MMR search
- 🤖 Generate grounded resume feedback with Mistral
- 🎯 Identify skills, skill gaps, job roles, and improvement areas
- 💬 Ask custom questions about the uploaded resume

## 🏗️ Architecture

This is a **single-repository full application**: the Streamlit interface and Python RAG/backend logic live together in this repository.

```
User → Streamlit UI (app.py)
     → PDF extraction + chunking
     → Mistral embeddings
     → ChromaDB vector store
     → MMR Retriever
     → Relevant resume context
     → Mistral LLM (rag_pipeline.py)
     → Grounded resume analysis
```

## 📁 Project structure
```
AI-Resume-Analyzer-RAG-/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Tech stack
| Layer | Technology |
|---|---|
| UI | Streamlit |
| Language | Python |
| LLM | Mistral Small Latest |
| Embeddings | Mistral Embed |
| RAG framework | LangChain |
| Vector database | ChromaDB |
| PDF processing | PyPDF / PyPDFLoader |
| Retrieval | MMR |
| Environment | python-dotenv |

## 🚀 Run locally

```bash
git clone https://github.com/sargunkaur297-droid/AI-Resume-Analyzer-RAG-.git
cd AI-Resume-Analyzer-RAG-
python -m venv .venv
```

Activate the environment, then install:
```bash
pip install -r requirements.txt
```

Create a `.env` file:
```env
MISTRAL_API_KEY=your_api_key_here
```

Run:
```bash
streamlit run app.py
```

## 💡 Example questions
- What are my strongest technical skills?
- Is my resume suitable for an AI Engineer role?
- What skills are missing for a backend developer role?
- How can I improve my resume?
- What interview questions could be asked about my projects?

## 🔐 Security
- API credentials are loaded from environment variables.
- `.env` is excluded through `.gitignore`.
- Uploaded PDFs are processed as temporary files and removed after extraction.
- Generated Chroma databases are ignored by Git.

## ⚠️ Limitations
- Analysis quality depends on PDF text extraction.
- The application currently supports PDF resumes.
- Responses are grounded in retrieved resume context and may report unavailable information when evidence is not found.

## 👩‍💻 Author
**Sargun Kaur** — B.Tech Computer Science & Engineering

## 🌐 Live demo
[Open the AI Resume Analyzer](https://fppxqusmdqguymudgsq9mn.streamlit.app)
