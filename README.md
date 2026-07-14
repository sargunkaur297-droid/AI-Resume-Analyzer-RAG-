# AI-Resume-Analyzer-RAG-
# 📄 AI Resume Analyzer RAG Chatbot

An AI-powered Resume Analyzer built using **Retrieval Augmented Generation (RAG)** that helps users analyze, understand, and improve their resumes. Users can upload a resume PDF and interact with an AI chatbot that provides personalized feedback, skill analysis, and improvement suggestions.

## 🚀 Project Overview

Recruiters often receive hundreds of resumes, making it difficult to identify strengths, weaknesses, and skill gaps. This project uses Generative AI and RAG architecture to analyze resume content and provide context-aware responses.

The system extracts information from a resume, converts it into embeddings, stores them in a vector database, and uses an AI model to answer user queries based on the uploaded document.

## ✨ Features

* 📄 Upload resume in PDF format
* 🔍 Extract and analyze resume content
* 🤖 AI chatbot for resume-related questions
* 🧠 Retrieval Augmented Generation (RAG) pipeline
* 📚 Vector-based document search
* 💡 Resume improvement suggestions
* 🎯 Skill gap identification for job roles
* 💬 Context-aware answers from uploaded resume

## 🛠️ Tech Stack

### Programming Language

* Python

### Frontend

* Streamlit

### Generative AI

* Large Language Model (Mistral/Gemini)
* LangChain

### Document Processing

* PyPDF
* LangChain Document Loaders

### Vector Database

* ChromaDB

### Other Tools

* Git & GitHub
* VS Code
* Python Virtual Environment

## 🧠 How It Works (RAG Pipeline)

```
Resume PDF
    |
    ↓
PDF Text Extraction
    |
    ↓
Text Chunking
    |
    ↓
Embedding Generation
    |
    ↓
Vector Database (ChromaDB)
    |
    ↓
User Query
    |
    ↓
Relevant Context Retrieval
    |
    ↓
LLM Generated Response
```

## 📂 Project Structure

```
AI-Resume-Analyzer-RAG/

│
├── app.py
├── chat.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── utils/
│   ├── pdf_loader.py
│   └── vector_store.py
│
└── documentloaders/
    ├── pdf.py
    ├── test.py
    └── vectorstores/
        ├── DB.py
        └── createdb.py
```

## ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/sargunkaur297-droid/AI-Resume-Analyzer-RAG-.git
```

Navigate to project folder:

```bash
cd AI-Resume-Analyzer-RAG
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
MISTRAL_API_KEY=your_api_key
```

Run the application:

```bash
streamlit run app.py
```

## 💬 Example Questions

Users can ask:

* "What are my strongest technical skills?"
* "Is my resume suitable for an AI Engineer role?"
* "What skills should I add?"
* "Improve my resume summary"
* "What projects should I include?"

## 🔐 Security

* API keys are stored using environment variables.
* Sensitive files like `.env` and vector databases are excluded using `.gitignore`.

## 🔮 Future Improvements

* ATS resume score prediction
* Job description matching
* Resume ranking system
* Interview question generation
* Multiple resume comparison
* Automated resume optimization

## 👩‍💻 Author

**Sargun Kaur**
B.Tech Computer Science Engineering

## 🌐 Live Demo

🔗 [AI Resume Analyzer RAG Chatbot](https://fppxqusmdqguymudgsq9mn.streamlit.app)


