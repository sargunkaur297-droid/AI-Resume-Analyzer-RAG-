import os
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
docs = [
    Document(page_content="Artificial Intelligence (AI) is a branch of computer science that enables machines to perform tasks that normally require human intelligence. These tasks include learning, reasoning, problem-solving, understanding language, and recognizing images.",metadata={"source":"AI_book"}),
    Document(page_content="Machine Learning is a subset of AI where algorithms learn patterns from data instead of being explicitly programmed. Deep Learning is a further subset that uses neural networks with multiple layers to solve complex problems such as image recognition and natural language processing.",metadata={"source":"DataScience_book"}),
    Document(page_content="AI is widely used in healthcare, finance, education, transportation, and customer support. Popular AI applications include recommendation systems, chatbots, virtual assistants, and self-driving cars.",metadata={"source":"DL_Book"}),
]
embeddings_model = MistralAIEmbeddings()



import uuid

db_path = f"chroma_db_{uuid.uuid4().hex}"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=db_path
)
result = vectorstore.similarity_search("what is used for data analysis?",k=2)
for r in result:
    print(r.page_content)
    print(r.metadata)
    retriver =vectorstore.as_retriever()
    docs=retriver.invoke("Explain deep learning")
    for d in docs:
        print(d.page_content)
