from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma  import Chroma
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
embedding_model = MistralAIEmbeddings()
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult" :0.5

    }
)
llm=ChatMistralAI(model = "mistral-small-latest")
template=ChatPromptTemplate.from_messages(
    [("system", """
You are an experienced HR recruiter.

Analyze the uploaded resume using only the provided context.

Return:
1. Resume Summary
2. Technical Skills
3. Soft Skills
4. ATS Score (out of 100)
5. Missing Skills
6. Suitable Job Roles
7. Resume Improvement Suggestions

If the information is not present in the resume, say:
"I could not find that information in the uploaded resume."

Do not make up information.
"""),
     ("human",
      """Context:
{context}
Question:
{question}
""")
     
    ])
print("AI Resume Analyzer Ready")
print("Press 0 to exit")
while True:
    query = """
Analyze this resume.

Return:
1. Resume Summary
2. Technical Skills
3. Soft Skills
4. ATS Score
5. Missing Skills
6. Suitable Job Roles
7. Resume Improvement Suggestions
"""
    
    if query == "0":
     break
    
    docs = retriever.invoke(query)
    context = "\n\n".join(
    [doc.page_content for doc in docs]
)
       
    final_prompt = template.invoke({
    "context":context,
    "question":query
})
    response = llm.invoke(final_prompt)
    print( f"AI:  {response.content}" )