from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documentloaders/Sargun_Kaur_Updated_Resume.pdf")

docs = loader.load()

print(docs[1])