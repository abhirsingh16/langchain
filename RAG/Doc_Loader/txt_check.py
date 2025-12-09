from langchain_community.document_loaders import TextLoader

loader = TextLoader(r"D:\LangChain\RAG\Doc_Loader\poem.txt", encoding="utf-8")

docs = loader.load()

print(docs[0])