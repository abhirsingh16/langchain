from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path=r"D:\LangChain\RAG\Doc_Loader", glob='*.pdf', loader_cls=PyPDFLoader)

docs = loader.load()

print(len(docs))

print(docs[2])