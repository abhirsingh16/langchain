from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = r"D:\LangChain\RAG\Doc_Loader\abi_credentials.csv")

docs = loader.load()

print(docs)