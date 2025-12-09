from langchain_community.document_loaders import PyPDFLoader

# [
#     Document(page_content = 'text from page 1', metadata = {'page'=0, source = 'arsenal.pdf'}),
#     Document(page_content = 'text from page 2', metadata = {'page'=1, sourc2 = 'arsenal.pdf'}),
# ]

loader = PyPDFLoader('arsenal.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)