from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

document = [
    "Delhi is capital of India",
    "Mumbai is capital of MH",
    "Kolkata is capital of WB"
]

vector = embedding.embed_documents(document)

print(str(vector))
