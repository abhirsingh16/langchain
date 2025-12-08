from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embeding-3-large",
    dimensions=48
)

document = [
    "Delhi is capital of India",
    "Mumbai is capital of MH",
    "Kolkata is capital of WB"
]

vector = embedding.embed_documents(document)

print(str(vector))