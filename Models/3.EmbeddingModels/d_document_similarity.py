from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

embeddings = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is a cricketer",
    "Ronaldo and Messi are footballers",
    "Roger Fedrerer is a tennis player",
    "Alonso is a F1 driver",
    "Sania Nehwal is a badminton player"
]

query = "Tell me about sania"

doc_embeddings = embeddings.embed_documents(documents)
query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)

for idx, val in enumerate(scores[0]):
    print(idx , val)

index,score = sorted(list(enumerate(scores)), key=lambda x:x[1])

print(documents[index])

# print("similarity cosine score: ",score)