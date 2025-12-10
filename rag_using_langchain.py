from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS


# Model Building

MODEL_ID =  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map = "auto",
    dtype = "auto"
)

pipe = pipeline(
    model = model,
    task = "text-generation",
    tokenizer = tokenizer,
    max_new_tokens = 256,
    temperature = 0.5
)

llm = HuggingFacePipeline(pipeline = pipe)

chat_model = ChatHuggingFace(llm = llm)



# Step 1.A  Indexing - Document Insertion

video_id = "DB9mjd-65gw"

try:
  transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
  transcript = " ".join(chunk.text for chunk in transcript_list)
  print(transcript)
except TranscriptsDisabled:
  print("No caption available for the Video")



# Step 1.B Indexing - Text Splitter

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.create_documents([transcript])
len(chunks)


# Step 1.C and 1.D - Embedding generation and storing in vectorstore

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.index_to_docstore_id
vector_store.get_by_ids(["----"])


# Step 2  Retrievial

retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":4})
retriever

retriever.invoke("what is morning habbits of a Millionaire?")


# Step 3 Augmentation

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Answer from the provided transcript context. "
     "If the context is insufficient then just say you don't know.\n\nContext:\n{context}"
    ),
    ("user", "Give me an answer to this question:\n{question}")
])

question = "is the topic on related to skills discussed in the video?"

retriever_docs = retriever.invoke(question)

retriever_docs

context_text = "\n\n".join(doc.page_content for doc in retriever_docs)

context_text

final_prompt = prompt.invoke({"context": 'contex_text', "question":"question"})

final_prompt


# Step 4 Generation

answer = chat_model.invoke(final_prompt)
print(answer.content)




# Building a chain for the above use case

from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

def format_docs(retriever_docs):
  context_text = "\n\n".join(docs.page_content for docs in retriever_docs)
  return context_text

parallel_chain = RunnableParallel({
  "context":retriever | RunnableLambda(format_docs),
  "question": RunnablePassthrough()
})

parallel_chain.invoke("who is Altman?")

main_chain = parallel_chain | prompt | chat_model | parser

main_chain.invoke("can you summerize the video?")
