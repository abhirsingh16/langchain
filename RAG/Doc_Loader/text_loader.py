from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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
    max_new_tokens = 400,
    temperature = 0.5
)

llm = HuggingFacePipeline(pipeline = pipe)

chat_model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

loader = TextLoader(r"D:\LangChain\RAG\Doc_Loader\poem.txt", encoding="utf-8")

docs = loader.load()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a cricket expert"),
    ("user", "write a summary on the following {poem}")
])

chain = prompt | chat_model | parser

print(chain.invoke({'poem': docs[0].page_content}))

