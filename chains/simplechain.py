from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map = 'auto'
)

gen_pipe = pipeline(
    model = model,
    task = "text-generation",
    tokenizer = tokenizer,
    max_new_tokens = 256,
    temperature = 0.2
)

llm = HuggingFacePipeline(pipeline = gen_pipe)

chat_model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant" ),
    ("user", "Generate 5 facts about the {topic}")
])

# message = prompt.formart_messages(topic="Arsenal Football Club")

chain = prompt | chat_model | parser

result = chain.invoke({"topic": "Arsenal Football Club"})

print(result)