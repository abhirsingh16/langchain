from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='auto',
    trust_remote_code=True
)

gen_pipeline = pipeline(
    task="text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens=300,
    temperature = 0.9
)


llm = HuggingFacePipeline(pipeline=gen_pipeline)

chat_model = ChatHuggingFace(llm = llm)

chat_template = ChatPromptTemplate([
    ('system','You are a helpful {domain} expert'),
    ('human','Explain me in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Virat Kolhi'})

print(prompt)

response = chat_model.invoke(prompt)

print("AI : ", response.content) 