from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer,pipeline,AutoModelForCausalLM

from dotenv import load_dotenv
import os

os.environ["HF_HOME"] = "D:/HuggingFace_models"

load_dotenv()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map = 'auto',
    trust_remote_code = True,
)

gen_pipe = pipeline(
    model = model,
    tokenizer = tokenizer,
    task = "text-generation",
    temperature = 0.5,
    max_new_tokens = 400
)

llm = HuggingFacePipeline(pipeline=gen_pipe)

chat_model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variable = ['topic']
)

template2 = PromptTemplate(
    template = "Write a 5 line summary on the following text. \n {text}",
    input_variable = ['text']
)

prompt1 = template1.invoke({'topic':'Quantum Physics'})

result = chat_model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = chat_model.invoke(prompt2)

print(result1.content)