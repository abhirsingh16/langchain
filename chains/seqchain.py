from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='auto'
)

gen_pipe = pipeline(
    model = model,
    task = 'text-generation',
    tokenizer=tokenizer,
    max_new_tokens = 150,
    temperature = 0.3,
)

llm = HuggingFacePipeline(pipeline = gen_pipe)

chat_model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant"),
    ("user","Give me a detailed report on {topic}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("user","Given me 5 point summary on the following text \n {text}")
])


report_chain = prompt1 | chat_model | parser

summary_chain = prompt2 | chat_model | parser


chain = {"text":report_chain} | summary_chain

result = chain.invoke({"topic":"Unemployment in India"})

print(result)

chain.get_graph().print_ascii()