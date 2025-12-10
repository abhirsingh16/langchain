from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.tools import tool
import requests

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


# Tool Create

@tool
def Multiply(a: int, b:int )-> int:
    """Mutiplication of two number"""
    return a*b

print(Multiply.invoke({"a":4, "b":10}))

print(Multiply.name)
print(Multiply.description)
print(Multiply.args)

tools = [Multiply]

# Tool binding

chat_model.invoke("Hi")

chat_model_with_tools = chat_model.bind_tools(tools)

chat_model.invoke("Hi, How are you?")

query = HumanMessage("can you multiply 3 * 1000")

message = [query]

result = chat_model_with_tools.invoke(message)

message.append(result)

tool_result = Multiply.invoke(result.tool_calls[0])

print(tool_result)

message.append(tool_result)

print(message)

chat_model_with_tools.invoke(message.content)