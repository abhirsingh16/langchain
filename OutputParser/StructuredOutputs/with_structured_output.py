from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv
import os

os.environ["HF_HOME"] = "D:/HuggingFace_models"


load_dotenv()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map = 'auto',
    trust_remote_code=True
)

gen_pipeline = pipeline(
    task = "text-generation",
    model = model,
    temperature = 0.5,
    max_new_tokens = 300,
    tokenizer = tokenizer
)


llm = HuggingFacePipeline(pipeline = gen_pipeline)

chat_model = ChatHuggingFace(llm=llm)


# schema
from typing import TypedDict, Annotated

class Review(TypedDict):
    summary: Annotated[str,"A brief summary on the review"]
    sentiment: Annotated[str,"Return the sentiment of the review as negitive or  positive"]

structured_model = chat_model.with_structured_output(Review)
result = structured_model.invoke("""
The hardware specification of the iphone 13 is bad but the model is outdated and lame.
The price is costly as well""")

print(result)
