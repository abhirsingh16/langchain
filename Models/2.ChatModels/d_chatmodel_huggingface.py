from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
import os

os.environ["HF_HOME"]="D:/Huggingface_cache"


llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs=dict(
        temperature=0.2,
        max_new_tokens=128,
    )
    
)
model = ChatHuggingFace(llm = llm )

result = model.invoke("What is the capital of India?")

print(result)
print(result.content)