from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
from transformers import pipeline

import streamlit as st
import os

os.environ["HF_HOME"]="D:/Huggingface_cache"

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens =100
    )

)

model = ChatHuggingFace(llm=llm)


user_input = st.text_input("Enter your Prompt")

st.header("Research Tool")

if st.button("Summarize"):
    result=model.invoke(user_input)
    st.write(result.content)

# result = model.invoke("You are helpful assistant. Tell me What is the population of mumbai?")
# print(result.content)

