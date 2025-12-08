from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

import streamlit as st

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=200
    )
)

model=ChatHuggingFace(llm=llm)

st.header("Research Tool")
paper_input = st.selectbox("Select Research Paper Name",[
    'Attention is all you need',
    'Bert:Pretraining of deep bidirectional',
    'Diffusion models beat Gans on Image synthesis'
])
style_input=st.selectbox("Select Explaination Style",[
    'Beginner Friendly',
    'Techincal',
    'Code Oriented',
    'Mathematical'
])
length_input = st.selectbox("Select Explaination Length",[
    'short(1-2 paragraphs)',
    'medium(3-5 paragraphs)',
    'long(detailed)'
])

template = PromptTemplate ( template = """ Please summarize the research paper titled "{paper_input}"
        with following specifications:
        Explaination style {style_input}
        Explaination Length {length_input}

        1. mathematical details:
        Include relevant mathematical equations if present in the paper
        Explain mathematical concept using simple intutive code snippets where applicable

        2. Analogies:
        Use relatible analogies to simplyfy complex ideas
        if certain information is not available in the paper respond with "Insufficient information"
        instead of gussing
        Ensure summary is clear accurate and aligned with provided style and strength """,
        
        input_variables=['paper_input','style_input','length_input']
        
        )


if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input':style_input,
        'length_input': length_input
    })
    st.write(result.content)