from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.runnables import RunnableParallel, RunnableLambda ,RunnablePassthrough,RunnableSequence, RunnableBranch

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

prompt1 = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant"),
    ("user","Give me a detailed report on {topic}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("user","Given me 5 point summary on the following text \n {text}")
])


report_chain = RunnableSequence(prompt1 , chat_model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300 , prompt2 | chat_model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, branch_chain)

print(final_chain.invoke({'topic': 'AI'}))