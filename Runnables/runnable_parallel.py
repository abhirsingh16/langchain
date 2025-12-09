from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableSequence

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
    max_new_tokens = 256,
    temperature = 0.5
)

llm = HuggingFacePipeline(pipeline = pipe)

chat_model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke on {topic}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("user", "Explain the following joke \n {text}")
])

joke_chain = RunnableSequence(prompt1, chat_model, parser)

explain_chain =  RunnableSequence(prompt2, chat_model, parser)

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "explain" :  explain_chain.with_config(run_name='explain')
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

print(final_chain.invoke({'topic':'AI'}))