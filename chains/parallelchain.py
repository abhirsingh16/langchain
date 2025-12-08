from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map = 'auto'
)

pipe = pipeline(
    task = 'text-generation',
    model = model,
    max_new_tokens = 256,
    tokenizer = tokenizer,
    temperature = 0.9
)

llm = HuggingFacePipeline(pipeline = pipe)

chat_model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = ChatPromptTemplate.from_messages([
    ("system","You are a historian"),
    ("user", "Generate a detailed notes on the {topic}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system","You are a school teacher"),
    ("user", "Generate a small question and answer on the following text\n\n {text}")
])

prompt3 = ChatPromptTemplate.from_messages([
    ("user", "Merge the notes and question and answer in a single document\n\n notes->{notes} and quiz->{quiz}")    
])


detailed_chain = prompt1 | chat_model | parser

qna_chain = prompt2 | chat_model | parser


parallel_chain = RunnableParallel(
    notes = detailed_chain,
    quiz = ({"text": detailed_chain} | qna_chain)
)

merge_chain = prompt3 | chat_model | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"topic":"Shivaji Maharaj of Marathas"})

print(result)