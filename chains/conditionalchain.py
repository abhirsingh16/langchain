from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pydantic import BaseModel, Field
from typing import Literal

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map = 'auto',
    dtype = 'auto'
)

pipe = pipeline(
    task = 'text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 256,
    temperature = 0.5
)

llm = HuggingFacePipeline(pipeline = pipe)

chat_model = ChatHuggingFace(llm = llm)

class Feedback(BaseModel):
    sentiment: Literal['positive','negative']= Field(... , description = 'Give the sentiment of the feedback')


parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = ChatPromptTemplate.from_messages([
    ('system','You are a helpful assistant'),
    ('user','Classify the sentiment of the following feedback text into positive or negitive feedback \n {feedback} \n {format_instructions}')
]).partial( 
    format_instructions=parser2.get_format_instructions())


prompt2 = ChatPromptTemplate.from_messages([
    ('user','Write a appropriate response to positive feedback \n {feedback}')
])

prompt3 = ChatPromptTemplate.from_messages([
    ('user','Write a appropriate response to negitive feedback \n {feedback}')
])

classifier_chain = prompt1 | chat_model | parser1


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | chat_model | parser1 ),
    (lambda x: x.sentiment == 'negitive', prompt3 | chat_model | parser1 ),
    RunnableLambda(lambda x: 'could not find the sentiment')
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : "This is a beautiful dress"})

print(result)