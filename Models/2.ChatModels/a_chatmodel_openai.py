from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature = 0.2, max_completion_tokens = 10)
# Temerature ranges from 0.0 to 2.0, 0-> deterministic task, 2-> creative task
# max_completion_tokens = 10 -> output will be of 10 tokens

result = model.invoke("What is the capital of India?")

print(result)
print(result.content)