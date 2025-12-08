from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

load_dotenv()


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


print(f"Loading model: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='auto',
    trust_remote_code = True
)

gen_pipeline=pipeline(
    task="text-generation",
    model = model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.5
)

llm=HuggingFacePipeline(pipeline=gen_pipeline)

chat_llm = ChatHuggingFace(llm=llm)

while True:
    user_input=input("user: ")
    if user_input=="exit":
        break
    result=chat_llm.invoke(user_input)
    print("AI: ", result)