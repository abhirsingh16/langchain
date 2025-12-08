from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer

from typing import Optional
import os

os.environ["HF_HOME"] = "D:/HuggingFace_models"


load_dotenv()

MODEL_ID =  "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map ='auto',
    trust_remote_code=True
    
)

gen_pipeline = pipeline(
    task = "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens=200,
    temperature = 0.5
)


llm = HuggingFacePipeline(pipeline=gen_pipeline)

chat_model = ChatHuggingFace(llm=llm)


class Student(BaseModel):
    name:str="Abhishek"
    age:Optional[int]=None
    email : EmailStr
    cgpa : float=Field(ge = 0, le=10, default = 5, description = 'A decimal value representation of the cgpa of students')


new_student={'age':'32',"email":"abc@gmail.con"}
student_obj = Student(**new_student)

student_dict = dict(student_obj)

print(student_dict['age'])
student_json = student_obj.model_dump_json()

print(student_json)