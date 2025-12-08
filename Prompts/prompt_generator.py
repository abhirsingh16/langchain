from langchain_core.prompts  import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv


load_dotenv()

template = PromptTemplate(template = """Can you tell me 
                           the capital of India?{text}""",
                           input_vaiables = ["text"])

template.save('template.json')
