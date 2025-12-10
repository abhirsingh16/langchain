from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a : int = Field(required=True, description="The first Number to mutiply")
    b : int = Field(required=True, description="The second Number to mutiply")

def Multiply_func(a : int, b : int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=Multiply_func,
    name = "Multiply",
    description = "Multiply two numbers",
    args_schema = MultiplyInput
)

result = multiply_tool.invoke({"a":3, "b": 10 })

print(result)