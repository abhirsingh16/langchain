from langchain_core.tools import tool

# custom tools

@tool
def add(a:int, b:int) ->int:
    """Add two numbers"""
    return a + b

@tool
def subtract(a:int, b:int) ->int:
    """Subtract two numbers"""
    return a-b

@tool
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a * b


class MathToolkit:

    def get_tools(self):
        return [add, multiply, subtract]
    

toolkit = MathToolkit()

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name} -> {tool.description}")