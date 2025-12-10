
# Search Tool
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("top news in India today")

print(results)

print(search_tool.name)

print(search_tool.args)

print(search_tool.description)


# Builtin Shell Tool
from langchain_community.tools import ShellTool

shell_tool = ShellTool()

result = shell_tool.invoke('ls')

print(result)


## Custom Tool

from langchain_core.tools import tool

@tool
def multiply(a:int, b:int ) -> int:
    """Multiply two numbers"""
    return a * b


result = multiply.invoke({"a":5, "b":3})
print(result)