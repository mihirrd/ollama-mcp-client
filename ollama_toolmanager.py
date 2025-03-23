from typing import Any, Dict, List, Callable
from dataclasses import dataclass

@dataclass
class OllamaTool:
    name: str
    function: Callable
    description: str
    properties: Dict[str, Any]
    required: list[str]


class OllamaToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, function:Callable, description: str, inputSchema: Dict[str, Any]):
        """
        Register a function as a tool.
        """
        properties = inputSchema['properties']
        required = inputSchema['required']
        tool = OllamaTool(name, function, description, properties, required)
        self.tools[name] = tool

    def get_tools(self) -> Dict[str, List[Dict]]:
        """
        Generate the tools specification.
        """
        tool_specs = []
        for name, tool in self.tools.items():
            tool_specs.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': tool.description,
                    'properties': tool.properties,
                    'required': tool.required
                }
            })
        return tool_specs

    async def execute_tool(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool based on the agent's request, handling name translation
        """
        function = payload["function"]
        name = function.name
        tool_input = function.arguments

        # TODO: remove the hard coded repo_path. This has been added as the
        # local model is not always adding it in the args. Also design a way
        # to pick the repo path dynamically.
        if "repo_path" not in tool_input.keys():
            tool_input.update({'repo_path' : "/Users/mihir.deshpande/Desktop/repos/ollama-mcp"})
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        try:
            tool_func = self.tools[name].function
            print("\nTool = \n", name)
            print("\nTool input = \n", tool_input)
            result = await tool_func(name, tool_input)
            return result
        except Exception as e:
            return {
                'tool': name,
                'content': [{
                    'text': f"Error executing tool: {str(e)}"
                }],
                'status': 'error'
            }

    def clear_tools(self):
        """Clear all registered tools"""
        self.tools.clear()
