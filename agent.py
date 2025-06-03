import ollama
from ollama_toolmanager import OllamaToolManager

class OllamaAgent:
    def __init__(self,model:str,
                 tool_manager: OllamaToolManager,
                 repo_path: str,
                 default_prompt="You are a helpful assistant who can use available tools to solve problems") -> None:
        self.model = model
        self.default_prompt = default_prompt
        self.messages = []
        self.repo_path = repo_path
        self.tool_manager = tool_manager

    async def get_response(self, content:str):
        self.messages.append({
            'role':'user',
            'content' : content
        })

        query = ollama.chat(
            model=self.model,
            messages=self.messages,
            tools=self.tool_manager.get_tools()
        )

        result = await self.handle_response(query)
        return result

    async def handle_response(self, response):
        try:

            tool_calls = response.message.tool_calls
            self.messages.append({
                'role' : 'tool',
                'content' : str(response)
            })
            if tool_calls:
                tool_payload = tool_calls[0]
                result = await self.tool_manager.execute_tool(tool_payload, self.repo_path)

            tool_response = []
            for content in result.content:
                tool_response.append(content.text)

            return "".join(tool_response)
        except Exception as e:
            print(e)
