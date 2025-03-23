import asyncio
from mcp import StdioServerParameters
from mcpclient import MCPClient
from ollama_toolmanager import OllamaToolManager
from agent import OllamaAgent

async def main():
    # Git server configuration
    git_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-git", "--repository", "/Users/mihir.deshpande/Desktop/repos/ollama-mcp"],
        env=None
    )

    agent = OllamaAgent("qwen2.5", OllamaToolManager())


    async with MCPClient(git_server_params) as mcpclient:
        _ ,tools_list = await mcpclient.get_available_tools()

        for tool in tools_list:
            agent.tool_manager.register_tool(
                name=tool.name,
                function=mcpclient.call_tool, # Passing the function reference here
                description=tool.description,
                inputSchema=tool.inputSchema
            )


        while True:
            try:
                user_prompt = input("\nWelcome to Git assistant, what can I help you with?\n")
                if user_prompt.lower() in ['quit', 'exit', 'q']:
                    break
                res = await agent.get_response(user_prompt)
                print("\nResponse:\n", res)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError occurred: {e}")
                break


if __name__ == "__main__":
    asyncio.run(main())
