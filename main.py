import asyncio
from mcp import StdioServerParameters
from mcpclient import MCPClient
from ollama_toolmanager import OllamaToolManager
from agent import OllamaAgent

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner

async def main():
    # Git server configuration
    git_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-git", "--repository", "."],
        env=None
    )
    console = Console()

    # Update model in OllamaAgent
    # List of local models supporting tool usage: https://ollama.com/search?c=tools
    agent = OllamaAgent("llama3.2:1b", OllamaToolManager())

    print("Fetching available tools from the MCP server")
    async with MCPClient(git_server_params) as mcpclient:
        _ ,tools_list = await mcpclient.get_available_tools()
        console.clear()
        console.print(Panel.fit("🚀 Welcome to Ollama MCP Client 🚀", padding=(1, 4)))
        console.status("Registering Tools", spinner="dots")
        for tool in tools_list:
            agent.tool_manager.register_tool(
                name=tool.name,
                function=mcpclient.call_tool, # Passing the function reference here
                description=tool.description,
                inputSchema=tool.inputSchema
            )

        while True:
            try:
                print("-" * 40)
                user_prompt = input("How can I help you?\n")
                print("-" * 40)
                if user_prompt.lower() in ['quit', 'exit', 'q']:
                    break
                print()
                with console.status("[bold green]Finding the right tool for the job...[/bold green]", spinner="dots"):
                    res = await agent.get_response(user_prompt)
                    console.print("\n[bold magenta]Result:[/bold magenta]")
                    console.print(Panel.fit(result, style="green"))

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
