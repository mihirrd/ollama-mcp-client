import asyncio
import ollama
# from mcp import StdioServerParameters # Moved into main()
from mcpclient import MCPClient
from ollama_toolmanager import OllamaToolManager
from agent import OllamaAgent

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from mcp import StdioServerParameters


def select_model_and_initialize_agent(console: Console):
    """
    Prompts the user to select an Ollama model and initializes the OllamaAgent.
    Returns the initialized agent, or None if selection fails or no models are available.
    """
    try:
        available_models_data = ollama.list()
        available_models = [model['model'] for model in available_models_data['models']]
    except Exception as e:
        console.print(f"[bold red]Error fetching Ollama models: {e}[/bold red]")
        console.print("Please ensure Ollama is running and accessible.")
        return None

    if not available_models:
        console.print("[bold red]No Ollama models found. Please pull a model first.[/bold red]")
        return None

    console.print("\n[bold cyan]Available Ollama Models:[/bold cyan]")
    for model_name in available_models:
        console.print(f"- {model_name}")
    console.print("-" * 40)

    selected_model_name = None
    while True:
        prompt_message = "Please select a model from the list above"
        # In a testing environment, Prompt.ask might behave differently or need mocking.
        # For now, assume it works as expected or will be mocked.
        user_choice = Prompt.ask(prompt_message, console=console).strip()
        if user_choice in available_models:
            selected_model_name = user_choice
            break
        else:
            console.print(f"[prompt.invalid]Invalid model name: '{user_choice}'. Please choose from the list.")


    prompt_message = "Enter repository path, use `pwd` to fetch full path."
    repo_path = Prompt.ask(prompt_message, console=console).strip()

    # Initialize OllamaToolManager here or pass as an argument if it's complex/shared
    tool_manager = OllamaToolManager()
    agent = OllamaAgent(selected_model_name, tool_manager, repo_path)

    git_server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-git", "--repository", repo_path],
        env=None
    )

    return [agent, git_server_params]

async def main():
    console = Console()

    agent, git_server_params = select_model_and_initialize_agent(console)
    if agent is None:
        console.print("[bold red]Agent initialization failed. Exiting.[/bold red]")
        return

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
                    console.print(Panel.fit(res, style="green"))

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
