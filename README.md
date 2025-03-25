# Ollama MCP (Model Control Protocol)

Ollama MCP is a tool for connecting Ollama-based language models with external tools and services using the Model Control Protocol (MCP). This integration enables LLMs to interact with various systems like Git repositories, shell commands, and other tool-enabled services.

## Features

- Seamless integration between Ollama language models and MCP servers
- Support for Git operations through MCP Git server
- Extensible tool management system
- Interactive command-line assistant interface

## Installation

1. Ensure you have Python 3.13+ installed
2. Clone this repository
3. Install dependencies:

```bash
# Create a virtual environment
uv add ruff check
# Activate the virtual environment
source .venv/bin/activate
# Install the package in development mode
uv pip install -e .
```

## Usage

### Running the Git Assistant

###!NOTE: 
Befoe you run the project, replace the repo path in 
execute_tool fn in ollama_tool_manager.py with your repository path. 
I know it's not ideal but the LLM just won't get the repo path by itself. 

```bash
uv run main.py
```

### To run tests
```bash
pytest -xvs tests/test_ollama_toolmanager.py
```

This will start an interactive CLI where you can ask the assistant to perform Git operations.

### Extending with Custom Tools

You can extend the system by:

1. Creating new tool wrappers
2. Registering them with the `OllamaToolManager`
3. Connecting to different MCP servers

## Components

- **OllamaToolManager**: Manages tool registrations and execution
- **MCPClient**: Handles communication with MCP servers
- **OllamaAgent**: Orchestrates Ollama LLM and tool usage

## Examples

```python
# Creating a Git-enabled agent
git_params = StdioServerParameters(
    command="uvx",
    args=["mcp-server-git", "--repository", "/path/to/repo"],
    env=None
)

# Connect and register tools
async with MCPClient(git_params) as client:
    # Register tools with the agent
    # Use the agent for Git operations
```

## Requirements

- Python 3.13+
- MCP 1.5.0+
- Ollama 0.4.7+

