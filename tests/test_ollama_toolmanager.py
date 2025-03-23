import pytest
import asyncio
from unittest.mock import MagicMock, patch
from ollama_toolmanager import OllamaToolManager, OllamaTool

# I did not write these tests. Cursor FTW!

# Test functions to register as tools
def add(a: int, b: int) -> int:
    return a + b

async def async_multiply(name: str, args: dict) -> dict:
    a = args.get('a', 0)
    b = args.get('b', 0)
    result = a * b
    return {
        'tool': name,
        'content': [{
            'text': str(result)
        }],
        'status': 'success'
    }


class TestOllamaToolManager:
    
    def setup_method(self):
        self.tool_manager = OllamaToolManager()
        
    def test_init(self):
        assert isinstance(self.tool_manager.tools, dict)
        assert len(self.tool_manager.tools) == 0
    
    def test_register_tool(self):
        # Register a synchronous function
        inputSchema = {
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
        
        self.tool_manager.register_tool(
            name="add_numbers",
            function=add,
            description="Add two numbers",
            inputSchema=inputSchema
        )
        
        assert len(self.tool_manager.tools) == 1
        assert "add_numbers" in self.tool_manager.tools
        
        tool = self.tool_manager.tools["add_numbers"]
        assert isinstance(tool, OllamaTool)
        assert tool.name == "add_numbers"
        assert tool.description == "Add two numbers"
        assert tool.function == add
        assert tool.properties == inputSchema["properties"]
        assert tool.required == inputSchema["required"]
    
    def test_get_tools(self):
        # Register a tool
        inputSchema = {
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
        
        self.tool_manager.register_tool(
            name="add_numbers",
            function=add,
            description="Add two numbers",
            inputSchema=inputSchema
        )
        
        tools_spec = self.tool_manager.get_tools()
        assert len(tools_spec) == 1
        
        tool_spec = tools_spec[0]
        assert tool_spec["type"] == "function"
        assert tool_spec["function"]["name"] == "add_numbers"
        assert tool_spec["function"]["description"] == "Add two numbers"
        assert tool_spec["function"]["properties"] == inputSchema["properties"]
        assert tool_spec["function"]["required"] == inputSchema["required"]
    
    def test_multiple_tools(self):
        # Register multiple tools
        add_schema = {
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
        
        self.tool_manager.register_tool(
            name="add_numbers",
            function=add,
            description="Add two numbers",
            inputSchema=add_schema
        )
        
        # Define a second function
        async def subtract(name: str, args: dict) -> dict:
            a = args.get('a', 0)
            b = args.get('b', 0)
            result = a - b
            return {
                'tool': name,
                'content': [{
                    'text': str(result)
                }],
                'status': 'success'
            }
            
        subtract_schema = {
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
        
        self.tool_manager.register_tool(
            name="subtract_numbers",
            function=subtract,
            description="Subtract b from a",
            inputSchema=subtract_schema
        )
        
        assert len(self.tool_manager.tools) == 2
        assert "add_numbers" in self.tool_manager.tools
        assert "subtract_numbers" in self.tool_manager.tools
        
        tools_spec = self.tool_manager.get_tools()
        assert len(tools_spec) == 2
        
        # Check the names are correct
        tool_names = [t["function"]["name"] for t in tools_spec]
        assert "add_numbers" in tool_names
        assert "subtract_numbers" in tool_names
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        # Register an async function
        multiply_schema = {
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
        
        self.tool_manager.register_tool(
            name="multiply",
            function=async_multiply,
            description="Multiply two numbers asynchronously",
            inputSchema=multiply_schema
        )
        
        # Create a mock payload similar to what would come from the LLM
        # Create a mock function object with attributes
        mock_function = MagicMock()
        mock_function.name = "multiply"
        mock_function.arguments = {"a": 5, "b": 3}
        
        mock_payload = {
            "function": mock_function
        }
        
        result = await self.tool_manager.execute_tool(mock_payload)
        
        assert result["status"] == "success"
        assert result["tool"] == "multiply"
        assert result["content"][0]["text"] == "15"
    
    @pytest.mark.asyncio
    async def test_execute_tool_error_unknown_tool(self):
        # Test with a tool that doesn't exist
        mock_function = MagicMock()
        mock_function.name = "unknown_tool"
        mock_function.arguments = {}
        
        mock_payload = {
            "function": mock_function
        }
        
        with pytest.raises(ValueError, match="Unknown tool: unknown_tool"):
            await self.tool_manager.execute_tool(mock_payload)
    
    @pytest.mark.asyncio
    async def test_execute_tool_error_in_execution(self):
        # Define a function that raises an exception
        async def failing_tool(name: str, args: dict) -> dict:
            raise ValueError("Test error")
        
        # Register the failing tool
        failing_schema = {
            "properties": {
                "x": {"type": "number"}
            },
            "required": ["x"]
        }
        
        self.tool_manager.register_tool(
            name="failing_tool",
            function=failing_tool,
            description="A tool that fails",
            inputSchema=failing_schema
        )
        
        # Create a mock payload
        mock_function = MagicMock()
        mock_function.name = "failing_tool"
        mock_function.arguments = {"x": 1}
        
        mock_payload = {
            "function": mock_function
        }
        
        # Execute should return error status
        result = await self.tool_manager.execute_tool(mock_payload)
        
        assert result["status"] == "error"
        assert "Error executing tool" in result["content"][0]["text"]
    
    @pytest.mark.asyncio
    @patch('builtins.print')  # Mock print to avoid console output during tests
    async def test_execute_tool_with_repo_path_added(self, mock_print):
        # Test that repo_path is added if not present
        async def git_tool(name: str, args: dict) -> dict:
            assert "repo_path" in args
            return {
                'tool': name,
                'content': [{
                    'text': "Git tool executed"
                }],
                'status': 'success'
            }
        
        git_schema = {
            "properties": {
                "branch": {"type": "string"}
            },
            "required": ["branch"]
        }
        
        self.tool_manager.register_tool(
            name="git_checkout",
            function=git_tool,
            description="Checkout a branch",
            inputSchema=git_schema
        )
        
        # Create a mock payload without repo_path
        mock_function = MagicMock()
        mock_function.name = "git_checkout"
        mock_function.arguments = {"branch": "main"}
        
        mock_payload = {
            "function": mock_function
        }
        
        result = await self.tool_manager.execute_tool(mock_payload)
        
        assert result["status"] == "success"
        assert result["content"][0]["text"] == "Git tool executed"
        
    def test_clear_tools(self):
        # Add a tool
        inputSchema = {
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
        
        self.tool_manager.register_tool(
            name="add_numbers",
            function=add,
            description="Add two numbers",
            inputSchema=inputSchema
        )
        
        # Clear tools
        self.tool_manager.clear_tools()
        
        # Verify tools are cleared
        assert len(self.tool_manager.tools) == 0 