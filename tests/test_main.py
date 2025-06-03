import sys
import unittest
from unittest.mock import patch, MagicMock

# Mock the 'mcp' package and its submodules to prevent ImportError
# This is because mcpclient.py (imported by main.py) imports from 'mcp'
# and its sub-packages.
mock_mcp_package = MagicMock()
mock_mcp_package.__path__ = ['dummy_path'] # Make it look like a package
sys.modules['mcp'] = mock_mcp_package
sys.modules['mcp.client'] = MagicMock()
sys.modules['mcp.client.stdio'] = MagicMock()

# Now that mcp and its structure are mocked, we can import main
import main  # This imports the refactored main.py
from agent import OllamaAgent # OllamaAgent is imported in main.py, so it's a dependency
from ollama_toolmanager import OllamaToolManager # Also a dependency for OllamaAgent
from rich.console import Console
from rich.prompt import Prompt

class TestMainModelSelection(unittest.TestCase):

    @patch('main.ollama.list')
    @patch('main.Prompt.ask')
    @patch('main.Console') # Mock Console to prevent actual printing
    def test_prompt_for_model_selection(self, MockConsole, MockPromptAsk, MockOllamaList):
        # Arrange
        mock_console_instance = MockConsole.return_value
        MockOllamaList.return_value = {'models': [{'name': 'model1:latest'}, {'name': 'model2:latest'}]}
        MockPromptAsk.return_value = 'model1:latest' # Simulate user selecting the first model

        # Act
        # We are testing the synchronous part that sets up the agent
        # The OllamaToolManager is created inside select_model_and_initialize_agent
        # We also need to mock OllamaAgent if its creation has side effects or further dependencies not mocked
        with patch('main.OllamaAgent') as MockOllamaAgentInstance:
            main.select_model_and_initialize_agent(mock_console_instance)

        # Assert
        MockOllamaList.assert_called_once()
        mock_console_instance.print.assert_any_call("\n[bold cyan]Available Ollama Models:[/bold cyan]")
        MockPromptAsk.assert_called_once_with("Please select a model from the list above", console=mock_console_instance)
        # Check if OllamaAgent was called (implies selection was successful)
        MockOllamaAgentInstance.assert_called_once()


    @patch('main.ollama.list')
    @patch('main.Prompt.ask')
    @patch('main.OllamaAgent') # Mock OllamaAgent to check its instantiation
    @patch('main.Console') # Mock Console
    def test_selected_model_passed_to_agent(self, MockConsole, MockOllamaAgent, MockPromptAsk, MockOllamaList):
        # Arrange
        mock_console_instance = MockConsole.return_value
        test_model_name = 'model2:latest'
        MockOllamaList.return_value = {'models': [{'name': 'model1:latest'}, {'name': test_model_name}]}
        MockPromptAsk.return_value = test_model_name # Simulate user selecting this model

        # Act
        # The OllamaToolManager is created inside select_model_and_initialize_agent,
        # so we don't need to mock it separately unless its constructor does something complex.
        initialized_agent = main.select_model_and_initialize_agent(mock_console_instance)

        # Assert
        MockOllamaList.assert_called_once()
        MockPromptAsk.assert_called_once_with("Please select a model from the list above", console=mock_console_instance)

        # Check that OllamaAgent was called with the selected model name and an instance of OllamaToolManager
        MockOllamaAgent.assert_called_once()
        args, kwargs = MockOllamaAgent.call_args
        self.assertEqual(args[0], test_model_name)
        self.assertIsInstance(args[1], OllamaToolManager) # Check it's called with a tool manager

        # Also check the returned agent is the one that was created
        self.assertIsNotNone(initialized_agent)
        self.assertIsInstance(initialized_agent, MockOllamaAgent.return_value.__class__) # Check it's the mocked agent

    @patch('main.ollama.list')
    @patch('main.Prompt.ask')
    @patch('main.Console')
    def test_invalid_model_selection_reprompts(self, MockConsole, MockPromptAsk, MockOllamaList):
        # Arrange
        mock_console_instance = MockConsole.return_value
        valid_model = 'model1:latest'
        MockOllamaList.return_value = {'models': [{'name': valid_model}]}
        # Simulate user entering invalid model first, then a valid one
        MockPromptAsk.side_effect = ['invalid_model', valid_model]

        with patch('main.OllamaAgent') as MockOllamaAgentInstance:
            # Act
            main.select_model_and_initialize_agent(mock_console_instance)

            # Assert
            self.assertEqual(MockPromptAsk.call_count, 2) # Called twice
            mock_console_instance.print.assert_any_call("[prompt.invalid]Invalid model name: 'invalid_model'. Please choose from the list.")
            MockOllamaAgentInstance.assert_called_once_with(valid_model, unittest.mock.ANY)


    @patch('main.ollama.list')
    @patch('main.Console')
    def test_no_models_found(self, MockConsole, MockOllamaList):
        # Arrange
        mock_console_instance = MockConsole.return_value
        MockOllamaList.return_value = {'models': []} # No models available

        # Act
        agent = main.select_model_and_initialize_agent(mock_console_instance)

        # Assert
        self.assertIsNone(agent)
        mock_console_instance.print.assert_any_call("[bold red]No Ollama models found. Please pull a model first.[/bold red]")
        MockConsole.return_value.ask.assert_not_called() # Prompt.ask should not be called

    @patch('main.ollama.list')
    @patch('main.Console')
    def test_ollama_list_raises_exception(self, MockConsole, MockOllamaList):
        # Arrange
        mock_console_instance = MockConsole.return_value
        MockOllamaList.side_effect = Exception("Ollama connection error")

        # Act
        agent = main.select_model_and_initialize_agent(mock_console_instance)

        # Assert
        self.assertIsNone(agent)
        mock_console_instance.print.assert_any_call("[bold red]Error fetching Ollama models: Ollama connection error[/bold red]")
        MockConsole.return_value.ask.assert_not_called()


if __name__ == '__main__':
    unittest.main()
