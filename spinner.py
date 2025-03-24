import threading
import itertools
import time
import sys
import asyncio
from typing import Callable, Optional, Any

class Spinner:
    """A text-based spinner to indicate processing in the terminal."""

    def __init__(
        self,
        message: str = "Processing",
        delay: float = 0.1,
        chars: Optional[list[str]] = None
    ):
        """
        Initialize a spinner.

        Args:
            message: The message to display next to the spinner
            delay: Time delay between spinner updates in seconds
            chars: Optional list of characters to use for the spinner animation
        """
        self.message = message
        self.delay = delay
        self.chars = chars or ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self._thread = None
        self._stop_event = threading.Event()
        self._last_line_length = 0

    def _spin(self):
        """The spinning animation logic."""
        spinner = itertools.cycle(self.chars)
        try:
            while not self._stop_event.is_set():
                for char in spinner:
                    # Create the current line to display
                    current_line = f"{char} {self.message}..."

                    # Calculate padding to ensure we overwrite the entire previous line
                    padding = ' ' * max(0, self._last_line_length - len(current_line))

                    # Save current line length for next iteration
                    self._last_line_length = len(current_line)

                    # Write the line with padding and reset cursor to beginning of line
                    sys.stdout.write(f"\r{current_line}{padding}")
                    sys.stdout.flush()

                    time.sleep(self.delay)
                    if self._stop_event.is_set():
                        break
        finally:
            # Clear the line completely when stopping
            sys.stdout.write(f"\r{' ' * (self._last_line_length + 5)}\r")
            sys.stdout.flush()

    def start(self):
        """Start the spinner animation."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stop the spinner animation."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def __enter__(self):
        """Start the spinner when used as a context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the spinner when exiting the context."""
        self.stop()


def with_spinner(message: str = "Processing"):
    """
    Decorator to add a spinner to a function.

    Args:
        message: The message to display next to the spinner
    """
    def decorator(func: Callable):
        async def wrapper_async(*args, **kwargs):
            with Spinner(message):
                return await func(*args, **kwargs)

        def wrapper_sync(*args, **kwargs):
            with Spinner(message):
                return func(*args, **kwargs)

        # Handle both async and sync functions
        if asyncio.iscoroutinefunction(func):
            return wrapper_async
        return wrapper_sync

    return decorator


# Demo of the spinner if this file is run directly
if __name__ == "__main__":
    import asyncio

    # Demo 1: Basic usage
    print("Demo 1: Basic spinner")
    spinner = Spinner("Loading")
    spinner.start()
    time.sleep(3)
    spinner.stop()
    print("Finished loading!")

    # Demo 2: Context manager
    print("\nDemo 2: Context manager")
    with Spinner("Processing data"):
        time.sleep(3)
    print("Data processed!")

    # Demo 3: Test with variable length messages
    print("\nDemo 3: Variable length messages")
    spinner = Spinner("Short message")
    spinner.start()
    time.sleep(2)
    spinner.message = "This is a much longer message that should replace the shorter one"
    time.sleep(2)
    spinner.message = "Short again"
    time.sleep(2)
    spinner.stop()
    print("Variable message test complete!")

    # Demo 4: Decorator with async function
    print("\nDemo 4: Async decorator")
    @with_spinner("Fetching data")
    async def fetch_data():
        await asyncio.sleep(3)
        return "Data fetched!"

    async def run_async_demo():
        result = await fetch_data()
        print(result)

    asyncio.run(run_async_demo())

    # Demo 5: Decorator with sync function
    print("\nDemo 5: Sync decorator")
    @with_spinner("Calculating")
    def calculate():
        time.sleep(3)
        return "Calculation complete!"

    result = calculate()
    print(result)
