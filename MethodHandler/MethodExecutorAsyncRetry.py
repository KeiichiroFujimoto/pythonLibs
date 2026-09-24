import asyncio
from typing import Callable, Any

class MethodExecutorAsyncRetry:
  @staticmethod
  async def retry_async(
    func: Callable[..., Any],
    *args,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    **kwargs
  ) -> Any:
    """
    Retry an async function with exponential backoff.

    Parameters:
        func: The async function to call.
        *args, **kwargs: Arguments to pass to the function.
        max_retries: Maximum number of attempts.
        retry_delay: Delay between retries in seconds.

    Returns:
        The result of the function if successful, or None if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                return None

