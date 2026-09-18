import time


def with_retry(fn, *args, max_retries=3, base_delay=1.0, retry_exceptions=(Exception,), label=None, **kwargs):
    """Call fn(*args, **kwargs), retrying on retry_exceptions with exponential backoff."""
    name = label or getattr(fn, "__name__", "call")

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except retry_exceptions as e:
            if attempt == max_retries - 1:
                print(f"\n❌ {name} failed after {max_retries} attempts")
                print(f"Error: {e}\n")
                raise

            wait_time = base_delay * (2 ** attempt)
            print(f"⚠️ {name} attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
