import torch
import gc
def clear_gpu_memory():
    """Clear GPU cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


import datetime
from functools import wraps

def time_execution(func):
    """
    Decorator that measures execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = datetime.datetime.now()
        result = func(*args, **kwargs)
        te = datetime.datetime.now() - t0
        print(f"Function {func.__name__} execution time: {str(te)}")
        return result
    return wrapper

# Alternative approach as a context manager
class TimeExecution:
    """
    Context manager for measuring execution time of a code block.
    """
    def __enter__(self):
        self.t0 = datetime.datetime.now()
        print(f'Code execution started at : {str(self.t0)}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        te = datetime.datetime.now() - self.t0
        print(f"Code block execution time: {str(te)}")
