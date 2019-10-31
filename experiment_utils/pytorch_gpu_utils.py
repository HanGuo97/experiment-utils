import torch
import click
import functools
from contextlib import contextmanager
from typing import Optional, Callable


@contextmanager
def calculate_gpu_status(
        tag: str="",
        printer: Optional[Callable]=None) -> None:

    if printer is None:
        printer = print
    
    if not callable(printer):
        raise TypeError("`printer` must be callable")

    pre_tag = click.style(f"Before GPU Status {tag}", fg="red")
    post_tag = click.style(f"After GPU Status {tag}", fg="red")

    pre_memory_cached = torch.cuda.memory_cached()
    pre_memory_allocated = torch.cuda.memory_allocated()
    pre_memory_cached_G = pre_memory_cached / (1024 ** 3)
    pre_memory_allocated_G = pre_memory_allocated / (1024 ** 3)
    printer(f"{pre_tag}: Cached {pre_memory_cached_G: .2f} GB\n"
            f"{pre_tag}: Allocated {pre_memory_allocated_G:.2f} GB")

    try:
        yield

    finally:
        post_memory_cached = torch.cuda.memory_cached()
        post_memory_allocated = torch.cuda.memory_allocated()
        post_memory_cached_G = post_memory_cached / (1024 ** 3)
        post_memory_allocated_G = post_memory_allocated / (1024 ** 3)

        diff_memory_cached_G = (
            post_memory_cached_G -
            pre_memory_cached_G)
        
        diff_memory_allocated_G = (
            post_memory_allocated_G -
            pre_memory_allocated_G)

        diff_memory_cached_sign = (
            "+" if diff_memory_cached_G >= 0 else "")
        diff_memory_allocated_sign = (
            "+" if diff_memory_allocated_G >= 0 else "")

        printer(
            f"{post_tag}: Cached {post_memory_cached_G: .2f} GB "
            f"({diff_memory_cached_sign}{diff_memory_cached_G:.2f} GB)\n"
            f"{post_tag}: Allocated {post_memory_allocated_G:.2f} GB "
            f"({diff_memory_allocated_sign}{diff_memory_allocated_G:.2f} GB)")


@contextmanager
def empty_cache_after_execute() -> None:
    yield
    torch.cuda.empty_cache()


def empty_cache_wrapper(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        torch.cuda.empty_cache()
        return value
    return wrapper_decorator
