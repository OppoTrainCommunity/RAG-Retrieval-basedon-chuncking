"""
Timing utilities for CV RAG System.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Timer:
    """
    Context manager and class for timing code execution.
    """
    
    def __init__(self, name: Optional[str] = None, log: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Name for the timer (used in logging).
            log: Whether to log the elapsed time.
        """
        self.name = name or "Timer"
        self.log = log
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        """Stop the timer and optionally log."""
        self.stop()
        if self.log:
            logger.info(f"{self.name}: {self.elapsed:.3f}s")
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed = None
    
    def stop(self) -> float:
        """
        Stop the timer.
        
        Returns:
            Elapsed time in seconds.
        """
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed = None


@contextmanager
def timer(name: Optional[str] = None, log: bool = True):
    """
    Context manager for timing code blocks.
    
    Args:
        name: Name for the timer.
        log: Whether to log elapsed time.
    
    Yields:
        Timer instance.
    
    Example:
        with timer("My operation") as t:
            do_something()
        print(f"Took {t.elapsed:.2f}s")
    """
    t = Timer(name=name, log=log)
    t.start()
    try:
        yield t
    finally:
        t.stop()
        if log:
            logger.info(f"{name or 'Operation'}: {t.elapsed:.3f}s")


def timed(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to decorate.
        name: Optional name for logging (defaults to function name).
    
    Returns:
        Decorated function.
    
    Example:
        @timed
        def my_function():
            pass
        
        @timed(name="Custom name")
        def another_function():
            pass
    """
    def decorator(fn: Callable) -> Callable:
        timer_name = name or fn.__name__
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{timer_name}: {elapsed:.3f}s")
            return result
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


class StepTimer:
    """
    Timer for tracking multiple steps in a pipeline.
    """
    
    def __init__(self):
        """Initialize step timer."""
        self.steps: dict = {}
        self._current_step: Optional[str] = None
        self._step_start: Optional[float] = None
    
    def start_step(self, step_name: str) -> None:
        """
        Start timing a step.
        
        Args:
            step_name: Name of the step.
        """
        if self._current_step is not None:
            self.end_step()
        
        self._current_step = step_name
        self._step_start = time.perf_counter()
    
    def end_step(self) -> Optional[float]:
        """
        End the current step.
        
        Returns:
            Elapsed time for the step.
        """
        if self._current_step is None:
            return None
        
        elapsed = time.perf_counter() - self._step_start
        self.steps[self._current_step] = elapsed
        
        step_name = self._current_step
        self._current_step = None
        self._step_start = None
        
        return elapsed
    
    def get_total(self) -> float:
        """
        Get total time across all steps.
        
        Returns:
            Total time in seconds.
        """
        return sum(self.steps.values())
    
    def get_summary(self) -> dict:
        """
        Get summary of all step timings.
        
        Returns:
            Dictionary with step times and total.
        """
        return {
            "steps": self.steps.copy(),
            "total": self.get_total(),
        }
    
    def reset(self) -> None:
        """Reset all step timings."""
        self.steps = {}
        self._current_step = None
        self._step_start = None
