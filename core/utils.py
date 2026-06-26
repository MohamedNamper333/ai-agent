"""Shared utility functions to reduce code duplication across tools."""
import os
import re
from pathlib import Path
from typing import Optional, Callable


def validate_path(path: str, base_dir: Optional[str] = None) -> Path:
    """Validate a path and prevent path traversal attacks.
    
    Args:
        path: The path to validate
        base_dir: The base directory to restrict to (defaults to current working directory)
    
    Returns:
        Resolved Path object
        
    Raises:
        PermissionError: If path traversal is detected
        FileNotFoundError: If path doesn't exist
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    base = Path(base_dir).resolve()
    target = (base / path).resolve()
    
    if not str(target).startswith(str(base)):
        raise PermissionError(f"Access denied: Path traversal detected for '{path}'")
    
    return target


def strip_html(text: str) -> str:
    """Strip HTML tags and decode entities from text.
    
    Args:
        text: HTML content to clean
        
    Returns:
        Clean text with HTML removed
    """
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    return text


def require_optional_import(module_name: str, package_name: str = None) -> Callable:
    """Decorator that checks for optional module availability.
    
    Args:
        module_name: The module to import
        package_name: Display name for error message (defaults to module_name)
    
    Returns:
        Decorator function that wraps the target function
    """
    if package_name is None:
        package_name = module_name
    
    def decorator(func: Callable) -> Callable:
        """Decorator."""
        def wrapper(*args, **kwargs):
            """Wrapper."""
            try:
                return func(*args, **kwargs)
            except ImportError:
                return f"Error: {package_name} not installed. Install: pip install {package_name}"
            except Exception as e:
                if "No module named" in str(e) or module_name in str(e):
                    return f"Error: {package_name} not installed. Install: pip install {package_name}"
                raise
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, backoff: float = 1.0) -> Callable:
    """Decorator that retries function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff: Base backoff time in seconds
    
    Returns:
        Decorator function
    """
    import time
    
    def decorator(func: Callable) -> Callable:
        """Decorator."""
        def wrapper(*args, **kwargs):
            """Wrapper."""
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(backoff * (2 ** attempt))
            raise last_exception
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def safe_file_operation(filepath: str, base_dir: Optional[str] = None) -> Path:
    """Validate file path and return resolved Path if safe.
    
    Args:
        filepath: Path to validate
        base_dir: Base directory to restrict to
    
    Returns:
        Resolved Path if valid
        
    Raises:
        PermissionError: If path traversal detected
        FileNotFoundError: If file doesn't exist
    """
    path = validate_path(filepath, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return path
