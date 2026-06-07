"""Custom Exception Classes for AI Agent Framework.

All custom exceptions inherit from AgentException for easy catching.
Use specific exceptions for better error handling and debugging.
"""

from enum import Enum
from typing import Optional


class ErrorSeverity(Enum):
    """Severity levels for exceptions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentException(Exception):
    """Base exception for all AI Agent framework errors."""
    
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.message = message
        self.severity = severity


# ═══════════════════════════════════════════════════════════════════════════
# Model & LLM Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class ModelException(AgentException):
    """Base exception for model/LLM errors."""
    severity = ErrorSeverity.HIGH


class ModelLoadError(ModelException):
    """Raised when model fails to load."""
    pass


class ModelTimeoutError(ModelException):
    """Raised when model request times out."""
    severity = ErrorSeverity.MEDIUM


class ModelConnectionError(ModelException):
    """Raised when can't connect to model server (Ollama, etc)."""
    pass


class ModelOverloadError(ModelException):
    """Raised when model server is overloaded."""
    severity = ErrorSeverity.MEDIUM


class ModelInvalidRequestError(ModelException):
    """Raised when request format is invalid."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Tool Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class ToolException(AgentException):
    """Base exception for tool errors."""
    pass


class ToolNotFoundError(ToolException):
    """Raised when requested tool doesn't exist."""
    pass


class ToolExecutionError(ToolException):
    """Raised when tool execution fails."""
    severity = ErrorSeverity.HIGH


class ToolTimeoutError(ToolException):
    """Raised when tool execution times out."""
    pass


class ToolSecurityError(ToolException):
    """Raised when tool detects security threat (injection, etc)."""
    severity = ErrorSeverity.CRITICAL


class ToolValidationError(ToolException):
    """Raised when tool arguments are invalid."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# File & I/O Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class FileOperationError(AgentException):
    """Base exception for file operation errors."""
    pass


class PathTraversalError(FileOperationError):
    """Raised when path traversal attack is detected."""
    severity = ErrorSeverity.CRITICAL


class FileNotFoundError(FileOperationError):
    """Raised when file doesn't exist."""
    pass


class FileReadError(FileOperationError):
    """Raised when file read fails."""
    pass


class FileWriteError(FileOperationError):
    """Raised when file write fails."""
    pass


class FilePermissionError(FileOperationError):
    """Raised when file operation lacks permissions."""
    pass


class FileSizeError(FileOperationError):
    """Raised when file exceeds size limits."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# RAG & Retrieval Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class RAGException(AgentException):
    """Base exception for RAG pipeline errors."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    severity = ErrorSeverity.HIGH


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""
    pass


class ChunkingError(RAGException):
    """Raised when document chunking fails."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Memory & Context Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class MemoryException(AgentException):
    """Base exception for memory/context errors."""
    pass


class ConversationNotFoundError(MemoryException):
    """Raised when conversation ID doesn't exist."""
    pass


class ContextWindowError(MemoryException):
    """Raised when context window is exceeded."""
    pass


class MemoryPersistenceError(MemoryException):
    """Raised when memory save/load fails."""
    severity = ErrorSeverity.HIGH


# ═══════════════════════════════════════════════════════════════════════════
# Validation & Security Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class ValidationException(AgentException):
    """Base exception for validation errors."""
    pass


class InvalidInputError(ValidationException):
    """Raised when user input is invalid."""
    pass


class SanitizationError(ValidationException):
    """Raised when input sanitization fails."""
    pass


class SecurityScanError(ValidationException):
    """Raised when security scan fails."""
    pass


class AuthenticationError(ValidationException):
    """Raised when authentication fails."""
    severity = ErrorSeverity.HIGH


class AuthorizationError(ValidationException):
    """Raised when authorization fails."""
    severity = ErrorSeverity.HIGH


class RateLimitError(ValidationException):
    """Raised when rate limit is exceeded."""
    severity = ErrorSeverity.MEDIUM


# ═══════════════════════════════════════════════════════════════════════════
# Configuration Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class ConfigException(AgentException):
    """Base exception for configuration errors."""
    severity = ErrorSeverity.CRITICAL


class InvalidConfigError(ConfigException):
    """Raised when config is invalid."""
    pass


class MissingConfigError(ConfigException):
    """Raised when required config is missing."""
    pass


class ConfigLoadError(ConfigException):
    """Raised when config file can't be loaded."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def format_exception(exc: Exception, include_traceback: bool = False) -> str:
    """Format exception for logging/display.
    
    Args:
        exc: Exception to format
        include_traceback: Whether to include full traceback
        
    Returns:
        Formatted exception string
    """
    if isinstance(exc, AgentException):
        return f"[{exc.severity.value.upper()}] {type(exc).__name__}: {exc.message}"
    return f"{type(exc).__name__}: {str(exc)}"
