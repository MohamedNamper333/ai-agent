"""Structured Logging System"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        
        return json.dumps(log_entry, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        msg = f"{color}[{timestamp}] {record.levelname:8} {record.name}: {record.getMessage()}{self.RESET}"
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


class AgentLogger:
    def __init__(self, name: str, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            self._setup_handlers()
        
        self._metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "tool_calls": 0,
            "start_time": time.time(),
        }
    
    def _setup_handlers(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(HumanReadableFormatter())
        self.logger.addHandler(console_handler)
        
        today = datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(
            self.log_dir / f"agent_{today}.jsonl",
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
        error_handler = logging.FileHandler(
            self.log_dir / f"errors_{today}.jsonl",
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra={"extra_data": kwargs})
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra={"extra_data": kwargs})
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra={"extra_data": kwargs})
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra={"extra_data": kwargs})
        self._metrics["total_errors"] += 1
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra={"extra_data": kwargs})
        self._metrics["total_errors"] += 1
    
    def log_request(self, endpoint: str, method: str, status_code: int, duration: float):
        self._metrics["total_requests"] += 1
        self.info(
            f"HTTP {method} {endpoint} - {status_code}",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
        )
    
    def log_tool_call(self, tool_name: str, success: bool, duration: float):
        self._metrics["tool_calls"] += 1
        level = "info" if success else "warning"
        getattr(self, level)(
            f"Tool call: {tool_name} ({'success' if success else 'failed'})",
            tool_name=tool_name,
            success=success,
            duration_ms=round(duration * 1000, 2),
        )
    
    def log_chat(self, message: str, response_length: int, user_id: Optional[str] = None):
        self.info(
            f"Chat message processed ({response_length} chars)",
            message_length=len(message),
            response_length=response_length,
            user_id=user_id,
        )
    
    @contextmanager
    def timer(self, operation: str):
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.info(f"Operation completed: {operation}", duration_ms=round(duration * 1000, 2))
    
    def get_metrics(self) -> dict:
        uptime = time.time() - self._metrics["start_time"]
        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self._metrics["total_requests"],
            "total_errors": self._metrics["total_errors"],
            "tool_calls": self._metrics["tool_calls"],
            "requests_per_minute": round(
                self._metrics["total_requests"] / max(uptime / 60, 1), 2
            ),
        }


_loggers: dict[str, AgentLogger] = {}


def get_logger(name: str = "agent") -> AgentLogger:
    if name not in _loggers:
        _loggers[name] = AgentLogger(name)
    return _loggers[name]
