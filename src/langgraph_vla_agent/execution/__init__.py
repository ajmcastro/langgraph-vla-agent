"""Execution layer.

Public surface for M1:
    ExecutorConfig — loop limits and safety-gate settings
    Executor       — observation→action loop
"""

from langgraph_vla_agent.execution.config import ExecutorConfig
from langgraph_vla_agent.execution.executor import Executor

__all__ = ["Executor", "ExecutorConfig"]
