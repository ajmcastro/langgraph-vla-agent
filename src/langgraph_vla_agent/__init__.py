"""LangGraph VLA Agent.

Agentic planning and orchestration layer over a Vision-Language-Action (VLA)
sensorimotor policy (SmolVLA / LeRobot).

LangGraph operates at the goal/subtask timescale.
The VLA policy operates at the observation→action timescale.
The two layers communicate only through the RobotPolicy and RobotEnvironment
abstractions — never through raw actuator commands.
"""

__version__ = "0.1.0"
