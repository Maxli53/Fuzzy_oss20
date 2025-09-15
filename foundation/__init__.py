"""
Foundation Layer - Institutional-Grade Cross-Cutting Infrastructure

This foundation layer provides core patterns and utilities that support all 10 stages
of the Financial Story LLM system. It implements institutional best practices while
remaining appropriate for a small quantitative hedge fund operation.

Architecture Patterns:
1. Data Models (Single Source of Truth)
2. Event-Driven Architecture
3. Pipeline Orchestration
4. Resource Management
5. Time Series Framework
6. Data Quality Framework
7. Workflow Orchestration

Additional Infrastructure:
- Multi-Asset Class Support
- Security & Authentication
- Market Session Management
- Data Reconciliation
- Portfolio Management
- External Integrations
- Quantitative Libraries
- Research & Analytics Platform
- Configuration Management
- Performance Optimization
- Risk Management
- Order Execution
- Error Handling
- System Monitoring
- Network Management
- Testing Infrastructure
"""

__version__ = "1.0.0"
__author__ = "Fuzzy OSS20 Team"

# Core foundation imports
from . import models
from . import events
from . import pipeline
from . import timeseries
from . import utils

# Make commonly used classes available at package level
from .models.base import BaseFoundationModel
from .utils.logging_config import get_logger

__all__ = [
    "models",
    "events",
    "pipeline",
    "timeseries",
    "utils",
    "BaseFoundationModel",
    "get_logger"
]