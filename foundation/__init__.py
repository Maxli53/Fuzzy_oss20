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
from .models.base import BaseFoundationModel, TimestampedModel, ValidatedModel

# Market data models
from .models.market import (
    TickData,
    OHLCVBar,
    OrderBookSnapshot,
    MarketSession
)

# Metadata models
from .models.metadata import (
    SpreadStatistics,
    TradeClassification,
    LiquidityProfile,
    ExecutionQuality,
    MarketRegime,
    ToxicityMetrics,
    InstitutionalFlow,
    SymbolDayMetadata,
    MetadataComputationContext
)

# Enumerations
from .models.enums import (
    TradeSign,
    TickDirection,
    ParticipantType,
    ExchangeCode,
    TimeInterval,
    VolatilityRegime,
    LiquidityState,
    TrendState,
    MicrostructureRegime,
    DataQualityLevel,
    BarType,
    TradeCondition
)

from .utils.logging_config import get_logger

__all__ = [
    # Core modules
    "models",
    "events",
    "pipeline",
    "timeseries",
    "utils",

    # Base models
    "BaseFoundationModel",
    "TimestampedModel",
    "ValidatedModel",

    # Market data models
    "TickData",
    "OHLCVBar",
    "OrderBookSnapshot",
    "MarketSession",

    # Metadata models
    "SpreadStatistics",
    "TradeClassification",
    "LiquidityProfile",
    "ExecutionQuality",
    "MarketRegime",
    "ToxicityMetrics",
    "InstitutionalFlow",
    "SymbolDayMetadata",
    "MetadataComputationContext",

    # Enumerations
    "TradeSign",
    "TickDirection",
    "ParticipantType",
    "ExchangeCode",
    "TimeInterval",
    "VolatilityRegime",
    "LiquidityState",
    "TrendState",
    "MicrostructureRegime",
    "DataQualityLevel",
    "BarType",
    "TradeCondition",

    # Utilities
    "get_logger"
]