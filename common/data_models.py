"""
Data models and schemas for the Financial Story LLM system.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class DataType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class FuzzyState(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PatternDirection(Enum):
    DROPS_SHARPLY = -2
    DROPS = -1
    REMAINS_STEADY = 0
    RISES = 1
    RISES_SHARPLY = 2


@dataclass
class MarketData:
    """Base class for market data"""
    timestamp: datetime
    symbol: str
    value: float
    data_type: DataType
    source: str


@dataclass
class FuzzyMembership:
    """Fuzzy set membership values"""
    very_low: float = 0.0
    low: float = 0.0
    medium: float = 0.0
    high: float = 0.0
    very_high: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'very_low': self.very_low,
            'low': self.low,
            'medium': self.medium,
            'high': self.high,
            'very_high': self.very_high
        }


@dataclass
class PatternSequence:
    """FPR pattern sequence"""
    sequence: List[PatternDirection]
    confidence: float
    narrative: str


@dataclass
class NarrativeComponent:
    """Individual narrative component from fuzzification"""
    indicator: str
    fuzzy_states: FuzzyMembership
    pattern: PatternSequence
    timestamp: datetime
    confidence: float


@dataclass
class MarketStory:
    """Complete market story"""
    timestamp: datetime
    symbol: str
    title: str
    narrative: str
    components: List[NarrativeComponent]
    confidence_score: float
    validation_scores: Dict[str, float]


@dataclass
class TradingSignal:
    """Extracted trading signal"""
    symbol: str
    direction: str  # bullish, bearish, neutral
    conviction: float
    time_horizon: str
    entry_zone: List[float]
    targets: List[float]
    stops: List[float]
    risk_factors: List[str]
    confidence: float


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    story_generation_latency: float
    signal_extraction_latency: float
    directional_accuracy: float
    story_coherence: float
    signal_reliability: float
    system_uptime: float