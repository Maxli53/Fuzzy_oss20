# ArcticDB Phase 1: Storage Requirements for Financial Story-Based LLM System

## Overview

This document specifies the ArcticDB storage architecture required to support a 10-stage Financial Story-Based LLM system. The storage layer must handle the complete data flow from raw IQFeed data through fuzzy logic processing to LLM-generated narratives and performance tracking.

**Core Data Flow**: 
```
IQFeed Raw Data → Traditional Indicators → Fuzzy Engine (FS + FPR) → Story Components → 
LLM Training Data → Story Predictions → Trading Signals → Performance Tracking
```

---

## 1. Environment Setup

### Dependencies
```bash
conda create -n quant_sp500 python=3.10
conda activate quant_sp500

pip install arcticdb pandas numpy
pip install pyiqfeed polygon-api-client  
pip install transformers torch peft
pip install jupyterlab pytest
```

### Project Structure
```
sp500_quant/
├── config/
│   ├── config.yaml
│   └── universe.json
├── data/
│   └── arctic_storage/
├── src/
│   ├── storage/
│   │   ├── flexible_store.py
│   │   └── query_interface.py
│   └── utils/
└── tests/
```

---

## 2. Core Storage Architecture

### Library Organization
```python
# Complete library structure to support 10-stage story system
required_libraries = {
    # Raw data storage
    'prices_1min', 'prices_5min', 'prices_daily',
    'market_internals',     # TICK, TRIN, VIX, etc.
    'news_raw',            # Polygon.io news feeds
    'economic_data',       # Quandl data
    'options_data',        # CBOE data
    
    # Fuzzy processing pipeline
    'indicators_technical', # SMA, RSI, MACD → input to fuzzy engine
    'fuzzy_sets',          # FS membership values (0-1)
    'fuzzy_patterns',      # FPR pattern match scores (0-1)  
    'fuzzy_output',        # Combined fuzzy results
    
    # Story system requirements
    'narrative_components', # Story building blocks from fuzzy output
    'assembled_stories',    # Complete market narratives
    'story_training_data',  # LLM training sequences
    'predicted_stories',    # LLM-generated predictions
    'validated_stories',    # Quality-checked stories
    
    # Signals and performance
    'extracted_signals',    # Trading signals from stories
    'story_performance',    # Prediction accuracy tracking
    'portfolio_integration' # Multi-asset synthesis
}
```

### Enhanced FlexibleDataStore
```python
class StoryBasedFlexibleStore:
    """
    Enhanced ArcticDB store supporting story-based LLM architecture.
    
    Key Features:
    - Schema-less DataFrame storage
    - Automatic library creation and routing  
    - Rich metadata generation for all data types
    - Optimized storage for large datasets
    - Story-specific data structures
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Initialize Arctic connection
        # Setup all required libraries
        # Configure caching and optimization
    
    def store_any_data(self, data_path: str, df: pd.DataFrame, 
                      metadata: Optional[Dict] = None) -> str:
        """
        Universal storage with automatic routing.
        
        Examples:
            store_any_data('prices/1min/AAPL', ohlcv_df)
            store_any_data('fuzzy_sets/AAPL', membership_df)
            store_any_data('stories/assembled/SPY', story_df)
        """
        
    # Fuzzy Logic Specific Storage
    def store_traditional_indicators(self, symbol: str, indicators_df: pd.DataFrame):
        """Store SMA, RSI, MACD etc. - inputs to fuzzy engine"""
        
    def store_fuzzy_sets_output(self, symbol: str, fuzzy_sets_df: pd.DataFrame):
        """Store fuzzy membership values (0-1) from FS processing"""
        
    def store_fuzzy_patterns_output(self, symbol: str, patterns_df: pd.DataFrame):
        """Store pattern match scores (0-1) from FPR processing"""
        
    # Story System Specific Storage  
    def store_narrative_components(self, symbol: str, components_df: pd.DataFrame):
        """Store story building blocks from fuzzy processing"""
        
    def store_assembled_story(self, symbol: str, story_data: Dict):
        """Store complete market narratives with metadata"""
        
    def store_story_training_data(self, training_batch: Dict):
        """Store multi-day story sequences for LLM training"""
        
    def store_predicted_story(self, symbol: str, prediction_data: Dict):
        """Store LLM-generated story predictions"""
        
    def store_extracted_signals(self, symbol: str, signals_df: pd.DataFrame):
        """Store trading signals derived from stories"""
        
    def store_story_performance(self, symbol: str, performance_df: pd.DataFrame):
        """Store story prediction accuracy and performance metrics"""
```

---

## 3. Data Structure Requirements

### Continuous Data Flow Storage

#### Traditional Indicators (Input to Fuzzy Engine)
```python
# Expected DataFrame structure for fuzzy processing inputs
indicators_df_structure = {
    'index': 'datetime',
    'columns': [
        'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'rsi_14', 'stoch_k', 'williams_r',
        'macd', 'macd_signal', 'macd_histogram',
        'atr_14', 'bollinger_upper', 'bollinger_lower',
        'volume_sma_20', 'volume_ratio', 'obv'
    ],
    'metadata': {
        'type': 'traditional_indicators',
        'symbol': 'AAPL',
        'ready_for_fuzzy': True
    }
}
```

#### Fuzzy Sets Output (5-State Membership Values)
```python
# Expected DataFrame structure for fuzzy sets output
fuzzy_sets_df_structure = {
    'index': 'datetime',
    'columns': [
        'rsi_very_low', 'rsi_low', 'rsi_medium', 'rsi_high', 'rsi_very_high',
        'sma_trend_down', 'sma_trend_neutral', 'sma_trend_up',
        'volume_very_low', 'volume_low', 'volume_normal', 'volume_high', 'volume_very_high',
        # ... membership values (0-1) for each indicator and state
    ],
    'metadata': {
        'type': 'fuzzy_sets_output',
        'symbol': 'AAPL',
        'fuzzy_states': 5,
        'membership_range': [0.0, 1.0]
    }
}
```

#### Fuzzy Patterns Output (FPR Pattern Matches)
```python
# Expected DataFrame structure for pattern recognition output
fuzzy_patterns_df_structure = {
    'index': 'datetime', 
    'columns': [
        'price_rises_then_drops', 'volume_surge_pattern',
        'momentum_divergence', 'reversal_pattern',
        'breakout_pattern', 'consolidation_pattern'
        # ... pattern match scores (0-1)
    ],
    'metadata': {
        'type': 'fuzzy_patterns_output',
        'symbol': 'AAPL', 
        'pattern_window': 10,
        'confidence_threshold': 0.6
    }
}
```

### Story System Storage Requirements

#### Narrative Components Structure
```python
narrative_components_structure = {
    'index': 'datetime',
    'columns': [
        'price_narrative',      # "price rose moderately, then accelerated"
        'volume_narrative',     # "volume elevated, suggesting institutional interest"  
        'momentum_narrative',   # "momentum indicators showing strength"
        'volatility_narrative', # "volatility environment remains constructive"
        'event_context',        # "earnings announcement positive"
        'confidence_score'      # Overall narrative confidence (0-1)
    ],
    'metadata': {
        'type': 'narrative_components',
        'symbol': 'AAPL',
        'story_ready': True
    }
}
```

#### Assembled Stories Structure
```python
assembled_story_structure = {
    'index': 'datetime',
    'columns': [
        'story_text',          # Complete narrative text
        'story_summary',       # Brief summary
        'coherence_score',     # Story quality metrics
        'completeness_score',  
        'directional_bias',    # bullish/bearish/neutral
        'confidence_level',    # Overall confidence
        'components_used',     # Which components were included
        'story_theme'          # market_update, earnings_reaction, etc.
    ],
    'metadata': {
        'type': 'assembled_story',
        'symbol': 'AAPL',
        'quality_validated': True
    }
}
```

#### LLM Training Data Structure  
```python
training_data_structure = {
    'columns': [
        'sequence_id',         # Unique identifier
        'context_stories',     # Previous 6 days of stories
        'target_story',        # Day 7 story (prediction target)
        'context_length',      # Number of context days
        'avg_quality_score',   # Average quality of sequence
        'market_regime',       # bull/bear/neutral
        'volatility_regime'    # high/medium/low
    ],
    'metadata': {
        'type': 'llm_training_data',
        'sequence_length': 7,
        'quality_threshold': 0.7,
        'training_ready': True
    }
}
```

---

## 4. Query Interface Requirements

### Enhanced Query Capabilities
```python
class StoryBasedQuery:
    """Enhanced query interface for story-based system"""
    
    def load_complete_pipeline_data(self, symbol: str, date: str = None):
        """Load data from all pipeline stages for analysis"""
        return {
            'raw_prices': self.load_multiple([f'prices_1min/{symbol}']),
            'indicators': self.load_multiple([f'indicators_technical/{symbol}']),
            'fuzzy_sets': self.load_multiple([f'fuzzy_sets/{symbol}']),
            'fuzzy_patterns': self.load_multiple([f'fuzzy_patterns/{symbol}']),
            'stories': self.load_multiple([f'assembled_stories/{symbol}']),
            'signals': self.load_multiple([f'extracted_signals/{symbol}'])
        }
    
    def get_training_sequences(self, date_range: Tuple[str, str], 
                              quality_threshold: float = 0.7):
        """Retrieve high-quality story sequences for LLM training"""
        
    def analyze_story_performance(self, symbol: str, lookback_days: int = 30):
        """Analyze story prediction accuracy for model improvement"""
        
    def get_fuzzy_signal_data(self, symbol: str):
        """Get complete fuzzy processing results for signal generation"""
```

---

## 5. Storage Optimization Features

### Performance Optimizations
```python
class OptimizedStorage:
    """Performance optimizations for large-scale processing"""
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatic dtype optimization (up to 50% memory reduction)"""
        
    def smart_compression(self, df: pd.DataFrame) -> Dict:
        """Optimal compression settings based on data characteristics"""
        
    def cache_management(self, cache_size_mb: int = 1000):
        """LRU cache for frequently accessed data"""
```

### Data Quality Validation
```python
class DataQualityValidator:
    """Comprehensive data validation for financial data"""
    
    def validate_price_data(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Validate OHLCV data consistency (High >= Low, etc.)"""
        
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Outlier detection for data quality assurance"""
        
    def assess_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score"""
```

---

## 6. Metadata Requirements

### Comprehensive Metadata System
```python
# Auto-generated metadata for all stored data
standard_metadata = {
    # Basic properties
    'shape': (rows, columns),
    'columns': ['list', 'of', 'column', 'names'],
    'dtypes': {'col1': 'float64', 'col2': 'int32'},
    'stored_at': '2024-01-15T10:30:00',
    'memory_usage_mb': 12.5,
    
    # Data quality
    'missing_data_pct': 0.02,
    'completeness_score': 98.0,
    'outliers_detected': 3,
    
    # Financial data specific
    'date_range': ['2024-01-01', '2024-01-15'],
    'frequency': 'daily',
    'symbol': 'AAPL',
    'data_source': 'iqfeed',
    
    # Story system specific
    'story_quality_score': 0.85,
    'fuzzy_confidence': 0.78,
    'narrative_completeness': 0.92,
    'processing_stage': 'stage_3_story_assembly'
}
```

---

## 7. Configuration Requirements

### System Configuration
```yaml
# config/config.yaml
arctic_storage_path: "lmdb://./data/arctic_storage"

# Data sources
iqfeed:
  host: "127.0.0.1"
  port: 9100
  username: "YOUR_USERNAME"
  password: "YOUR_PASSWORD"

polygon_api_key: "YOUR_API_KEY"
quandl_api_key: "YOUR_API_KEY"

# Storage optimization
storage:
  cache_size_mb: 1000
  compression_enabled: true
  auto_optimize_dtypes: true

# Fuzzy processing settings
fuzzy:
  n_states: 5
  rolling_window: 60
  std_multiplier: 2.0
  pattern_window: 10

# Story system settings  
stories:
  sequence_length: 7
  quality_threshold: 0.7
  max_story_length: 2000
```

---

## 8. Technical Requirements Summary

### Storage Capabilities Required
- **Schema-less storage**: Handle any DataFrame structure
- **Automatic routing**: Library selection from data path  
- **Rich metadata**: Comprehensive data documentation
- **Performance optimization**: Memory and storage efficiency
- **Fuzzy data support**: Membership values and pattern scores
- **Story data structures**: Narrative text with quality metrics
- **Training data management**: Multi-day sequences for LLM
- **Performance tracking**: Accuracy metrics over time

### Query Capabilities Required
- **Pattern matching**: Wildcard searches across libraries
- **Multi-stage loading**: Complete pipeline data retrieval
- **Date range filtering**: Time-based data slicing
- **Quality filtering**: Data selection by quality metrics
- **Cross-asset analysis**: Portfolio-level data synthesis

### Performance Requirements
- **Latency**: < 30 seconds for story generation pipeline
- **Throughput**: 500+ symbols simultaneously  
- **Storage efficiency**: 50%+ reduction via optimization
- **Memory management**: Smart caching for frequent access
- **Scalability**: Support for full S&P 500 processing

### Data Validation Requirements
- **Price data consistency**: OHLC validation, outlier detection
- **Story quality assessment**: Coherence and completeness scoring
- **Fuzzy output validation**: Membership value ranges
- **Training data quality**: Sequence validation for LLM training

---

## 9. Integration Points

### Input Data Sources
- **IQFeed**: Real-time and historical OHLCV data
- **Polygon.io**: News feeds and corporate actions  
- **Quandl**: Economic indicators
- **CBOE**: Options and volatility data

### Processing Integration
- **Fuzzy Engine**: Store inputs (indicators) and outputs (membership values)
- **LLM System**: Store training data and generated predictions
- **Signal Generation**: Store extracted trading signals
- **Performance Tracking**: Store accuracy and learning metrics

### Output Consumers
- **Trading Systems**: Signal extraction and decision support
- **Research Platform**: Multi-stage data analysis
- **Performance Monitoring**: Continuous improvement feedback
- **Portfolio Management**: Cross-asset story synthesis

This ArcticDB specification provides the complete storage foundation required to support your 10-stage Financial Story-Based LLM Architecture while maintaining flexibility for future enhancements.