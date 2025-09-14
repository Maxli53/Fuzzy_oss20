"""
DTN Symbol Parser - Pattern-based automatic categorization of IQFeed/DTN symbols
Based on DTN Calculated Indicators PDF patterns for smart routing and metadata extraction
"""
import re
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Parsed symbol information"""
    symbol: str
    category: str
    subcategory: str
    base_symbol: Optional[str] = None  # Added missing base_symbol attribute
    exchange: Optional[str] = None
    underlying: Optional[str] = None
    expiration: Optional[str] = None
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # 'C' or 'P'
    instrument_type: Optional[str] = None  # Added missing instrument_type
    is_valid: bool = True  # Added missing is_valid attribute
    metadata: Optional[Dict] = None
    storage_namespace: str = "default"


class DTNSymbolParser:
    """
    Pattern-based parser for IQFeed/DTN symbols enabling automatic categorization
    and smart routing for exploratory quantitative research.

    Handles:
    - Regular equities (AAPL, MSFT)
    - DTN Calculated Indicators (.Z suffix)
    - Options (complex patterns)
    - Futures (month codes, years)
    - Forex pairs
    - Indices (^SPX, $SPX)
    """

    def __init__(self):
        self.dtn_calculated_patterns = self._build_dtn_patterns()
        self.equity_patterns = self._build_equity_patterns()
        self.options_patterns = self._build_options_patterns()
        self.futures_patterns = self._build_futures_patterns()
        self.forex_patterns = self._build_forex_patterns()

        logger.info("DTNSymbolParser initialized with comprehensive pattern matching")

    def parse_symbol(self, symbol: str) -> SymbolInfo:
        """
        Parse any symbol and return categorized information.

        Args:
            symbol: Any IQFeed/DTN symbol (e.g., 'AAPL', 'JTNT.Z', 'AAPL240315C00150000')

        Returns:
            SymbolInfo with category, subcategory, and metadata
        """
        symbol = symbol.upper().strip()

        # Try each category in order of specificity
        parsers = [
            (self._parse_dtn_calculated, "dtn_calculated"),
            (self._parse_options, "options"),
            (self._parse_futures, "futures"),
            (self._parse_forex, "forex"),
            (self._parse_equity, "equity")
        ]

        for parser_func, category in parsers:
            result = parser_func(symbol)
            if result:
                result.category = category
                result.storage_namespace = self._get_storage_namespace(result)
                logger.debug(f"Parsed {symbol} as {category}: {result.subcategory}")
                return result

        # Fallback for unknown symbols
        return SymbolInfo(
            symbol=symbol,
            category="unknown",
            subcategory="unclassified",
            storage_namespace="unknown"
        )

    def _parse_dtn_calculated(self, symbol: str) -> Optional[SymbolInfo]:
        """Parse DTN calculated indicators (Pages 2-16 of DTN PDF)"""
        if not symbol.endswith('.Z'):
            return None

        base_symbol = symbol[:-2]  # Remove .Z suffix

        # Check against known DTN patterns
        for pattern, info in self.dtn_calculated_patterns.items():
            if re.match(pattern, base_symbol):
                return SymbolInfo(
                    symbol=symbol,
                    category="dtn_calculated",
                    subcategory=info['subcategory'],
                    exchange=info.get('exchange', 'DTN'),
                    metadata={
                        'description': info.get('description', ''),
                        'page_reference': info.get('page', ''),
                        'data_type': info.get('data_type', 'indicator'),
                        'category_group': info.get('category_group', 'market_internals')
                    }
                )

        # Generic DTN calculated indicator
        return SymbolInfo(
            symbol=symbol,
            category="dtn_calculated",
            subcategory="calculated_indicator",
            exchange="DTN",
            metadata={'description': 'DTN Calculated Indicator'}
        )

    def _parse_equity(self, symbol: str) -> Optional[SymbolInfo]:
        """Parse regular equity symbols"""
        # Basic equity patterns
        if re.match(r'^[A-Z]{1,5}$', symbol):
            return SymbolInfo(
                symbol=symbol,
                category="equity",
                subcategory="common_stock",
                base_symbol=symbol,
                instrument_type="stock",
                exchange="UNKNOWN",
                metadata={'asset_class': 'equity'}
            )

        # Index symbols
        if symbol.startswith(('^', '$')):
            return SymbolInfo(
                symbol=symbol,
                category="equity",
                subcategory="index",
                underlying=symbol[1:],
                metadata={'asset_class': 'index'}
            )

        # ETF patterns (common suffixes)
        if symbol.endswith(('ETF', 'EFA', 'VTI', 'SPY', 'QQQ')):
            return SymbolInfo(
                symbol=symbol,
                category="equity",
                subcategory="etf",
                metadata={'asset_class': 'etf'}
            )

        return None

    def _parse_options(self, symbol: str) -> Optional[SymbolInfo]:
        """Parse options symbols (OCC format)"""
        # Standard OCC format: AAPL240315C00150000
        # Pattern: ROOT + YYMMDD + C/P + 8-digit price
        occ_pattern = r'^([A-Z]{1,6})(\d{6})([CP])(\d{8})$'
        match = re.match(occ_pattern, symbol)

        if match:
            root, date_str, option_type, price_str = match.groups()

            # Parse expiration date
            try:
                exp_date = datetime.strptime(date_str, '%y%m%d')
                expiration = exp_date.strftime('%Y-%m-%d')
            except ValueError:
                expiration = date_str

            # Parse strike price (divide by 1000 for standard format)
            strike_price = float(price_str) / 1000

            return SymbolInfo(
                symbol=symbol,
                category="options",
                subcategory="equity_option",
                underlying=root,
                expiration=expiration,
                strike_price=strike_price,
                option_type=option_type,
                metadata={
                    'asset_class': 'option',
                    'option_style': 'american',
                    'multiplier': 100
                }
            )

        return None

    def _parse_futures(self, symbol: str) -> Optional[SymbolInfo]:
        """Parse futures symbols"""
        # Common futures patterns: ESU23, NQM24, etc.
        futures_pattern = r'^([A-Z]{1,3})([FGHJKMNQUVXZ])(\d{2,4})$'
        match = re.match(futures_pattern, symbol)

        if match:
            product, month_code, year = match.groups()

            month_mapping = {
                'F': 'January', 'G': 'February', 'H': 'March', 'J': 'April',
                'K': 'May', 'M': 'June', 'N': 'July', 'Q': 'August',
                'U': 'September', 'V': 'October', 'X': 'November', 'Z': 'December'
            }

            # Convert year to full year
            if len(year) == 2:
                year_int = int(year)
                if year_int >= 90:
                    full_year = 1900 + year_int
                else:
                    full_year = 2000 + year_int
            else:
                full_year = int(year)

            expiration = f"{full_year}-{list(month_mapping.keys()).index(month_code) + 1:02d}"

            return SymbolInfo(
                symbol=symbol,
                category="futures",
                subcategory="futures_contract",
                underlying=product,
                expiration=expiration,
                metadata={
                    'asset_class': 'futures',
                    'month_code': month_code,
                    'month_name': month_mapping.get(month_code, 'Unknown'),
                    'contract_year': full_year
                }
            )

        return None

    def _parse_forex(self, symbol: str) -> Optional[SymbolInfo]:
        """Parse forex symbols"""
        # Standard forex pairs: EURUSD, GBPJPY, etc.
        forex_pattern = r'^([A-Z]{3})([A-Z]{3})$'
        match = re.match(forex_pattern, symbol)

        if match and len(symbol) == 6:
            base_currency, quote_currency = match.groups()

            return SymbolInfo(
                symbol=symbol,
                category="forex",
                subcategory="currency_pair",
                metadata={
                    'asset_class': 'forex',
                    'base_currency': base_currency,
                    'quote_currency': quote_currency
                }
            )

        return None

    def _build_dtn_patterns(self) -> Dict[str, Dict]:
        """Build DTN calculated indicator patterns from PDF analysis"""
        return {
            # Page 2-3: Issues and Volume
            r'^TINT$': {'subcategory': 'total_issues', 'description': 'NYSE Total Issues', 'page': '2', 'category_group': 'breadth'},
            r'^TIQT$': {'subcategory': 'total_issues', 'description': 'NASDAQ Total Issues', 'page': '2', 'category_group': 'breadth'},
            r'^VINT$': {'subcategory': 'market_volume', 'description': 'NYSE Volume', 'page': '3', 'category_group': 'volume'},
            r'^VIQT$': {'subcategory': 'market_volume', 'description': 'NASDAQ Volume', 'page': '3', 'category_group': 'volume'},

            # Page 4: Net Tick
            r'^JTNT$': {'subcategory': 'net_tick', 'description': 'NYSE Net Tick', 'page': '4', 'category_group': 'sentiment'},
            r'^JTQT$': {'subcategory': 'net_tick', 'description': 'NASDAQ Net Tick', 'page': '4', 'category_group': 'sentiment'},

            # Page 5: TRIN
            r'^RINT$': {'subcategory': 'trin', 'description': 'NYSE TRIN', 'page': '5', 'category_group': 'sentiment'},
            r'^RIQT$': {'subcategory': 'trin', 'description': 'NASDAQ TRIN', 'page': '5', 'category_group': 'sentiment'},
            r'^RI6T$': {'subcategory': 'trin', 'description': '6-Month TRIN', 'page': '5', 'category_group': 'sentiment'},
            r'^RI1T$': {'subcategory': 'trin', 'description': '1-Month TRIN', 'page': '5', 'category_group': 'sentiment'},

            # Page 6: Highs and Lows
            r'^H\d+NH$': {'subcategory': 'highs_lows', 'description': 'New Highs', 'page': '6', 'category_group': 'breadth'},
            r'^H\d+NL$': {'subcategory': 'highs_lows', 'description': 'New Lows', 'page': '6', 'category_group': 'breadth'},

            # Page 8: Moving Averages
            r'^M\d+V$': {'subcategory': 'moving_avg', 'description': 'Volume Above MA', 'page': '8', 'category_group': 'momentum'},
            r'^M\d+B$': {'subcategory': 'moving_avg', 'description': 'Price Above MA', 'page': '8', 'category_group': 'momentum'},

            # Page 9: Premium
            r'^PREM$': {'subcategory': 'premium', 'description': 'Market Premium', 'page': '9', 'category_group': 'valuation'},
            r'^PRNQ$': {'subcategory': 'premium', 'description': 'NASDAQ Premium', 'page': '9', 'category_group': 'valuation'},
            r'^PRYM$': {'subcategory': 'premium', 'description': 'Premium YTD', 'page': '9', 'category_group': 'valuation'},

            # Pages 12-16: Options Indicators
            r'^[TP]COE[AT]$': {'subcategory': 'options_tick', 'description': 'Options Tick Data', 'page': '12', 'category_group': 'options'},
            r'^[IP]COE[AT]$': {'subcategory': 'options_issues', 'description': 'Options Issues', 'page': '13', 'category_group': 'options'},
            r'^[OP]COET$': {'subcategory': 'options_oi', 'description': 'Options Open Interest', 'page': '14', 'category_group': 'options'},
            r'^[VD][CP]OET$': {'subcategory': 'options_volume', 'description': 'Options Volume', 'page': '15', 'category_group': 'options'},
            r'^S[CP]OET$': {'subcategory': 'options_trin', 'description': 'Options TRIN', 'page': '16', 'category_group': 'options'},
        }

    def _build_equity_patterns(self) -> Dict[str, str]:
        """Build equity symbol patterns"""
        return {
            'common_stock': r'^[A-Z]{1,5}$',
            'index': r'^[\^$][A-Z0-9]+$',
            'etf': r'^[A-Z]{2,5}$'  # Will be refined with additional logic
        }

    def _build_options_patterns(self) -> Dict[str, str]:
        """Build options symbol patterns"""
        return {
            'occ_standard': r'^[A-Z]{1,6}\d{6}[CP]\d{8}$',
            'weekly': r'^[A-Z]{1,6}W\d{6}[CP]\d{8}$'
        }

    def _build_futures_patterns(self) -> Dict[str, str]:
        """Build futures symbol patterns"""
        return {
            'standard': r'^[A-Z]{1,3}[FGHJKMNQUVXZ]\d{2,4}$',
            'micro': r'^M[A-Z]{1,3}[FGHJKMNQUVXZ]\d{2,4}$'
        }

    def _build_forex_patterns(self) -> Dict[str, str]:
        """Build forex symbol patterns"""
        return {
            'major_pairs': r'^(EUR|GBP|USD|JPY|AUD|CAD|CHF|NZD){6}$',
            'exotic_pairs': r'^[A-Z]{6}$'
        }

    def _get_storage_namespace(self, symbol_info: SymbolInfo) -> str:
        """Determine ArcticDB storage namespace based on symbol category"""
        namespace_map = {
            'dtn_calculated': f'iqfeed/dtn_calculated/{symbol_info.subcategory}',
            'equity': f'iqfeed/base/{symbol_info.subcategory}',
            'options': f'iqfeed/base/options',
            'futures': f'iqfeed/base/futures',
            'forex': f'iqfeed/base/forex',
            'unknown': 'iqfeed/base/unknown'
        }

        return namespace_map.get(symbol_info.category, 'iqfeed/base/default')

    def get_symbol_category(self, symbol: str) -> str:
        """Quick category lookup for symbol"""
        return self.parse_symbol(symbol).category

    def get_storage_key(self, symbol: str, date: str) -> Tuple[str, str]:
        """Get storage namespace and key for symbol"""
        symbol_info = self.parse_symbol(symbol)
        namespace = symbol_info.storage_namespace
        key = f"{symbol}_{date}"
        return namespace, key

    def is_dtn_calculated(self, symbol: str) -> bool:
        """Check if symbol is a DTN calculated indicator"""
        return symbol.upper().endswith('.Z')

    def is_equity_option(self, symbol: str) -> bool:
        """Check if symbol is an equity option"""
        return self.parse_symbol(symbol).subcategory == 'equity_option'

    def get_underlying_symbol(self, symbol: str) -> Optional[str]:
        """Extract underlying symbol for derivatives"""
        symbol_info = self.parse_symbol(symbol)
        return symbol_info.underlying

    def categorize_symbols(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Categorize a list of symbols by type"""
        categorized = {}

        for symbol in symbols:
            category = self.get_symbol_category(symbol)
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(symbol)

        return categorized

    def get_symbol_metadata(self, symbol: str) -> Dict:
        """Get comprehensive metadata for a symbol"""
        symbol_info = self.parse_symbol(symbol)

        metadata = {
            'symbol': symbol_info.symbol,
            'category': symbol_info.category,
            'subcategory': symbol_info.subcategory,
            'storage_namespace': symbol_info.storage_namespace,
            'parsed_at': datetime.now().isoformat()
        }

        # Add optional fields if present
        optional_fields = ['exchange', 'underlying', 'expiration', 'strike_price', 'option_type']
        for field in optional_fields:
            value = getattr(symbol_info, field, None)
            if value is not None:
                metadata[field] = value

        # Add custom metadata
        if symbol_info.metadata:
            metadata.update(symbol_info.metadata)

        return metadata