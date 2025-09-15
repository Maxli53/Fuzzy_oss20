"""
Multi-Asset Class Support

Asset-specific business logic and calculations for different instrument types.
Provides unified interface while handling asset-specific features like options Greeks,
futures contango, and ETF tracking errors.

Supported Asset Classes:
- Equities: Dividends, splits, sectors
- Options: Greeks (delta, gamma, theta, vega, rho), implied volatility
- Futures: Contango/backwardation, roll dates, margins
- ETFs: NAV tracking, underlying holdings
"""

__all__ = []