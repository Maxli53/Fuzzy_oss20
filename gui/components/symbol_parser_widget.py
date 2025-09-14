"""
Symbol Parser Widget for Streamlit
Displays symbol parsing results in a structured format
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any

def display_symbol_parse_results(parse_result: Dict[str, Any]):
    """
    Display symbol parsing results in a structured Streamlit widget

    Args:
        parse_result: Result dictionary from GUIDataInterface.parse_symbol()
    """
    if parse_result['success']:
        symbol = parse_result['symbol']
        parsed_info = parse_result['parsed_info']
        routing = parse_result['routing_recommendation']

        # Create columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"✅ Symbol '{symbol}' parsed successfully")

            # Basic symbol information
            st.subheader("Symbol Information")
            info_data = {
                'Category': parsed_info['category'].title(),
                'Subcategory': parsed_info['subcategory'].title() if parsed_info['subcategory'] else 'N/A',
                'Base Symbol': parsed_info['base_symbol'],
                'Exchange': parsed_info['exchange'] or 'Unknown',
                'Instrument Type': parsed_info['instrument_type'].title() if parsed_info['instrument_type'] else 'Unknown',
                'Valid': '✅ Yes' if parsed_info['is_valid'] else '❌ No'
            }

            for key, value in info_data.items():
                st.write(f"**{key}:** {value}")

        with col2:
            st.subheader("Storage Routing")

            if routing == 'flexible':
                st.success(f"🎯 **Recommended Backend:** Flexible Storage")
                st.info("This symbol will use the new flexible ArcticDB storage with automatic library creation.")
            elif routing == 'legacy':
                st.warning(f"🔄 **Recommended Backend:** Legacy Storage")
                st.info("This symbol will use the legacy TickStore for backward compatibility.")
            else:
                st.info(f"📂 **Recommended Backend:** {routing.title()}")

            # Show expected storage path
            category = parsed_info['category']
            base_symbol = parsed_info['base_symbol']

            st.subheader("Expected Storage Path")
            if category == 'equity':
                expected_path = f"iqfeed/equity/stocks/{base_symbol}/ticks/YYYY-MM-DD"
            elif category == 'dtn_calculated':
                subcategory = parsed_info['subcategory'] or 'indicators'
                expected_path = f"dtn/{subcategory}/{base_symbol}/ticks/YYYY-MM-DD"
            elif category == 'futures':
                expected_path = f"iqfeed/futures/{base_symbol}/ticks/YYYY-MM-DD"
            elif category == 'options':
                expected_path = f"iqfeed/options/{base_symbol}/ticks/YYYY-MM-DD"
            elif category == 'forex':
                expected_path = f"iqfeed/forex/{base_symbol}/ticks/YYYY-MM-DD"
            else:
                expected_path = f"iqfeed/{category}/{base_symbol}/ticks/YYYY-MM-DD"

            st.code(expected_path, language=None)

        # Show detailed parsing breakdown
        with st.expander("🔍 Detailed Parsing Breakdown"):
            st.json(parsed_info)

    else:
        st.error(f"❌ Symbol parsing failed: {parse_result['error']}")
        st.write(f"**Symbol:** {parse_result['symbol']}")


def display_symbol_input_widget() -> str:
    """
    Display symbol input widget with examples and validation

    Returns:
        The entered symbol string
    """
    st.subheader("Enter Symbol to Parse")

    # Example symbols for different categories
    examples = {
        'Stocks': ['AAPL', 'MSFT', 'TSLA', 'GOOGL'],
        'Futures': ['@ES#', '@NQ#', '@CL#', '@GC#'],
        'Forex': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
        'Options': ['AAPL230120C00150000', 'SPY231215P00400000'],
        'DTN Indicators': ['$TICK', '$TRIN', '$VIX', '$ADVN']
    }

    # Create example buttons
    st.write("**Quick Examples:**")
    cols = st.columns(5)

    selected_example = None
    for i, (category, symbols) in enumerate(examples.items()):
        with cols[i]:
            st.write(f"*{category}*")
            for symbol in symbols:
                if st.button(symbol, key=f"example_{symbol}"):
                    selected_example = symbol

    # Main input field
    if selected_example:
        symbol = st.text_input(
            "Symbol:",
            value=selected_example,
            placeholder="Enter any symbol (e.g., AAPL, @ES#, EUR/USD)",
            help="Enter any financial symbol. The parser will automatically categorize it."
        )
    else:
        symbol = st.text_input(
            "Symbol:",
            placeholder="Enter any symbol (e.g., AAPL, @ES#, EUR/USD)",
            help="Enter any financial symbol. The parser will automatically categorize it."
        )

    return symbol.strip().upper() if symbol else ""


def display_symbol_categories_info():
    """Display information about supported symbol categories"""
    with st.expander("ℹ️ Supported Symbol Categories"):
        categories_info = {
            "Equities": {
                "description": "Common stocks and ETFs",
                "examples": ["AAPL", "MSFT", "SPY", "QQQ"],
                "storage": "iqfeed/equity/stocks/"
            },
            "Futures": {
                "description": "Futures contracts",
                "examples": ["@ES#", "@NQ#", "@CL#", "@GC#"],
                "storage": "iqfeed/futures/"
            },
            "Forex": {
                "description": "Currency pairs",
                "examples": ["EUR/USD", "GBP/USD", "USD/JPY"],
                "storage": "iqfeed/forex/"
            },
            "Options": {
                "description": "Options contracts",
                "examples": ["AAPL230120C00150000", "SPY231215P00400000"],
                "storage": "iqfeed/options/"
            },
            "DTN Indicators": {
                "description": "Market indicators and calculated values",
                "examples": ["$TICK", "$TRIN", "$VIX", "$ADVN"],
                "storage": "dtn/indicators/"
            }
        }

        for category, info in categories_info.items():
            st.write(f"**{category}**")
            st.write(f"- {info['description']}")
            st.write(f"- Examples: {', '.join(info['examples'])}")
            st.write(f"- Storage: `{info['storage']}`")
            st.write("")


def create_symbol_testing_section():
    """
    Create a complete symbol testing section with input and results

    Returns:
        Tuple of (symbol, parse_button_clicked)
    """
    st.header("🔍 Symbol Parser & Categorizer")

    display_symbol_categories_info()

    symbol = display_symbol_input_widget()

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        parse_button = st.button(
            "Parse Symbol",
            disabled=not symbol,
            help="Parse the symbol to see its category and routing information"
        )

    with col2:
        clear_button = st.button("Clear")

    if clear_button:
        st.rerun()

    return symbol, parse_button