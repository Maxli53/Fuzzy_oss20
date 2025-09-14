"""
Stage 1 Data Engine - Testing GUI
Real-time testing interface for flexible storage system
"""
import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our components
from gui.data_interface import GUIDataInterface
from gui.components.symbol_parser_widget import (
    create_symbol_testing_section,
    display_symbol_parse_results
)
from gui.components.storage_viewer import (
    create_storage_inspector_section,
    display_round_trip_test_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page configuration
st.set_page_config(
    page_title="Stage 1 Data Engine - Testing GUI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'data_interface' not in st.session_state:
        st.session_state.data_interface = None
    if 'last_parse_result' not in st.session_state:
        st.session_state.last_parse_result = None
    if 'last_fetch_result' not in st.session_state:
        st.session_state.last_fetch_result = None
    if 'last_store_result' not in st.session_state:
        st.session_state.last_store_result = None

@st.cache_resource
def get_data_interface():
    """Get cached data interface instance"""
    return GUIDataInterface()

def main():
    """Main Streamlit application"""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🏦 Stage 1 Data Engine")
        st.write("Real-time testing interface for flexible storage system")

        # Initialize data interface
        if st.session_state.data_interface is None:
            with st.spinner("Initializing DataEngine..."):
                st.session_state.data_interface = get_data_interface()

        data_interface = st.session_state.data_interface

        # Connection status
        st.subheader("🔌 Connection Status")
        connection_status = data_interface.get_connection_status()

        status_icons = {
            'data_engine': 'DataEngine',
            'iqfeed': 'IQFeed',
            'polygon': 'Polygon',
            'storage': 'Storage'
        }

        for key, label in status_icons.items():
            if connection_status.get(key, False):
                st.success(f"✅ {label}")
            else:
                st.error(f"❌ {label}")

        st.divider()

        # Quick actions
        st.subheader("⚡ Quick Actions")

        if st.button("🔄 Refresh Connections", use_container_width=True):
            st.session_state.data_interface = None
            st.rerun()

        if st.button("📊 View Storage Stats", use_container_width=True):
            st.session_state.active_tab = "Storage Inspector"
            st.rerun()

    # Main content
    st.title("🏦 Stage 1 Data Engine - Testing GUI")
    st.write("Test your flexible storage system with real market data")

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Symbol Testing",
        "📂 Storage Inspector",
        "🔄 Round-Trip Testing",
        "📈 Live Data Testing"
    ])

    with tab1:
        symbol_testing_tab(data_interface)

    with tab2:
        storage_inspector_tab(data_interface)

    with tab3:
        round_trip_testing_tab(data_interface)

    with tab4:
        live_data_testing_tab(data_interface)

def symbol_testing_tab(data_interface):
    """Symbol parsing and categorization testing"""
    st.header("🔍 Symbol Testing & Parsing")

    symbol, parse_button = create_symbol_testing_section()

    if parse_button and symbol:
        with st.spinner(f"Parsing symbol {symbol}..."):
            parse_result = data_interface.parse_symbol(symbol)
            st.session_state.last_parse_result = parse_result

    # Display parse results
    if st.session_state.last_parse_result:
        st.divider()
        display_symbol_parse_results(st.session_state.last_parse_result)

def storage_inspector_tab(data_interface):
    """Storage system inspection and browsing"""
    create_storage_inspector_section(data_interface)

def round_trip_testing_tab(data_interface):
    """Complete round-trip testing"""
    st.header("🔄 Round-Trip Testing")
    st.write("Complete test: Parse → Fetch → Store → Retrieve → Verify")

    col1, col2 = st.columns([2, 1])

    with col1:
        test_symbol = st.text_input(
            "Symbol to test:",
            placeholder="Enter symbol for complete round-trip test",
            help="This will test the entire pipeline with real data"
        )

    with col2:
        st.write("")  # Spacing
        run_test = st.button(
            "🚀 Run Complete Test",
            disabled=not test_symbol,
            use_container_width=True
        )

    if run_test and test_symbol:
        st.divider()

        with st.spinner(f"Running complete round-trip test for {test_symbol}..."):
            test_result = data_interface.perform_round_trip_test(test_symbol.strip().upper())

        display_round_trip_test_results(test_result)

def live_data_testing_tab(data_interface):
    """Live data fetching and storage testing"""
    st.header("📈 Live Data Testing")
    st.write("Fetch real market data and test storage in real-time")

    # Symbol input
    col1, col2 = st.columns([2, 1])

    with col1:
        live_symbol = st.text_input(
            "Symbol for live data:",
            placeholder="Enter symbol to fetch real data",
            help="Fetch real-time or recent market data"
        )

    with col2:
        st.write("")  # Spacing
        fetch_button = st.button(
            "📡 Fetch Real Data",
            disabled=not live_symbol,
            use_container_width=True
        )

    # Data fetching options
    with st.expander("🛠️ Fetch Options"):
        col1, col2, col3 = st.columns(3)

        with col1:
            data_type = st.selectbox(
                "Data Type:",
                options=['ticks', 'bars', 'quotes'],
                index=0,
                help="Type of market data to fetch"
            )

        with col2:
            lookback_days = st.number_input(
                "Lookback Days:",
                min_value=1,
                max_value=30,
                value=1,
                help="Number of days to look back"
            )

        with col3:
            max_records = st.number_input(
                "Max Records:",
                min_value=100,
                max_value=10000,
                value=1000,
                help="Maximum records to fetch"
            )

    if fetch_button and live_symbol:
        st.divider()

        symbol = live_symbol.strip().upper()

        # Step 1: Parse symbol
        st.subheader(f"Step 1: Parsing {symbol}")
        with st.spinner("Parsing symbol..."):
            parse_result = data_interface.parse_symbol(symbol)

        if parse_result['success']:
            st.success("✅ Symbol parsed successfully")
            st.json(parse_result['parsed_info'])
        else:
            st.error(f"❌ Symbol parsing failed: {parse_result['error']}")
            return

        # Step 2: Fetch real data
        st.subheader(f"Step 2: Fetching Real Data for {symbol}")
        with st.spinner(f"Fetching {data_type} data from market sources..."):
            fetch_result = data_interface.fetch_real_data(
                symbol=symbol,
                data_type=data_type,
                lookback_days=lookback_days,
                max_records=max_records
            )

        if fetch_result['success']:
            data = fetch_result['data']
            metadata = fetch_result['metadata']
            source = fetch_result['source']

            st.success(f"✅ Data fetched successfully from {source}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", metadata['records_count'])
            with col2:
                st.metric("Source", source)
            with col3:
                st.metric("Columns", len(metadata['columns']))

            # Display data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(data.head(20), use_container_width=True)

            # Step 3: Store data
            st.subheader(f"Step 3: Storing Data for {symbol}")
            store_button = st.button(f"💾 Store {len(data)} records")

            if store_button:
                with st.spinner("Storing data in flexible storage..."):
                    store_result = data_interface.store_data(
                        symbol=symbol,
                        data=data,
                        data_type=data_type
                    )

                if store_result['success']:
                    st.success("✅ Data stored successfully")
                    st.write(f"**Backend:** {store_result['backend_used']}")
                    st.write(f"**Storage Location:** {store_result['storage_location']}")

                    # Step 4: Verify storage
                    st.subheader(f"Step 4: Verifying Storage for {symbol}")
                    with st.spinner("Retrieving stored data for verification..."):
                        verify_result = data_interface.retrieve_stored_data(symbol)

                    if verify_result['success']:
                        retrieved_data = verify_result['data']
                        st.success(f"✅ Data verified - Retrieved {len(retrieved_data)} records")

                        # Compare original vs retrieved
                        comparison = pd.DataFrame({
                            'Metric': ['Record Count', 'Column Count', 'Index Start', 'Index End'],
                            'Original': [
                                len(data),
                                len(data.columns),
                                str(data.index.min()),
                                str(data.index.max())
                            ],
                            'Retrieved': [
                                len(retrieved_data),
                                len(retrieved_data.columns),
                                str(retrieved_data.index.min()),
                                str(retrieved_data.index.max())
                            ]
                        })

                        st.dataframe(comparison, use_container_width=True)

                        # Show if data matches
                        if len(data) == len(retrieved_data) and list(data.columns) == list(retrieved_data.columns):
                            st.success("🎉 Perfect match! Storage system working correctly.")
                        else:
                            st.warning("⚠️ Some differences detected between original and retrieved data.")

                    else:
                        st.error(f"❌ Data verification failed: {verify_result['error']}")

                else:
                    st.error(f"❌ Data storage failed: {store_result['error']}")

        else:
            st.error(f"❌ Data fetch failed: {fetch_result['error']}")
            st.write("**Possible causes:**")
            st.write("- Market is closed")
            st.write("- Symbol not found")
            st.write("- Data source not connected")
            st.write("- Network issues")

    # Footer
    st.divider()
    st.write(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()