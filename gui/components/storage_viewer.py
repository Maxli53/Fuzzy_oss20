"""
Storage Viewer Component for Streamlit
Displays stored data and storage system information
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

def display_storage_statistics(stats_result: Dict[str, Any]):
    """
    Display storage system statistics

    Args:
        stats_result: Result from GUIDataInterface.get_storage_statistics()
    """
    if not stats_result.get('success', False):
        st.error(f"❌ Failed to load storage statistics: {stats_result.get('error', 'Unknown error')}")
        return

    routing_stats = stats_result.get('routing_stats', {})
    engine_stats = stats_result.get('engine_stats', {})
    connection_status = stats_result.get('connection_status', {})

    # Connection Status
    st.subheader("🔌 Connection Status")
    cols = st.columns(4)

    statuses = [
        ('DataEngine', connection_status.get('data_engine', False)),
        ('IQFeed', connection_status.get('iqfeed', False)),
        ('Polygon', connection_status.get('polygon', False)),
        ('Storage', connection_status.get('storage', False))
    ]

    for i, (component, status) in enumerate(statuses):
        with cols[i]:
            if status:
                st.success(f"✅ {component}")
            else:
                st.error(f"❌ {component}")

    # Storage Routing Statistics
    st.subheader("📊 Storage Routing Statistics")

    col1, col2 = st.columns(2)

    with col1:
        # Routing metrics
        routing_metrics = {
            'Total Routes': routing_stats.get('total_routes', 0),
            'Flexible Routes': routing_stats.get('flexible_routes', 0),
            'Legacy Routes': routing_stats.get('legacy_routes', 0),
            'Routing Errors': routing_stats.get('routing_errors', 0)
        }

        for metric, value in routing_metrics.items():
            st.metric(label=metric, value=value)

    with col2:
        # Symbol categories pie chart
        categories = routing_stats.get('symbol_categories', {})
        if categories:
            fig = px.pie(
                values=list(categories.values()),
                names=list(categories.keys()),
                title="Symbols by Category"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Engine Statistics
    if engine_stats:
        st.subheader("⚙️ DataEngine Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Collections Today", engine_stats.get('collections_today', 0))
            st.metric("Active Symbols", len(engine_stats.get('active_symbols', set())))

        with col2:
            st.metric("Data Points Collected", engine_stats.get('data_points_collected', 0))
            st.metric("Discovered Symbols", engine_stats.get('discovered_symbols_count', 0))

        with col3:
            st.metric("Fetch Requests", engine_stats.get('fetch_requests', 0))
            st.metric("Errors Today", engine_stats.get('errors_today', 0))

        # Last collection time
        last_collection = engine_stats.get('last_collection')
        if last_collection:
            st.write(f"**Last Collection:** {last_collection}")

    # Performance metrics
    performance_metrics = routing_stats.get('performance_metrics', {})
    if performance_metrics:
        st.subheader("⚡ Performance Metrics")

        perf_data = []
        for backend, metrics in performance_metrics.items():
            perf_data.append({
                'Backend': backend.title(),
                'Total Operations': metrics.get('total_operations', 0),
                'Successful Operations': metrics.get('successful_operations', 0),
                'Average Time (s)': round(metrics.get('avg_time', 0), 3),
                'Success Rate (%)': round(
                    (metrics.get('successful_operations', 0) /
                     max(metrics.get('total_operations', 1), 1)) * 100, 1
                )
            })

        if perf_data:
            df = pd.DataFrame(perf_data)
            st.dataframe(df, use_container_width=True)


def display_stored_symbols_browser(symbols_result: Dict[str, Any]):
    """
    Display browser for stored symbols

    Args:
        symbols_result: Result from GUIDataInterface.discover_stored_symbols()
    """
    if not symbols_result.get('success', False):
        st.error(f"❌ Failed to load stored symbols: {symbols_result.get('error', 'Unknown error')}")
        return

    symbols = symbols_result.get('symbols', [])
    metadata = symbols_result.get('metadata', {})

    st.subheader(f"📂 Stored Symbols ({metadata.get('total_symbols', 0)} found)")

    if not symbols:
        st.info("No symbols found in storage. Try storing some data first!")
        return

    # Category filter
    categories = metadata.get('categories', {})
    if categories:
        selected_categories = st.multiselect(
            "Filter by Category:",
            options=list(categories.keys()),
            default=list(categories.keys()),
            help="Select which symbol categories to display"
        )

        # Filter symbols by selected categories
        filtered_symbols = [
            s for s in symbols
            if s.get('category', 'unknown') in selected_categories
        ]
    else:
        filtered_symbols = symbols

    # Display symbols in a table
    if filtered_symbols:
        symbol_data = []
        for symbol_info in filtered_symbols:
            symbol_data.append({
                'Symbol': symbol_info.get('symbol', 'Unknown'),
                'Category': symbol_info.get('category', 'unknown').title(),
                'Backend': symbol_info.get('backend', 'unknown').title(),
                'Data Types': ', '.join(symbol_info.get('data_types', [])) if symbol_info.get('data_types') else 'N/A',
                'Last Updated': symbol_info.get('last_updated', 'Unknown')
            })

        df = pd.DataFrame(symbol_data)
        st.dataframe(df, use_container_width=True)

        # Symbol selector for detailed view
        symbol_names = [s.get('symbol', 'Unknown') for s in filtered_symbols]
        selected_symbol = st.selectbox(
            "Select symbol for detailed view:",
            options=[''] + symbol_names,
            help="Choose a symbol to view its detailed information and data"
        )

        if selected_symbol:
            return selected_symbol

    return None


def display_data_visualization(data_result: Dict[str, Any], symbol: str):
    """
    Display retrieved data with interactive charts

    Args:
        data_result: Result from GUIDataInterface.retrieve_stored_data()
        symbol: Symbol being displayed
    """
    if not data_result.get('success', False):
        st.error(f"❌ Failed to load data for {symbol}: {data_result.get('error', 'Unknown error')}")
        return

    data = data_result.get('data')
    metadata = data_result.get('metadata', {})

    if data is None or data.empty:
        st.warning(f"No data found for symbol {symbol}")
        return

    st.subheader(f"📈 Data for {symbol}")

    # Data information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records", metadata.get('records_retrieved', len(data)))
    with col2:
        st.metric("Columns", len(data.columns))
    with col3:
        st.write(f"**Date Range:** {metadata.get('date_range', 'Unknown')}")

    # Display raw data table
    with st.expander(f"📋 Raw Data Preview ({len(data)} records)"):
        st.dataframe(data.head(100), use_container_width=True)

    # Create visualization if data has appropriate columns
    try:
        if not data.index.empty:
            # Check for common financial data columns
            price_cols = [col for col in data.columns if col.lower() in ['price', 'close', 'last', 'mid']]
            volume_cols = [col for col in data.columns if 'volume' in col.lower() or 'size' in col.lower()]

            if price_cols:
                st.subheader("📊 Price Chart")

                fig = go.Figure()

                # Add price line
                price_col = price_cols[0]
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[price_col],
                    mode='lines',
                    name=f'{symbol} {price_col}',
                    line=dict(color='blue', width=1)
                ))

                fig.update_layout(
                    title=f'{symbol} - {price_col.title()} Over Time',
                    xaxis_title='Time',
                    yaxis_title=price_col.title(),
                    height=400,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            # Volume chart if available
            if volume_cols:
                st.subheader("📊 Volume Chart")

                volume_col = volume_cols[0]
                fig_volume = px.bar(
                    data.reset_index(),
                    x=data.index,
                    y=volume_col,
                    title=f'{symbol} - {volume_col.title()} Over Time'
                )
                fig_volume.update_layout(height=300)
                st.plotly_chart(fig_volume, use_container_width=True)

            # Summary statistics
            st.subheader("📊 Summary Statistics")
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(data[numeric_cols].describe(), use_container_width=True)

    except Exception as e:
        st.warning(f"Could not create visualization: {str(e)}")
        st.write("Displaying data table instead:")
        st.dataframe(data, use_container_width=True)


def create_storage_inspector_section(data_interface):
    """
    Create complete storage inspector section

    Args:
        data_interface: GUIDataInterface instance
    """
    st.header("📂 Storage Inspector")

    # Load storage statistics
    stats_result = data_interface.get_storage_statistics()
    display_storage_statistics(stats_result)

    st.divider()

    # Load and display stored symbols
    symbols_result = data_interface.discover_stored_symbols()
    selected_symbol = display_stored_symbols_browser(symbols_result)

    if selected_symbol:
        st.divider()

        # Date range selector for data retrieval
        st.subheader(f"📅 Date Range for {selected_symbol}")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:",
                value=datetime.now() - timedelta(days=7),
                help="Start date for data retrieval"
            )
        with col2:
            end_date = st.date_input(
                "End Date:",
                value=datetime.now(),
                help="End date for data retrieval"
            )

        if st.button(f"Load Data for {selected_symbol}"):
            with st.spinner(f"Loading data for {selected_symbol}..."):
                data_result = data_interface.retrieve_stored_data(
                    symbol=selected_symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                display_data_visualization(data_result, selected_symbol)


def display_round_trip_test_results(test_result: Dict[str, Any]):
    """
    Display results of round-trip testing

    Args:
        test_result: Result from GUIDataInterface.perform_round_trip_test()
    """
    symbol = test_result.get('symbol', 'Unknown')
    overall_success = test_result.get('overall_success', False)
    steps = test_result.get('steps', {})

    st.subheader(f"🔄 Round-Trip Test Results for {symbol}")

    if overall_success:
        st.success("✅ All tests passed! Storage system is working correctly.")
    else:
        st.error(f"❌ Test failed: {test_result.get('error', 'Unknown error')}")

    # Display each test step
    step_names = {
        'parse': 'Symbol Parsing',
        'fetch': 'Data Fetching',
        'store': 'Data Storage',
        'retrieve': 'Data Retrieval',
        'verify': 'Data Verification'
    }

    for step_key, step_name in step_names.items():
        if step_key in steps:
            step_result = steps[step_key]
            with st.expander(f"{step_name} {'✅' if step_result.get('success', False) else '❌'}"):
                if step_result.get('success', False):
                    st.success(f"{step_name} completed successfully")

                    # Show step-specific details
                    if step_key == 'parse':
                        st.json(step_result.get('parsed_info', {}))
                    elif step_key == 'fetch':
                        metadata = step_result.get('metadata', {})
                        st.write(f"**Source:** {step_result.get('source', 'Unknown')}")
                        st.write(f"**Records:** {metadata.get('records_count', 0)}")
                        st.write(f"**Columns:** {metadata.get('columns', [])}")
                    elif step_key == 'store':
                        st.write(f"**Backend:** {step_result.get('backend_used', 'Unknown')}")
                        st.write(f"**Location:** {step_result.get('storage_location', 'Unknown')}")
                    elif step_key == 'retrieve':
                        metadata = step_result.get('metadata', {})
                        st.write(f"**Records Retrieved:** {metadata.get('records_retrieved', 0)}")
                    elif step_key == 'verify':
                        st.write(f"**Original Records:** {step_result.get('original_records', 0)}")
                        st.write(f"**Retrieved Records:** {step_result.get('retrieved_records', 0)}")
                        st.write(f"**Columns Match:** {'✅' if step_result.get('columns_match', False) else '❌'}")
                        st.write(f"**Data Types Match:** {'✅' if step_result.get('data_types_match', False) else '❌'}")
                else:
                    st.error(f"{step_name} failed: {step_result.get('error', 'Unknown error')}")

    # Test metadata
    test_timestamp = test_result.get('test_timestamp', 'Unknown')
    st.write(f"**Test Timestamp:** {test_timestamp}")