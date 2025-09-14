"""
IQFeed Charts Style GUI - Professional trading interface
Replicates the IQFeed Charts interface design and functionality
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, time as dt_time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gui.data_interface import GUIDataInterface

# Page configuration
st.set_page_config(
    page_title="IQFeed Charts - Professional Trading Interface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def create_professional_controls():
    """Create IQFeed Charts style control panel"""

    # Top control bar
    col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 1, 1, 2])

    with col1:
        symbol = st.text_input("Symbol", value="AAPL", label_visibility="collapsed", placeholder="Enter symbol...")

    with col2:
        bar_type = st.selectbox(
            "Data Type",
            options=[
                # Time-based
                'ticks', '1s', '5s', '1m', '5m', '15m', '1h', 'daily',
                # Advanced bars
                'tick_50', 'tick_100', 'tick_200',
                'volume_1000', 'volume_5000', 'volume_10000',
                'dollar_10000', 'dollar_50000', 'dollar_100000',
                'imbalance', 'volatility', 'range', 'renko'
            ],
            index=3,  # Default to '1m'
            label_visibility="collapsed"
        )

    with col3:
        chart_style = st.selectbox(
            "Chart Style",
            options=['OHLC', 'Candlestick', 'Line', 'Area'],
            index=1,  # Default to Candlestick
            label_visibility="collapsed"
        )

    with col4:
        trading_days = st.number_input("Days", min_value=1, max_value=8, value=1, label_visibility="collapsed")

    with col5:
        market_hours = st.checkbox("Market Hours Only", value=True)

    with col6:
        col6a, col6b = st.columns(2)
        with col6a:
            include_premarket = st.checkbox("Pre-market", value=False)
        with col6b:
            include_afterhours = st.checkbox("After-hours", value=False)

    return symbol, bar_type, chart_style, trading_days, market_hours, include_premarket, include_afterhours

def get_bar_parameters(bar_type):
    """Get advanced bar type parameters"""
    params = {}

    if bar_type.startswith('tick_'):
        params['tick_threshold'] = int(bar_type.split('_')[1])
    elif bar_type.startswith('volume_'):
        params['volume_threshold'] = int(bar_type.split('_')[1])
    elif bar_type.startswith('dollar_'):
        params['dollar_threshold'] = int(bar_type.split('_')[1])
    elif bar_type == 'imbalance':
        params['imbalance_threshold'] = 0.2
    elif bar_type == 'volatility':
        params['volatility_threshold'] = 0.02
    elif bar_type == 'range':
        params['range_threshold'] = 1.0
    elif bar_type == 'renko':
        params['renko_brick_size'] = 0.5

    return params

def create_professional_chart(data, symbol, bar_type, chart_style):
    """Create professional trading chart matching IQFeed Charts style"""

    if data is None or data.empty:
        st.error("No data available for charting")
        return

    # Create subplots: Price chart + Volume chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} - {bar_type.upper()}', 'Volume'),
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        shared_xaxes=True
    )

    # Price chart
    if chart_style == 'Candlestick' and all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        fig.add_trace(
            go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=symbol,
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ),
            row=1, col=1
        )
    elif chart_style == 'OHLC' and all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        fig.add_trace(
            go.Ohlc(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=symbol
            ),
            row=1, col=1
        )
    else:
        # Line chart (fallback)
        price_col = 'close' if 'close' in data.columns else 'price'
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data[price_col],
                mode='lines',
                name=symbol,
                line=dict(color='#0066cc', width=2)
            ),
            row=1, col=1
        )

    # Volume chart
    if 'volume' in data.columns:
        fig.add_trace(
            go.Bar(
                x=data['timestamp'],
                y=data['volume'],
                name='Volume',
                marker_color='rgba(100, 100, 100, 0.6)'
            ),
            row=2, col=1
        )

    # Professional styling to match IQFeed Charts
    fig.update_layout(
        title=None,
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_dark',
        font=dict(family="Arial", size=10),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#2e2e2e'
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='#444444')
    fig.update_yaxes(showgrid=True, gridcolor='#444444')

    return fig

def display_data_status(data, symbol):
    """Display data status like IQFeed Charts"""
    if data is not None and not data.empty:
        total_points = len(data)
        displayed_points = min(total_points, 200)  # Limit display for performance

        st.caption(f"📊 {total_points} datapoints available. Displaying {displayed_points}.")

        # Show data range
        if 'timestamp' in data.columns:
            date_range = f"{data['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {data['timestamp'].max().strftime('%Y-%m-%d %H:%M')}"
            st.caption(f"🕒 Data range: {date_range}")

def main():
    """Main IQFeed Charts style interface"""

    # Professional header
    st.markdown("### 📈 IQFeed Charts - Professional Trading Interface")
    st.markdown("---")

    # Initialize data interface
    if 'data_interface' not in st.session_state:
        with st.spinner("Initializing trading data interface..."):
            st.session_state.data_interface = GUIDataInterface()

    data_interface = st.session_state.data_interface

    # Professional control panel
    symbol, bar_type, chart_style, trading_days, market_hours, include_premarket, include_afterhours = create_professional_controls()

    # Advanced parameters section (collapsible)
    with st.expander("⚙️ Advanced Parameters"):
        bar_params = get_bar_parameters(bar_type)

        if bar_type.startswith('tick_'):
            tick_threshold = st.slider("Tick Threshold", 10, 1000, bar_params.get('tick_threshold', 50))
            bar_params['tick_threshold'] = tick_threshold
        elif bar_type.startswith('volume_'):
            volume_threshold = st.number_input("Volume Threshold", 100, 100000, bar_params.get('volume_threshold', 1000))
            bar_params['volume_threshold'] = volume_threshold
        elif bar_type.startswith('dollar_'):
            dollar_threshold = st.number_input("Dollar Threshold", 1000, 1000000, bar_params.get('dollar_threshold', 10000))
            bar_params['dollar_threshold'] = dollar_threshold

    # Fetch button
    if st.button(f"🔄 Fetch {bar_type.upper()} Data", type="primary"):

        with st.spinner(f"Fetching {bar_type} data for {symbol}..."):

            # Prepare fetch parameters
            fetch_params = {
                'market_hours_only': market_hours,
                'include_premarket': include_premarket,
                'include_afterhours': include_afterhours,
                'lookback_days': trading_days,
                'max_records': 10000 if bar_type == 'ticks' else 2000
            }
            fetch_params.update(bar_params)

            # Fetch data
            if bar_type == 'ticks':
                fetch_result = data_interface.fetch_real_data(
                    symbol=symbol,
                    data_type='ticks',
                    **fetch_params
                )
            else:
                # For bars, pass through our collector
                try:
                    collector_data = data_interface.data_engine.iqfeed_collector.collect_bars(
                        symbols=[symbol],
                        bar_type=bar_type,
                        **fetch_params
                    )

                    if collector_data is not None and not collector_data.empty:
                        fetch_result = {
                            'success': True,
                            'data': collector_data,
                            'source': 'IQFeed',
                            'metadata': {
                                'records_count': len(collector_data),
                                'date_range': f"{collector_data['timestamp'].min()} to {collector_data['timestamp'].max()}"
                            }
                        }
                    else:
                        fetch_result = {'success': False, 'error': 'No data returned'}

                except Exception as e:
                    fetch_result = {'success': False, 'error': str(e)}

        if fetch_result['success']:
            data = fetch_result['data']

            # Display data status (IQFeed Charts style)
            display_data_status(data, symbol)

            # Create professional chart
            fig = create_professional_chart(data, symbol, bar_type, chart_style)

            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Data preview table
            with st.expander("📋 Raw Data"):
                st.dataframe(data.head(50), use_container_width=True)

        else:
            st.error(f"❌ Failed to fetch data: {fetch_result['error']}")

if __name__ == "__main__":
    main()