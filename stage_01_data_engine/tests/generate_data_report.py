"""
Data Collection Report Generator
Creates comprehensive HTML and console reports of collected data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Import our data engine components
from stage_01_data_engine.core.data_engine import DataEngine
from stage_01_data_engine.collectors.tick_collector import TickCollector
from stage_01_data_engine.collectors.dtn_indicators_collector import DTNIndicatorCollector
from stage_01_data_engine.storage.tick_store import TickStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataReportGenerator:
    """Generate comprehensive data collection and quality reports"""

    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'TSLA']
        self.report_data = {}
        self.report_timestamp = datetime.now()

        logger.info("Initialized data report generator")

    def collect_comprehensive_data(self) -> dict:
        """Collect all available data for reporting"""
        logger.info("=== Collecting Comprehensive Data for Report ===")

        try:
            engine = DataEngine()
            report_data = {
                'collection_timestamp': self.report_timestamp.isoformat(),
                'symbols_tested': self.test_symbols,
                'data_sources': {},
                'performance_metrics': {},
                'data_samples': {},
                'quality_metrics': {}
            }

            # Collect market snapshot
            logger.info("Collecting market snapshot...")
            snapshot = engine.collect_market_snapshot(self.test_symbols)

            for data_type, df in snapshot.items():
                if df is not None and not df.empty:
                    # Store summary statistics
                    report_data['data_sources'][data_type] = {
                        'record_count': len(df),
                        'columns': list(df.columns),
                        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                        'date_range': {
                            'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                            'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
                        }
                    }

                    # Store sample data (first 5 rows)
                    sample_data = df.head(5).to_dict('records')
                    # Convert any datetime objects to strings for JSON serialization
                    for record in sample_data:
                        for key, value in record.items():
                            if isinstance(value, (datetime, pd.Timestamp)):
                                record[key] = value.isoformat()
                            elif isinstance(value, (np.int64, np.float64)):
                                record[key] = value.item()

                    report_data['data_samples'][data_type] = sample_data

                    logger.info(f"✅ {data_type}: {len(df)} records collected")

            # Collect individual collector statistics
            logger.info("Collecting collector statistics...")

            # Tick collector stats
            tick_collector = TickCollector()
            tick_stats = tick_collector.get_stats()
            report_data['performance_metrics']['tick_collector'] = tick_stats

            # DTN collector stats
            dtn_collector = DTNIndicatorCollector()
            available_indicators = dtn_collector.list_available_indicators()
            indicator_groups = dtn_collector.get_indicator_groups()

            report_data['performance_metrics']['dtn_collector'] = {
                'available_indicators': len(available_indicators),
                'indicator_groups': indicator_groups,
                'sample_indicators': available_indicators[:10]  # First 10 for sample
            }

            # Engine stats
            engine_stats = engine.get_stats()
            report_data['performance_metrics']['data_engine'] = engine_stats

            # Market regime analysis
            regime = engine.get_market_regime()
            report_data['quality_metrics']['market_regime'] = regime

            # Test storage system
            logger.info("Testing storage system...")
            tick_store = TickStore()
            storage_stats = tick_store.get_storage_stats()
            report_data['performance_metrics']['storage'] = storage_stats

            self.report_data = report_data
            logger.info("✅ Comprehensive data collection completed")
            return report_data

        except Exception as e:
            logger.error(f"❌ Data collection failed: {e}")
            return {}

    def generate_console_report(self):
        """Generate detailed console report"""
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE DATA COLLECTION REPORT")
        logger.info("="*80)

        if not self.report_data:
            logger.error("❌ No report data available")
            return

        # Header information
        logger.info(f"\n📅 Report Generated: {self.report_timestamp}")
        logger.info(f"🎯 Symbols Tested: {', '.join(self.report_data.get('symbols_tested', []))}")

        # Data sources summary
        logger.info(f"\n📊 DATA SOURCES SUMMARY")
        logger.info("-" * 40)

        data_sources = self.report_data.get('data_sources', {})
        total_records = 0

        for source, info in data_sources.items():
            record_count = info.get('record_count', 0)
            total_records += record_count
            memory_mb = info.get('memory_usage_mb', 0)

            logger.info(f"✅ {source.upper()}:")
            logger.info(f"   Records: {record_count:,}")
            logger.info(f"   Memory: {memory_mb:.2f} MB")
            logger.info(f"   Columns: {len(info.get('columns', []))}")

            if info.get('date_range', {}).get('start'):
                logger.info(f"   Date Range: {info['date_range']['start']} to {info['date_range']['end']}")

        logger.info(f"\n🎯 TOTAL RECORDS: {total_records:,}")

        # Performance metrics
        logger.info(f"\n⚡ PERFORMANCE METRICS")
        logger.info("-" * 40)

        perf_metrics = self.report_data.get('performance_metrics', {})

        if 'data_engine' in perf_metrics:
            engine_stats = perf_metrics['data_engine']
            logger.info(f"✅ DATA ENGINE:")
            logger.info(f"   Collections Today: {engine_stats.get('collections_today', 0)}")
            logger.info(f"   Data Points Collected: {engine_stats.get('data_points_collected', 0):,}")
            logger.info(f"   Active Symbols: {engine_stats.get('active_symbol_count', 0)}")
            logger.info(f"   Errors Today: {engine_stats.get('errors_today', 0)}")

        if 'dtn_collector' in perf_metrics:
            dtn_stats = perf_metrics['dtn_collector']
            logger.info(f"✅ DTN INDICATORS:")
            logger.info(f"   Available Indicators: {dtn_stats.get('available_indicators', 0)}")
            logger.info(f"   Indicator Groups: {len(dtn_stats.get('indicator_groups', []))}")

        # Quality metrics
        logger.info(f"\n🎯 QUALITY METRICS")
        logger.info("-" * 40)

        quality_metrics = self.report_data.get('quality_metrics', {})

        if 'market_regime' in quality_metrics:
            regime = quality_metrics['market_regime']
            logger.info(f"✅ MARKET REGIME:")
            for key, value in regime.items():
                logger.info(f"   {key.title()}: {value}")

        # Data samples
        logger.info(f"\n📋 DATA SAMPLES")
        logger.info("-" * 40)

        data_samples = self.report_data.get('data_samples', {})

        for data_type, samples in data_samples.items():
            logger.info(f"\n✅ {data_type.upper()} SAMPLE:")
            if samples and len(samples) > 0:
                sample = samples[0]  # Show first sample
                for key, value in sample.items():
                    if isinstance(value, float):
                        logger.info(f"   {key}: {value:.4f}")
                    else:
                        logger.info(f"   {key}: {value}")
            else:
                logger.info("   No sample data available")

    def generate_html_report(self, output_file: str = "data_collection_report.html"):
        """Generate HTML report"""
        logger.info(f"Generating HTML report: {output_file}")

        if not self.report_data:
            logger.error("❌ No report data available for HTML generation")
            return False

        try:
            html_content = self._create_html_content()

            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"✅ HTML report generated: {output_path.absolute()}")
            return True

        except Exception as e:
            logger.error(f"❌ HTML report generation failed: {e}")
            return False

    def _create_html_content(self) -> str:
        """Create HTML content for the report"""
        data_sources = self.report_data.get('data_sources', {})
        performance = self.report_data.get('performance_metrics', {})
        quality = self.report_data.get('quality_metrics', {})
        samples = self.report_data.get('data_samples', {})

        # Calculate summary statistics
        total_records = sum(info.get('record_count', 0) for info in data_sources.values())
        total_memory = sum(info.get('memory_usage_mb', 0) for info in data_sources.values())

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stage 1 Data Engine Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h3 {{ color: #7f8c8d; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .summary-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
                .summary-card h3 {{ margin-top: 0; color: #2c3e50; }}
                .summary-card .number {{ font-size: 2em; font-weight: bold; color: #3498db; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .data-table th {{ background-color: #3498db; color: white; }}
                .data-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .status-good {{ color: #27ae60; font-weight: bold; }}
                .status-warning {{ color: #f39c12; font-weight: bold; }}
                .status-error {{ color: #e74c3c; font-weight: bold; }}
                .json-data {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: monospace; }}
                .timestamp {{ color: #7f8c8d; font-style: italic; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Stage 1 Data Engine Report</h1>
                <p class="timestamp">Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>

                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Total Records</h3>
                        <div class="number">{total_records:,}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Data Sources</h3>
                        <div class="number">{len(data_sources)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Memory Usage</h3>
                        <div class="number">{total_memory:.2f} MB</div>
                    </div>
                    <div class="summary-card">
                        <h3>Symbols Tested</h3>
                        <div class="number">{len(self.report_data.get('symbols_tested', []))}</div>
                    </div>
                </div>

                <h2>📊 Data Sources</h2>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Data Source</th>
                            <th>Records</th>
                            <th>Columns</th>
                            <th>Memory (MB)</th>
                            <th>Date Range</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # Add data sources rows
        for source, info in data_sources.items():
            date_range = "N/A"
            if info.get('date_range', {}).get('start'):
                start = info['date_range']['start'][:19]  # Remove microseconds
                end = info['date_range']['end'][:19]
                date_range = f"{start} to {end}"

            html += f"""
                        <tr>
                            <td><strong>{source.upper()}</strong></td>
                            <td>{info.get('record_count', 0):,}</td>
                            <td>{len(info.get('columns', []))}</td>
                            <td>{info.get('memory_usage_mb', 0):.2f}</td>
                            <td>{date_range}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>

                <h2>⚡ Performance Metrics</h2>
        """

        # Add performance metrics
        if 'data_engine' in performance:
            engine_stats = performance['data_engine']
            html += f"""
                <h3>Data Engine Statistics</h3>
                <table class="data-table">
                    <tr><td>Collections Today</td><td>{engine_stats.get('collections_today', 0)}</td></tr>
                    <tr><td>Data Points Collected</td><td>{engine_stats.get('data_points_collected', 0):,}</td></tr>
                    <tr><td>Active Symbols</td><td>{engine_stats.get('active_symbol_count', 0)}</td></tr>
                    <tr><td>Errors Today</td><td class="{'status-good' if engine_stats.get('errors_today', 0) == 0 else 'status-error'}">{engine_stats.get('errors_today', 0)}</td></tr>
                </table>
            """

        # Add market regime
        if 'market_regime' in quality:
            regime = quality['market_regime']
            html += f"""
                <h2>🎯 Market Regime Analysis</h2>
                <table class="data-table">
            """
            for key, value in regime.items():
                html += f"<tr><td>{key.title()}</td><td><strong>{value}</strong></td></tr>"
            html += "</table>"

        # Add data samples
        html += "<h2>📋 Data Samples</h2>"
        for data_type, sample_list in samples.items():
            if sample_list and len(sample_list) > 0:
                html += f"<h3>{data_type.upper()} Sample</h3>"
                html += f'<div class="json-data">{json.dumps(sample_list[0], indent=2)}</div>'

        html += """
            </div>
        </body>
        </html>
        """

        return html

    def generate_json_report(self, output_file: str = "data_collection_report.json"):
        """Generate JSON report for programmatic access"""
        logger.info(f"Generating JSON report: {output_file}")

        if not self.report_data:
            logger.error("❌ No report data available for JSON generation")
            return False

        try:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.report_data, f, indent=2, default=str)

            logger.info(f"✅ JSON report generated: {output_path.absolute()}")
            return True

        except Exception as e:
            logger.error(f"❌ JSON report generation failed: {e}")
            return False

    def run_complete_report_generation(self):
        """Run complete report generation process"""
        logger.info("🚀 Starting Complete Data Report Generation")
        logger.info("=" * 60)

        # Collect data
        logger.info("Step 1: Collecting comprehensive data...")
        data_collected = self.collect_comprehensive_data()

        if not data_collected:
            logger.error("❌ Data collection failed - cannot generate reports")
            return False

        # Generate console report
        logger.info("Step 2: Generating console report...")
        self.generate_console_report()

        # Generate HTML report
        logger.info("Step 3: Generating HTML report...")
        html_success = self.generate_html_report()

        # Generate JSON report
        logger.info("Step 4: Generating JSON report...")
        json_success = self.generate_json_report()

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("🎯 REPORT GENERATION SUMMARY")
        logger.info(f"✅ Console Report: Generated")
        logger.info(f"{'✅' if html_success else '❌'} HTML Report: {'Generated' if html_success else 'Failed'}")
        logger.info(f"{'✅' if json_success else '❌'} JSON Report: {'Generated' if json_success else 'Failed'}")

        if html_success and json_success:
            logger.info("🎉 All reports generated successfully!")
            return True
        else:
            logger.warning("⚠️ Some reports failed to generate")
            return False


def main():
    """Run the data report generation"""
    try:
        # Ensure output directory exists
        Path("./reports").mkdir(exist_ok=True)

        # Generate reports
        report_generator = DataReportGenerator()
        success = report_generator.run_complete_report_generation()

        if success:
            print("\n🚀 Data report generation successful!")
            print("📊 Check the generated HTML and JSON reports!")
            print("✨ Data collection verification complete!")
        else:
            print("\n⚠️ Report generation had issues")
            print("📋 Check logs for details")

        return success

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False


if __name__ == "__main__":
    main()