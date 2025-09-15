"""
Startup Progress Monitoring and System Assessment
Provides visual feedback during system startup and catch-up operations
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StartupProgressMonitor:
    """
    Provides comprehensive startup progress monitoring with visual feedback.
    """

    def __init__(self, state_file: str = "pipeline_state.json"):
        """Initialize the startup monitor"""
        self.state_file = Path(state_file)
        self.start_time = datetime.now()
        self.progress = {
            'data_completeness': 0.0,
            'metadata_coverage': 0.0,
            'system_readiness': 0.0,
            'estimated_time_remaining': None,
            'current_operation': 'Initializing...',
            'detailed_status': {}
        }
        self.component_status = {}
        self.issues = []
        self.recommendations = []
        self.stats = {
            'days_fetched': 0,
            'ticks_processed': 0,
            'metadata_computed': 0
        }

    def display_startup_dashboard(self):
        """Display real-time progress dashboard"""

        # Clear screen
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/Mac
            os.system('clear')

        print("=" * 80)
        print("                    FUZZY OSS20 - STARTUP PROGRESS")
        print("=" * 80)
        print(f"Started: {self.start_time.strftime('%H:%M:%S')}")
        print(f"Elapsed: {self.get_elapsed_time()}")
        print("-" * 80)

        # Overall Progress Bar
        self.print_progress_bar(
            "OVERALL SYSTEM READINESS",
            self.progress['system_readiness'],
            color='green' if self.progress['system_readiness'] > 80 else 'yellow'
        )

        print("\n" + "=" * 80)
        print("DETAILED PROGRESS:")
        print("-" * 80)

        # Component Progress
        components = [
            ("Data Ingestion", self.progress['data_completeness']),
            ("Metadata Computation", self.progress['metadata_coverage']),
            ("Index Building", self.progress.get('index_progress', 0)),
            ("Cache Warming", self.progress.get('cache_progress', 0))
        ]

        for name, value in components:
            self.print_progress_bar(name, value)

        # Current Operation
        print("\n" + "=" * 80)
        print(f"CURRENT OPERATION: {self.progress['current_operation']}")
        print("-" * 80)

        # Detailed Status
        if self.progress['detailed_status']:
            print("\nDETAILS:")
            for key, value in self.progress['detailed_status'].items():
                print(f"  • {key}: {value}")

        # ETA
        if self.progress['estimated_time_remaining']:
            print("\n" + "=" * 80)
            print(f"ESTIMATED TIME REMAINING: {self.progress['estimated_time_remaining']}")

        print("=" * 80)

    def print_progress_bar(self, label: str, percentage: float, width: int = 50, color: Optional[str] = None):
        """Print a visual progress bar"""
        percentage = min(100, max(0, percentage))  # Clamp to 0-100
        filled = int(width * percentage / 100)
        bar = '█' * filled + '░' * (width - filled)

        # Color coding
        if color == 'green':
            bar_str = f"\033[92m{bar}\033[0m"
        elif color == 'yellow':
            bar_str = f"\033[93m{bar}\033[0m"
        elif color == 'red':
            bar_str = f"\033[91m{bar}\033[0m"
        else:
            bar_str = bar

        print(f"{label:25} [{bar_str}] {percentage:6.1f}%")

    def get_elapsed_time(self) -> str:
        """Get formatted elapsed time"""
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"

    def update_progress(self, component: str, value: float, operation: Optional[str] = None):
        """Update progress for a component"""
        self.progress[component] = value

        if operation:
            self.progress['current_operation'] = operation

        # Update system readiness (weighted average)
        weights = {
            'data_completeness': 0.4,
            'metadata_coverage': 0.3,
            'index_progress': 0.2,
            'cache_progress': 0.1
        }

        total_weight = 0
        weighted_sum = 0

        for comp, weight in weights.items():
            if comp in self.progress:
                weighted_sum += self.progress[comp] * weight
                total_weight += weight

        if total_weight > 0:
            self.progress['system_readiness'] = weighted_sum / total_weight

    def estimate_time_remaining(self, completed_tasks: int, total_tasks: int,
                              avg_task_time: float) -> str:
        """Estimate time remaining based on current progress"""
        if completed_tasks >= total_tasks:
            return "Complete"

        remaining_tasks = total_tasks - completed_tasks
        estimated_seconds = remaining_tasks * avg_task_time

        if estimated_seconds < 60:
            return f"{int(estimated_seconds)} seconds"
        elif estimated_seconds < 3600:
            return f"{int(estimated_seconds / 60)} minutes"
        else:
            return f"{estimated_seconds / 3600:.1f} hours"


class CatchUpAssessment:
    """
    Analyzes what needs to be done on startup and creates catch-up plan.
    """

    def __init__(self, data_engine, important_symbols: List[str] = None):
        """
        Initialize assessment with data engine.

        Args:
            data_engine: DataEngine instance for checking stored data
            important_symbols: Priority symbols to check first
        """
        self.data_engine = data_engine
        self.important_symbols = important_symbols or ['AAPL', 'SPY', 'QQQ']
        self.tracked_symbols = self._get_tracked_symbols()

    def _get_tracked_symbols(self) -> List[str]:
        """Get list of symbols we're tracking"""
        # Start with important symbols
        symbols = self.important_symbols.copy()

        # Add any additional configured symbols
        # This would come from configuration
        additional = ['MSFT', 'GOOGL', 'TSLA', 'META', 'NVDA']
        symbols.extend(s for s in additional if s not in symbols)

        return symbols

    def assess_system_state(self) -> Dict:
        """
        Comprehensive system assessment on startup.

        Returns:
            Dictionary with assessment results
        """
        logger.info("Starting system assessment...")

        assessment = {
            'timestamp': datetime.now().isoformat(),
            'data_gaps': self.analyze_data_gaps(),
            'metadata_gaps': self.analyze_metadata_gaps(),
            'priority_tasks': [],
            'estimated_catch_up_time': None,
            'system_health': self.check_system_health()
        }

        # Identify priority tasks
        assessment['priority_tasks'] = self.identify_priority_tasks(assessment)

        # Estimate catch-up time
        assessment['estimated_catch_up_time'] = self.estimate_catch_up_time(assessment)

        # Generate summary
        assessment['summary'] = self.generate_assessment_summary(assessment)

        return assessment

    def analyze_data_gaps(self) -> Dict:
        """Detailed data gap analysis"""
        gaps = {
            'total_missing_days': 0,
            'by_symbol': {},
            'critical_gaps': [],  # Recent important data
            'historical_gaps': []  # Older, less critical
        }

        for symbol in self.tracked_symbols:
            # Get last stored date
            last_stored = self.get_last_stored_date(symbol)

            # Get expected trading days
            expected_days = self.get_trading_days_since(last_stored)

            # Find missing days
            stored_days = self.get_stored_days(symbol)
            missing_days = [d for d in expected_days if d not in stored_days]

            gaps['by_symbol'][symbol] = {
                'missing_count': len(missing_days),
                'last_data': last_stored.strftime('%Y-%m-%d') if last_stored else 'Never',
                'missing_days': [d.strftime('%Y-%m-%d') for d in missing_days],
                'completeness_pct': (1 - len(missing_days)/max(len(expected_days), 1)) * 100
            }

            gaps['total_missing_days'] += len(missing_days)

            # Categorize gaps
            for day in missing_days:
                days_old = (datetime.now().date() - day).days
                gap_info = {'symbol': symbol, 'date': day.strftime('%Y-%m-%d')}

                if days_old <= 2:
                    gaps['critical_gaps'].append(gap_info)
                else:
                    gaps['historical_gaps'].append(gap_info)

        return gaps

    def analyze_metadata_gaps(self) -> Dict:
        """Analyze missing or incomplete metadata"""
        gaps = {
            'total_missing': 0,
            'incomplete': [],
            'outdated': []
        }

        for symbol in self.tracked_symbols:
            stored_days = self.get_stored_days(symbol)

            for day in stored_days:
                metadata = self.get_metadata(symbol, day)

                if metadata is None:
                    gaps['total_missing'] += 1
                    gaps['incomplete'].append({
                        'symbol': symbol,
                        'date': day.strftime('%Y-%m-%d'),
                        'issue': 'missing'
                    })
                elif self.is_metadata_outdated(metadata):
                    gaps['outdated'].append({
                        'symbol': symbol,
                        'date': day.strftime('%Y-%m-%d'),
                        'version': metadata.get('version', 'unknown')
                    })

        return gaps

    def get_last_stored_date(self, symbol: str) -> Optional[datetime]:
        """Get the last date we have data for a symbol"""
        try:
            # Query storage for latest data
            stored_data = self.data_engine.storage_router.list_stored_data(symbol)
            if stored_data:
                dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in stored_data]
                return max(dates)
            return None
        except Exception as e:
            logger.error(f"Error getting last stored date for {symbol}: {e}")
            return None

    def get_trading_days_since(self, start_date: Optional[datetime]) -> List[datetime]:
        """Get list of trading days from start_date to now"""
        if start_date is None:
            # If no data, go back 8 days max
            start_date = datetime.now() - timedelta(days=8)

        # Use pandas to get trading days (excludes weekends)
        # In production, would use actual trading calendar
        date_range = pd.bdate_range(start=start_date, end=datetime.now())

        return [d.to_pydatetime().date() for d in date_range]

    def get_stored_days(self, symbol: str) -> List[datetime]:
        """Get list of days we have data for"""
        try:
            stored_data = self.data_engine.storage_router.list_stored_data(symbol)
            return [datetime.strptime(d['date'], '%Y-%m-%d').date()
                   for d in stored_data]
        except Exception:
            return []

    def get_metadata(self, symbol: str, date: datetime) -> Optional[Dict]:
        """Get metadata for a symbol-date"""
        try:
            return self.data_engine.storage_router.get_metadata(
                symbol, date.strftime('%Y-%m-%d')
            )
        except Exception:
            return None

    def is_metadata_outdated(self, metadata: Dict) -> bool:
        """Check if metadata needs recomputation"""
        from ..storage.metadata_computer import MetadataComputer

        current_version = MetadataComputer.METADATA_VERSION
        metadata_version = metadata.get('version', '0.0')

        return metadata_version < current_version

    def check_system_health(self) -> Dict:
        """Check overall system health"""
        health = {
            'storage_status': 'healthy',
            'storage_usage': 'Unknown',
            'last_successful_run': None,
            'errors_last_24h': 0,
            'warnings_last_24h': 0
        }

        # Check storage
        try:
            # This would check actual storage metrics
            health['storage_usage'] = '45GB / 100GB'  # Placeholder
        except Exception as e:
            health['storage_status'] = 'error'
            logger.error(f"Storage health check failed: {e}")

        return health

    def identify_priority_tasks(self, assessment: Dict) -> List[Dict]:
        """Identify and prioritize tasks based on assessment"""
        tasks = []

        # Critical data gaps (yesterday's data)
        for gap in assessment['data_gaps']['critical_gaps']:
            tasks.append({
                'type': 'fetch_data',
                'priority': 1,
                'symbol': gap['symbol'],
                'date': gap['date'],
                'description': f"Fetch {gap['symbol']} data for {gap['date']}"
            })

        # Missing metadata
        for incomplete in assessment['metadata_gaps']['incomplete'][:10]:  # Limit to 10
            tasks.append({
                'type': 'compute_metadata',
                'priority': 2,
                'symbol': incomplete['symbol'],
                'date': incomplete['date'],
                'description': f"Compute metadata for {incomplete['symbol']} on {incomplete['date']}"
            })

        # Sort by priority
        tasks.sort(key=lambda x: x['priority'])

        return tasks

    def estimate_catch_up_time(self, assessment: Dict) -> str:
        """Estimate time needed to catch up"""
        # Rough estimates
        time_per_fetch = 5  # seconds
        time_per_metadata = 2  # seconds

        total_seconds = 0

        # Data fetches
        total_seconds += len(assessment['data_gaps']['critical_gaps']) * time_per_fetch

        # Metadata computations
        total_seconds += assessment['metadata_gaps']['total_missing'] * time_per_metadata

        if total_seconds < 60:
            return f"{int(total_seconds)} seconds"
        elif total_seconds < 3600:
            return f"{int(total_seconds / 60)} minutes"
        else:
            return f"{total_seconds / 3600:.1f} hours"

    def generate_assessment_summary(self, assessment: Dict) -> Dict:
        """Generate human-readable summary"""
        summary = {
            'status': 'READY',
            'message': '',
            'actions_required': []
        }

        # Determine overall status
        if assessment['data_gaps']['total_missing_days'] == 0:
            summary['status'] = 'READY'
            summary['message'] = "System is fully up to date!"
        elif len(assessment['data_gaps']['critical_gaps']) > 0:
            summary['status'] = 'CATCHING_UP'
            summary['message'] = f"Missing {len(assessment['data_gaps']['critical_gaps'])} critical data points"
            summary['actions_required'].append('Fetching recent market data...')
        else:
            summary['status'] = 'PARTIAL'
            summary['message'] = "System operational, some historical data missing"

        return summary

    def display_assessment_dashboard(self, assessment: Dict):
        """Display assessment results in dashboard format"""
        print("\n" + "=" * 80)
        print("                        SYSTEM ASSESSMENT")
        print("=" * 80)

        # Data Coverage by Symbol
        print("\n📈 DATA COVERAGE BY SYMBOL:")
        print("-" * 80)

        for symbol, info in assessment['data_gaps']['by_symbol'].items():
            completeness = info['completeness_pct']
            bar = self.make_mini_bar(completeness)

            if completeness == 100:
                status = "✅"
            elif completeness > 90:
                status = "⚠️"
            else:
                status = "❌"

            print(f"{status} {symbol:6} {bar} {completeness:5.1f}% | Last: {info['last_data']}")

        # System Health
        print("\n🏥 SYSTEM HEALTH:")
        print("-" * 80)
        health = assessment['system_health']
        print(f"  Storage: {health['storage_status']} ({health['storage_usage']})")
        print(f"  Errors (24h): {health['errors_last_24h']}")
        print(f"  Warnings (24h): {health['warnings_last_24h']}")

        # Action Plan
        if assessment['priority_tasks']:
            print(f"\n📋 ACTION PLAN ({len(assessment['priority_tasks'])} tasks):")
            print("-" * 80)
            for task in assessment['priority_tasks'][:5]:  # Show first 5
                print(f"  • {task['description']}")
            if len(assessment['priority_tasks']) > 5:
                print(f"  ... and {len(assessment['priority_tasks']) - 5} more")

        # Time Estimate
        if assessment['estimated_catch_up_time']:
            print(f"\n⏱️  Estimated catch-up time: {assessment['estimated_catch_up_time']}")

    def make_mini_bar(self, percentage: float, width: int = 20) -> str:
        """Create a mini progress bar for inline display"""
        percentage = min(100, max(0, percentage))
        filled = int(width * percentage / 100)
        return '█' * filled + '░' * (width - filled)


class StartupSummaryDashboard:
    """
    Displays final summary after startup is complete.
    """

    def __init__(self, monitor: StartupProgressMonitor,
                 assessment: CatchUpAssessment):
        """Initialize with monitor and assessment instances"""
        self.monitor = monitor
        self.assessment = assessment
        self.completion_time = datetime.now()

    def display_startup_summary(self):
        """Show comprehensive startup summary"""
        duration = self.completion_time - self.monitor.start_time

        print("\n" + "=" * 80)
        print("                    ✅ STARTUP COMPLETE")
        print("=" * 80)

        # Time Statistics
        print(f"\n⏱️  TIMING:")
        print(f"  • Started: {self.monitor.start_time.strftime('%H:%M:%S')}")
        print(f"  • Completed: {self.completion_time.strftime('%H:%M:%S')}")
        print(f"  • Duration: {self._format_duration(duration)}")

        # Data Statistics
        print(f"\n📊 DATA STATISTICS:")
        print(f"  • Symbols tracked: {len(self.assessment.tracked_symbols)}")
        print(f"  • Days fetched: {self.monitor.stats['days_fetched']}")
        print(f"  • Ticks processed: {self.monitor.stats['ticks_processed']:,}")
        print(f"  • Metadata computed: {self.monitor.stats['metadata_computed']}")

        # System Status
        print(f"\n🎯 SYSTEM STATUS:")
        for component, status in self.monitor.component_status.items():
            icon = "✅" if status == 'ready' else "⚠️"
            print(f"  {icon} {component}: {status}")

        # Coverage Report
        print(f"\n📈 COVERAGE REPORT:")
        print(f"  • Data completeness: {self.monitor.progress['data_completeness']:.1f}%")
        print(f"  • Metadata coverage: {self.monitor.progress['metadata_coverage']:.1f}%")
        print(f"  • Overall readiness: {self.monitor.progress['system_readiness']:.1f}%")

        # Any Issues
        if self.monitor.issues:
            print(f"\n⚠️  ISSUES DETECTED:")
            for issue in self.monitor.issues:
                print(f"  • {issue}")

        # Recommendations
        if self.monitor.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.monitor.recommendations:
                print(f"  • {rec}")

        print("\n" + "=" * 80)
        print("System ready for use. Happy trading! 🚀")
        print("=" * 80)

    def _format_duration(self, duration: timedelta) -> str:
        """Format duration nicely"""
        total_seconds = duration.total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"