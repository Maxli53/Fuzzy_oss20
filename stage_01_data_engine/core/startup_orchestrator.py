"""
Startup Orchestrator - Main entry point for system startup
Coordinates state recovery, assessment, and catch-up operations
"""

import sys
import time
import logging
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from .startup_monitor import StartupProgressMonitor, CatchUpAssessment, StartupSummaryDashboard
from .state_manager import StateManager, CatchUpExecutor, TaskPriority
from ..core.data_engine import DataEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StartupOrchestrator:
    """
    Orchestrates the entire startup process including:
    - State recovery
    - System assessment
    - Catch-up execution
    - Progress monitoring
    """

    def __init__(self,
                 data_engine: Optional[DataEngine] = None,
                 state_file: str = "pipeline_state.json",
                 important_symbols: Optional[List[str]] = None,
                 display_progress: bool = True):
        """
        Initialize the startup orchestrator.

        Args:
            data_engine: DataEngine instance (creates new if None)
            state_file: Path to state persistence file
            important_symbols: Priority symbols to track
            display_progress: Whether to show visual progress
        """
        self.display_progress = display_progress

        # Initialize components
        logger.info("Initializing startup orchestrator...")

        # State manager
        self.state_manager = StateManager(state_file=state_file)

        # Update important symbols if provided
        if important_symbols:
            self.state_manager.update_config({
                'important_symbols': important_symbols
            })

        # Data engine
        if data_engine is None:
            logger.info("Creating new DataEngine instance...")
            self.data_engine = DataEngine()
        else:
            self.data_engine = data_engine

        # Progress monitor
        self.progress_monitor = StartupProgressMonitor(state_file=state_file)

        # Assessment
        self.assessment = CatchUpAssessment(
            data_engine=self.data_engine,
            important_symbols=self.state_manager.state['config']['important_symbols']
        )

        # Catch-up executor
        self.catch_up_executor = CatchUpExecutor(
            state_manager=self.state_manager,
            data_engine=self.data_engine
        )

        # Summary dashboard
        self.summary_dashboard = StartupSummaryDashboard(
            monitor=self.progress_monitor,
            assessment=self.assessment
        )

    def run_startup_sequence(self) -> bool:
        """
        Run the complete startup sequence.

        Returns:
            True if startup successful
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting FUZZY OSS20 Data Pipeline")
            logger.info("=" * 80)

            # Record startup in state
            startup_info = self.state_manager.on_startup()

            # Phase 1: System Assessment
            if self.display_progress:
                self.progress_monitor.update_progress(
                    'system_readiness', 10,
                    'Assessing system state...'
                )
                self.progress_monitor.display_startup_dashboard()

            logger.info("Phase 1: Assessing system state...")
            assessment_results = self.assessment.assess_system_state()

            # Display assessment
            if self.display_progress:
                self.assessment.display_assessment_dashboard(assessment_results)
                time.sleep(2)  # Let user see assessment

            # Phase 2: Prepare Catch-Up Plan
            if self.display_progress:
                self.progress_monitor.update_progress(
                    'system_readiness', 20,
                    'Preparing catch-up plan...'
                )
                self.progress_monitor.display_startup_dashboard()

            logger.info("Phase 2: Preparing catch-up plan...")
            catch_up_tasks = self._prepare_catch_up_plan(assessment_results)

            if catch_up_tasks:
                logger.info(f"Found {len(catch_up_tasks)} tasks to complete")

                # Phase 3: Execute Catch-Up
                if self.display_progress:
                    self.progress_monitor.update_progress(
                        'system_readiness', 30,
                        'Executing catch-up tasks...'
                    )
                    self.progress_monitor.display_startup_dashboard()

                logger.info("Phase 3: Executing catch-up tasks...")
                self._execute_catch_up(catch_up_tasks)

            else:
                logger.info("System is up to date, no catch-up needed")

            # Phase 4: Verify System Health
            if self.display_progress:
                self.progress_monitor.update_progress(
                    'system_readiness', 80,
                    'Verifying system health...'
                )
                self.progress_monitor.display_startup_dashboard()

            logger.info("Phase 4: Verifying system health...")
            health_check = self._verify_system_health()

            # Phase 5: Finalize Startup
            if self.display_progress:
                self.progress_monitor.update_progress(
                    'system_readiness', 95,
                    'Finalizing startup...'
                )
                self.progress_monitor.display_startup_dashboard()

            logger.info("Phase 5: Finalizing startup...")
            self._finalize_startup()

            # Mark complete
            if self.display_progress:
                self.progress_monitor.update_progress(
                    'system_readiness', 100,
                    'Startup complete!'
                )
                self.progress_monitor.update_progress('data_completeness', 100)
                self.progress_monitor.update_progress('metadata_coverage', 100)

                # Show final summary
                time.sleep(1)
                self.summary_dashboard.display_startup_summary()

            logger.info("Startup sequence completed successfully")
            return True

        except Exception as e:
            logger.error(f"Startup sequence failed: {e}", exc_info=True)
            self.progress_monitor.issues.append(f"Startup failed: {str(e)}")

            if self.display_progress:
                self.summary_dashboard.display_startup_summary()

            return False

        finally:
            # Always save state
            self.state_manager.save_state(immediate=True)

    def _prepare_catch_up_plan(self, assessment: dict) -> List[dict]:
        """
        Prepare prioritized catch-up plan based on assessment.

        Args:
            assessment: System assessment results

        Returns:
            List of prioritized tasks
        """
        tasks = []

        # Add critical data gaps (yesterday's data)
        for gap in assessment['data_gaps']['critical_gaps']:
            tasks.append({
                'id': f"fetch_{gap['symbol']}_{gap['date']}",
                'type': 'fetch_data',
                'symbol': gap['symbol'],
                'date': gap['date'],
                'priority': TaskPriority.CRITICAL.value,
                'description': f"Fetch {gap['symbol']} for {gap['date']}"
            })

        # Add missing metadata computations
        for incomplete in assessment['metadata_gaps']['incomplete'][:20]:  # Limit
            tasks.append({
                'id': f"metadata_{incomplete['symbol']}_{incomplete['date']}",
                'type': 'compute_metadata',
                'symbol': incomplete['symbol'],
                'date': incomplete['date'],
                'priority': TaskPriority.HIGH.value,
                'description': f"Compute metadata for {incomplete['symbol']} on {incomplete['date']}"
            })

        # Add historical gaps (lower priority)
        for gap in assessment['data_gaps']['historical_gaps'][:10]:  # Limit
            tasks.append({
                'id': f"historical_{gap['symbol']}_{gap['date']}",
                'type': 'fetch_data',
                'symbol': gap['symbol'],
                'date': gap['date'],
                'priority': TaskPriority.LOW.value,
                'description': f"Backfill {gap['symbol']} for {gap['date']}"
            })

        # Sort by priority
        tasks.sort(key=lambda x: x['priority'])

        # Add to state manager
        for task in tasks:
            self.state_manager.add_task(
                task,
                TaskPriority(task['priority'])
            )

        return tasks

    def _execute_catch_up(self, tasks: List[dict]):
        """
        Execute catch-up tasks with progress monitoring.

        Args:
            tasks: List of tasks to execute
        """
        total_tasks = len(tasks)
        completed = 0

        def progress_callback(progress: float, task: dict):
            """Update progress monitor"""
            nonlocal completed
            completed += 1

            self.progress_monitor.update_progress(
                'data_completeness',
                progress,
                f"Processing: {task.get('description', 'task')}"
            )

            self.progress_monitor.stats['days_fetched'] = completed

            # Update detailed status
            self.progress_monitor.progress['detailed_status'] = {
                'Completed': f"{completed}/{total_tasks}",
                'Current': task.get('description', 'Unknown'),
                'Progress': f"{progress:.1f}%"
            }

            # Estimate time remaining
            if completed > 0:
                elapsed = (datetime.now() - self.progress_monitor.start_time).total_seconds()
                avg_time = elapsed / completed
                remaining = (total_tasks - completed) * avg_time
                self.progress_monitor.progress['estimated_time_remaining'] = \
                    self.progress_monitor.estimate_time_remaining(
                        completed, total_tasks, avg_time
                    )

            if self.display_progress:
                self.progress_monitor.display_startup_dashboard()

        # Execute tasks
        self.catch_up_executor.execute_catch_up(progress_callback)

        # Update final stats
        self.progress_monitor.stats['ticks_processed'] = \
            self.state_manager.state['statistics'].get('total_ticks_processed', 0)
        self.progress_monitor.stats['metadata_computed'] = completed

    def _verify_system_health(self) -> bool:
        """
        Verify system health after catch-up.

        Returns:
            True if system is healthy
        """
        health_checks = {
            'Data Engine': 'ready',
            'Metadata Computer': 'ready',
            'Storage (ArcticDB)': 'ready',
            'IQFeed Connection': 'ready'
        }

        # Check data engine
        if not self.data_engine:
            health_checks['Data Engine'] = 'error'

        # Check IQFeed connection
        try:
            if hasattr(self.data_engine, 'iqfeed_collector'):
                if self.data_engine.iqfeed_collector:
                    health_checks['IQFeed Connection'] = 'ready'
                else:
                    health_checks['IQFeed Connection'] = 'not configured'
        except Exception:
            health_checks['IQFeed Connection'] = 'error'

        # Check storage
        try:
            # Simple storage check
            if hasattr(self.data_engine, 'storage_router'):
                health_checks['Storage (ArcticDB)'] = 'ready'
        except Exception:
            health_checks['Storage (ArcticDB)'] = 'error'

        # Update monitor
        self.progress_monitor.component_status = health_checks

        # Check for issues
        all_ready = all(status == 'ready' for status in health_checks.values())

        if not all_ready:
            for component, status in health_checks.items():
                if status != 'ready':
                    self.progress_monitor.issues.append(
                        f"{component}: {status}"
                    )

        return all_ready

    def _finalize_startup(self):
        """Finalize startup process"""
        # Add any recommendations based on what we found
        stats = self.state_manager.get_statistics()

        if stats['total_errors'] > 10:
            self.progress_monitor.recommendations.append(
                "High error count detected - check logs for issues"
            )

        # Check data coverage
        if self.progress_monitor.progress['data_completeness'] < 90:
            self.progress_monitor.recommendations.append(
                "Data coverage below 90% - consider extended backfill"
            )

        # Save final state
        self.state_manager.save_state(immediate=True)

    def shutdown(self):
        """Clean shutdown of the system"""
        logger.info("Initiating system shutdown...")

        try:
            # Record shutdown in state
            self.state_manager.on_shutdown()

            # Shutdown executor
            self.catch_up_executor.shutdown()

            logger.info("System shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def main():
    """
    Main entry point for standalone execution.
    """
    import argparse

    parser = argparse.ArgumentParser(description='FUZZY OSS20 Startup Orchestrator')
    parser.add_argument('--symbols', nargs='+',
                       default=['AAPL', 'SPY', 'QQQ'],
                       help='Important symbols to track')
    parser.add_argument('--state-file', default='pipeline_state.json',
                       help='State persistence file')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable visual progress display')
    parser.add_argument('--max-backfill', type=int, default=8,
                       help='Maximum days to backfill')

    args = parser.parse_args()

    # Create and run orchestrator
    orchestrator = StartupOrchestrator(
        important_symbols=args.symbols,
        state_file=args.state_file,
        display_progress=not args.no_display
    )

    # Update configuration
    orchestrator.state_manager.update_config({
        'max_backfill_days': args.max_backfill
    })

    try:
        # Run startup sequence
        success = orchestrator.run_startup_sequence()

        if success:
            logger.info("System ready for operation")
            # System would continue running here
            # For demo, just wait a bit then shutdown
            time.sleep(5)
        else:
            logger.error("Startup failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")

    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()