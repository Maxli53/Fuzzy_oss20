"""
State Persistence and Recovery for Non-24/7 Operation
Manages system state between sessions and handles catch-up on startup
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue
import threading
import time

import pandas as pd

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1  # Must complete immediately
    HIGH = 2      # Important, do soon
    NORMAL = 3    # Regular priority
    LOW = 4       # Can wait
    DEFERRED = 5  # Do when idle


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StateManager:
    """
    Manages persistent state for the data pipeline.
    Handles recovery and catch-up after system restarts.
    """

    STATE_VERSION = "1.0"

    def __init__(self, state_file: str = "pipeline_state.json",
                 backup_count: int = 3):
        """
        Initialize state manager.

        Args:
            state_file: Path to state persistence file
            backup_count: Number of backup files to keep
        """
        self.state_file = Path(state_file)
        self.backup_count = backup_count
        self.state = self._load_state()
        self.lock = threading.Lock()
        self._save_timer = None

    def _load_state(self) -> Dict:
        """Load state from file or create new"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    logger.info(f"Loaded state from {self.state_file}")

                    # Validate and migrate if needed
                    state = self._migrate_state(state)
                    return state

            except Exception as e:
                logger.error(f"Error loading state: {e}")
                # Try backup files
                return self._load_from_backup()
        else:
            logger.info("No state file found, creating new state")
            return self._create_new_state()

    def _create_new_state(self) -> Dict:
        """Create a new clean state"""
        return {
            'version': self.STATE_VERSION,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'last_startup': None,
            'last_shutdown': None,

            # Data ingestion tracking
            'last_successful_ingestion': {},  # symbol -> timestamp
            'failed_ingestions': [],          # List of failed attempts

            # Metadata tracking
            'metadata_computed': {},  # symbol/date -> status
            'metadata_version': {},   # symbol/date -> version

            # Pending work
            'pending_tasks': [],      # Tasks to complete
            'deferred_tasks': [],     # Low priority tasks

            # System statistics
            'statistics': {
                'total_sessions': 0,
                'total_runtime_seconds': 0,
                'total_ticks_processed': 0,
                'total_errors': 0
            },

            # Configuration
            'config': {
                'important_symbols': ['AAPL', 'SPY', 'QQQ'],
                'max_backfill_days': 8,
                'metadata_compute_mode': 'lazy'  # 'immediate' or 'lazy'
            }
        }

    def _migrate_state(self, state: Dict) -> Dict:
        """Migrate old state format to current version"""
        current_version = self.STATE_VERSION

        if 'version' not in state:
            # Very old format, recreate
            logger.warning("State file has no version, resetting")
            return self._create_new_state()

        if state['version'] < current_version:
            logger.info(f"Migrating state from {state['version']} to {current_version}")
            # Perform migrations here
            state['version'] = current_version

        return state

    def _load_from_backup(self) -> Dict:
        """Try to load from backup files"""
        for i in range(1, self.backup_count + 1):
            backup_file = Path(f"{self.state_file}.backup{i}")
            if backup_file.exists():
                try:
                    with open(backup_file, 'r') as f:
                        state = json.load(f)
                        logger.info(f"Loaded state from backup: {backup_file}")
                        return self._migrate_state(state)
                except Exception as e:
                    logger.error(f"Error loading backup {backup_file}: {e}")
                    continue

        # No valid backup found
        logger.warning("No valid backup found, creating new state")
        return self._create_new_state()

    def save_state(self, immediate: bool = False):
        """
        Save state to file.

        Args:
            immediate: If True, save immediately. Otherwise, defer for batching
        """
        with self.lock:
            if immediate:
                self._save_to_file()
            else:
                # Defer saving to batch updates
                if self._save_timer:
                    self._save_timer.cancel()

                self._save_timer = threading.Timer(5.0, self._save_to_file)
                self._save_timer.start()

    def _save_to_file(self):
        """Actually save state to file with rotation"""
        with self.lock:
            try:
                # Update timestamp
                self.state['last_updated'] = datetime.now().isoformat()

                # Rotate backups
                self._rotate_backups()

                # Save current state
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f, indent=2, default=str)

                logger.debug("State saved successfully")

            except Exception as e:
                logger.error(f"Error saving state: {e}")

    def _rotate_backups(self):
        """Rotate backup files"""
        # Move existing backups
        for i in range(self.backup_count - 1, 0, -1):
            old_backup = Path(f"{self.state_file}.backup{i}")
            new_backup = Path(f"{self.state_file}.backup{i+1}")
            if old_backup.exists():
                old_backup.rename(new_backup)

        # Current file becomes backup1
        if self.state_file.exists():
            backup1 = Path(f"{self.state_file}.backup1")
            self.state_file.rename(backup1)

    def on_startup(self) -> Dict:
        """
        Called when system starts up.

        Returns:
            Dictionary with startup information and tasks
        """
        logger.info("System startup initiated")

        with self.lock:
            # Record startup
            self.state['last_startup'] = datetime.now().isoformat()
            self.state['statistics']['total_sessions'] += 1

            # Calculate downtime
            if self.state['last_shutdown']:
                last_shutdown = datetime.fromisoformat(self.state['last_shutdown'])
                downtime = datetime.now() - last_shutdown
                logger.info(f"System was down for {downtime}")

            self.save_state()

        # Identify what needs to be done
        startup_info = {
            'last_shutdown': self.state.get('last_shutdown'),
            'pending_tasks': self.get_pending_tasks(),
            'catch_up_needed': self.identify_catch_up_tasks(),
            'system_config': self.state['config']
        }

        return startup_info

    def on_shutdown(self):
        """Called when system shuts down"""
        logger.info("System shutdown initiated")

        with self.lock:
            self.state['last_shutdown'] = datetime.now().isoformat()

            # Calculate session runtime
            if self.state['last_startup']:
                startup = datetime.fromisoformat(self.state['last_startup'])
                runtime = (datetime.now() - startup).total_seconds()
                self.state['statistics']['total_runtime_seconds'] += runtime

            self.save_state(immediate=True)

    def record_ingestion(self, symbol: str, date: str, success: bool,
                        error: Optional[str] = None):
        """
        Record data ingestion attempt.

        Args:
            symbol: Stock symbol
            date: Date string
            success: Whether ingestion succeeded
            error: Error message if failed
        """
        with self.lock:
            if success:
                self.state['last_successful_ingestion'][symbol] = datetime.now().isoformat()
            else:
                self.state['failed_ingestions'].append({
                    'symbol': symbol,
                    'date': date,
                    'timestamp': datetime.now().isoformat(),
                    'error': error
                })

            self.save_state()

    def record_metadata_computation(self, symbol: str, date: str,
                                   version: str, success: bool):
        """Record metadata computation"""
        with self.lock:
            key = f"{symbol}/{date}"

            if success:
                self.state['metadata_computed'][key] = {
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                }
                self.state['metadata_version'][key] = version
            else:
                self.state['metadata_computed'][key] = {
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                }

            self.save_state()

    def add_task(self, task: Dict, priority: TaskPriority = TaskPriority.NORMAL):
        """
        Add a task to pending queue.

        Args:
            task: Task dictionary with type, symbol, date, etc.
            priority: Task priority
        """
        with self.lock:
            task['priority'] = priority.value
            task['added_at'] = datetime.now().isoformat()
            task['status'] = TaskStatus.PENDING.value

            if priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                self.state['pending_tasks'].append(task)
            else:
                self.state['deferred_tasks'].append(task)

            self.save_state()

    def get_pending_tasks(self) -> List[Dict]:
        """Get list of pending tasks sorted by priority"""
        with self.lock:
            tasks = self.state['pending_tasks'].copy()
            tasks.sort(key=lambda x: (x.get('priority', 999),
                                     x.get('added_at', '')))
            return tasks

    def update_task_status(self, task_id: str, status: TaskStatus,
                          error: Optional[str] = None):
        """Update status of a task"""
        with self.lock:
            # Find task in pending or deferred
            for task_list in [self.state['pending_tasks'],
                             self.state['deferred_tasks']]:
                for task in task_list:
                    if task.get('id') == task_id:
                        task['status'] = status.value
                        task['updated_at'] = datetime.now().isoformat()
                        if error:
                            task['error'] = error

                        # Remove if completed or failed
                        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            task_list.remove(task)

                        self.save_state()
                        return

    def identify_catch_up_tasks(self) -> List[Dict]:
        """
        Identify what needs to be caught up based on time away.

        Returns:
            List of catch-up tasks
        """
        tasks = []

        # Get important symbols
        important_symbols = self.state['config']['important_symbols']

        for symbol in important_symbols:
            # Check last ingestion
            last_ingestion = self.state['last_successful_ingestion'].get(symbol)

            if last_ingestion:
                last_date = datetime.fromisoformat(last_ingestion).date()
                days_behind = (datetime.now().date() - last_date).days

                if days_behind > 0:
                    # Need to catch up
                    for i in range(min(days_behind,
                                     self.state['config']['max_backfill_days'])):
                        date = datetime.now().date() - timedelta(days=i)

                        # Skip weekends
                        if date.weekday() >= 5:
                            continue

                        tasks.append({
                            'type': 'fetch_data',
                            'symbol': symbol,
                            'date': date.strftime('%Y-%m-%d'),
                            'priority': TaskPriority.HIGH.value if i <= 1
                                       else TaskPriority.NORMAL.value
                        })
            else:
                # Never fetched, get recent data
                tasks.append({
                    'type': 'fetch_data',
                    'symbol': symbol,
                    'date': datetime.now().date().strftime('%Y-%m-%d'),
                    'priority': TaskPriority.HIGH.value
                })

        return tasks

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        with self.lock:
            return self.state['statistics'].copy()

    def update_config(self, config_updates: Dict):
        """Update configuration settings"""
        with self.lock:
            self.state['config'].update(config_updates)
            self.save_state(immediate=True)


class CatchUpExecutor:
    """
    Executes catch-up tasks with intelligent scheduling and prioritization.
    """

    def __init__(self, state_manager: StateManager, data_engine,
                 max_workers: int = 4):
        """
        Initialize catch-up executor.

        Args:
            state_manager: StateManager instance
            data_engine: DataEngine for fetching/storing data
            max_workers: Maximum parallel workers
        """
        self.state_manager = state_manager
        self.data_engine = data_engine
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = PriorityQueue()
        self.running_tasks = {}
        self.completed_count = 0
        self.failed_count = 0

    def execute_catch_up(self, progress_callback=None):
        """
        Execute catch-up tasks.

        Args:
            progress_callback: Callback function for progress updates
        """
        logger.info("Starting catch-up execution")

        # Get all pending tasks
        tasks = self.state_manager.get_pending_tasks()
        tasks.extend(self.state_manager.identify_catch_up_tasks())

        # Add to priority queue
        for task in tasks:
            priority = task.get('priority', TaskPriority.NORMAL.value)
            self.task_queue.put((priority, task))

        total_tasks = self.task_queue.qsize()
        logger.info(f"Processing {total_tasks} catch-up tasks")

        # Process tasks
        futures = []
        while not self.task_queue.empty() or futures:
            # Submit new tasks if workers available
            while not self.task_queue.empty() and len(futures) < self.max_workers:
                _, task = self.task_queue.get()
                future = self.executor.submit(self._execute_task, task)
                futures.append(future)
                self.running_tasks[future] = task

            # Check for completed tasks
            completed = []
            for future in as_completed(futures, timeout=0.1):
                completed.append(future)
                task = self.running_tasks[future]

                try:
                    result = future.result()
                    if result:
                        self.completed_count += 1
                        logger.info(f"Task completed: {task.get('description', task)}")
                    else:
                        self.failed_count += 1
                        logger.error(f"Task failed: {task.get('description', task)}")

                except Exception as e:
                    self.failed_count += 1
                    logger.error(f"Task exception: {e}")

                finally:
                    del self.running_tasks[future]

                # Update progress
                if progress_callback:
                    progress = (self.completed_count + self.failed_count) / total_tasks * 100
                    progress_callback(progress, task)

            # Remove completed futures
            for future in completed:
                futures.remove(future)

            # Brief pause to prevent CPU spinning
            if futures:
                time.sleep(0.1)

        logger.info(f"Catch-up complete: {self.completed_count} succeeded, "
                   f"{self.failed_count} failed")

    def _execute_task(self, task: Dict) -> bool:
        """
        Execute a single task.

        Args:
            task: Task dictionary

        Returns:
            True if successful
        """
        task_type = task.get('type')

        try:
            if task_type == 'fetch_data':
                return self._fetch_data(task)
            elif task_type == 'compute_metadata':
                return self._compute_metadata(task)
            else:
                logger.warning(f"Unknown task type: {task_type}")
                return False

        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return False

    def _fetch_data(self, task: Dict) -> bool:
        """Fetch historical data for a symbol"""
        symbol = task['symbol']
        date = task['date']

        try:
            # Use data engine to fetch
            data = self.data_engine.fetch_historical_data(
                symbol=symbol,
                date=date
            )

            if data is not None and not data.empty:
                # Store the data
                success = self.data_engine.store_data(
                    symbol=symbol,
                    date=date,
                    data=data
                )

                # Record in state
                self.state_manager.record_ingestion(
                    symbol=symbol,
                    date=date,
                    success=success
                )

                return success
            else:
                logger.warning(f"No data received for {symbol} on {date}")
                return False

        except Exception as e:
            logger.error(f"Error fetching {symbol} for {date}: {e}")
            self.state_manager.record_ingestion(
                symbol=symbol,
                date=date,
                success=False,
                error=str(e)
            )
            return False

    def _compute_metadata(self, task: Dict) -> bool:
        """Compute metadata for stored data"""
        symbol = task['symbol']
        date = task['date']

        try:
            # Load data
            data = self.data_engine.load_data(symbol=symbol, date=date)

            if data is None or data.empty:
                logger.warning(f"No data to compute metadata for {symbol} on {date}")
                return False

            # Compute metadata
            from ..storage.metadata_computer import MetadataComputer
            computer = MetadataComputer()
            metadata = computer.compute_all_metadata(data, symbol, date)

            # Store metadata
            success = self.data_engine.store_metadata(
                symbol=symbol,
                date=date,
                metadata=metadata
            )

            # Record in state
            self.state_manager.record_metadata_computation(
                symbol=symbol,
                date=date,
                version=metadata.get('version', '0.0'),
                success=success
            )

            return success

        except Exception as e:
            logger.error(f"Error computing metadata for {symbol} on {date}: {e}")
            self.state_manager.record_metadata_computation(
                symbol=symbol,
                date=date,
                version='0.0',
                success=False
            )
            return False

    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)