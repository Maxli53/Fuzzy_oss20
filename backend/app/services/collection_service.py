"""
CollectionService for IQFeed integration.
Manages real-time data collection and storage to ArcticDB.
"""

import sys
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import numpy as np
import pytz
import structlog
from enum import Enum

# Add paths for IQFeed collector
sys.path.append('C:/Users/maxli/PycharmProjects/PythonProject/Fuzzy_oss20')
from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector
from stage_01_data_engine.storage.tick_store import TickStore
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

from app.core.config import settings
from app.services.validation_service import ValidationService

logger = structlog.get_logger()


class CollectionStatus(Enum):
    """Collection task status."""
    IDLE = "idle"
    COLLECTING = "collecting"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class CollectionTask:
    """Represents a data collection task."""

    def __init__(
        self,
        task_id: str,
        symbol: str,
        collection_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        self.task_id = task_id
        self.symbol = symbol
        self.collection_type = collection_type
        self.start_time = start_time
        self.end_time = end_time
        self.status = CollectionStatus.IDLE
        self.ticks_collected = 0
        self.errors = []
        self.created_at = datetime.now(pytz.timezone(settings.TIMEZONE))
        self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "task_id": self.task_id,
            "symbol": self.symbol,
            "collection_type": self.collection_type,
            "status": self.status.value,
            "ticks_collected": self.ticks_collected,
            "errors": self.errors[-10:],  # Last 10 errors
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


class CollectionService:
    """
    Service for managing IQFeed data collection.
    Handles real-time and historical data collection with validation.
    """

    def __init__(self):
        self.et_tz = pytz.timezone(settings.TIMEZONE)
        self.collector = None
        self.tick_store = None
        self.validation_service = ValidationService()
        self.active_tasks: Dict[str, CollectionTask] = {}
        self.streaming_symbols: Set[str] = set()
        self._initialize_services()

    def _initialize_services(self):
        """Initialize IQFeed collector and tick store."""
        try:
            # Initialize IQFeed collector
            self.collector = IQFeedCollector()
            if not self.collector.ensure_connection():
                raise ConnectionError("Failed to connect to IQFeed")

            # Initialize tick store
            self.tick_store = TickStore()

            logger.info("Collection services initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize collection services", error=str(e))
            raise

    async def collect_historical_ticks(
        self,
        symbol: str,
        num_days: int = 1,
        max_ticks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Collect historical tick data from IQFeed.

        Args:
            symbol: Stock symbol
            num_days: Number of days to collect
            max_ticks: Maximum number of ticks

        Returns:
            Collection result dictionary
        """
        task_id = f"hist_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        task = CollectionTask(
            task_id=task_id,
            symbol=symbol,
            collection_type="historical",
            start_time=datetime.now(self.et_tz) - timedelta(days=num_days),
            end_time=datetime.now(self.et_tz)
        )

        self.active_tasks[task_id] = task

        try:
            task.status = CollectionStatus.COLLECTING
            logger.info(f"Starting historical collection for {symbol}", num_days=num_days)

            # Fetch tick data from IQFeed
            tick_array = self.collector.get_tick_data(
                symbol=symbol,
                num_days=num_days,
                max_ticks=max_ticks
            )

            if len(tick_array) == 0:
                task.status = CollectionStatus.COMPLETED
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "message": f"No data available for {symbol}",
                    "ticks_collected": 0
                }

            # Convert to Pydantic models
            pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)
            task.ticks_collected = len(pydantic_ticks)

            # Validate ticks
            validation_results = []
            for tick in pydantic_ticks[:100]:  # Validate sample
                tick_dict = tick.model_dump()
                result = await self.validation_service.validate_realtime_tick(symbol, tick_dict)
                if result.status.value != "passed":
                    validation_results.append(result.to_dict())

            # Store ticks in ArcticDB
            stored_count = self.tick_store.store_ticks(pydantic_ticks)

            task.status = CollectionStatus.COMPLETED
            task.updated_at = datetime.now(self.et_tz)

            logger.info(
                f"Historical collection completed",
                symbol=symbol,
                ticks_collected=task.ticks_collected,
                stored=stored_count
            )

            return {
                "task_id": task_id,
                "status": "completed",
                "symbol": symbol,
                "ticks_collected": task.ticks_collected,
                "ticks_stored": stored_count,
                "validation_warnings": len(validation_results),
                "validation_samples": validation_results[:5]
            }

        except Exception as e:
            task.status = CollectionStatus.ERROR
            task.errors.append(str(e))
            task.updated_at = datetime.now(self.et_tz)

            logger.error("Historical collection failed", error=str(e), symbol=symbol)

            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "ticks_collected": task.ticks_collected
            }

    async def start_realtime_collection(
        self,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Start real-time tick collection for symbols.

        Args:
            symbols: List of symbols to collect

        Returns:
            Status dictionary
        """
        try:
            added_symbols = []
            already_streaming = []

            for symbol in symbols:
                if symbol not in self.streaming_symbols:
                    # Create collection task
                    task_id = f"rt_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    task = CollectionTask(
                        task_id=task_id,
                        symbol=symbol,
                        collection_type="realtime",
                        start_time=datetime.now(self.et_tz)
                    )
                    task.status = CollectionStatus.COLLECTING
                    self.active_tasks[task_id] = task

                    # Add to streaming set
                    self.streaming_symbols.add(symbol)
                    added_symbols.append(symbol)

                    # Start streaming task
                    asyncio.create_task(self._stream_symbol(symbol, task_id))
                else:
                    already_streaming.append(symbol)

            return {
                "status": "started",
                "added_symbols": added_symbols,
                "already_streaming": already_streaming,
                "total_streaming": len(self.streaming_symbols)
            }

        except Exception as e:
            logger.error("Failed to start realtime collection", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }

    async def stop_realtime_collection(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Stop real-time tick collection.

        Args:
            symbols: Symbols to stop (None = stop all)

        Returns:
            Status dictionary
        """
        try:
            if symbols is None:
                symbols = list(self.streaming_symbols)

            stopped_symbols = []
            for symbol in symbols:
                if symbol in self.streaming_symbols:
                    self.streaming_symbols.remove(symbol)
                    stopped_symbols.append(symbol)

                    # Update task status
                    for task_id, task in self.active_tasks.items():
                        if task.symbol == symbol and task.status == CollectionStatus.COLLECTING:
                            task.status = CollectionStatus.COMPLETED
                            task.updated_at = datetime.now(self.et_tz)

            return {
                "status": "stopped",
                "stopped_symbols": stopped_symbols,
                "remaining_streaming": len(self.streaming_symbols)
            }

        except Exception as e:
            logger.error("Failed to stop realtime collection", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }

    async def _stream_symbol(self, symbol: str, task_id: str):
        """
        Stream real-time data for a single symbol.
        This is a long-running coroutine.

        Args:
            symbol: Symbol to stream
            task_id: Associated task ID
        """
        buffer = []
        buffer_size = 100  # Store every 100 ticks

        try:
            while symbol in self.streaming_symbols:
                try:
                    # Get latest ticks (this would be replaced with actual streaming)
                    tick_array = self.collector.get_tick_data(
                        symbol=symbol,
                        num_days=0,  # Today only
                        max_ticks=10
                    )

                    if len(tick_array) > 0:
                        # Convert to Pydantic
                        pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)
                        buffer.extend(pydantic_ticks)

                        # Update task
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id].ticks_collected += len(pydantic_ticks)
                            self.active_tasks[task_id].updated_at = datetime.now(self.et_tz)

                        # Store when buffer is full
                        if len(buffer) >= buffer_size:
                            self.tick_store.store_ticks(buffer)
                            logger.debug(f"Stored {len(buffer)} ticks for {symbol}")
                            buffer = []

                    # Wait before next poll
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Error streaming {symbol}: {str(e)}")
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id].errors.append(str(e))

        finally:
            # Store remaining buffer
            if buffer:
                self.tick_store.store_ticks(buffer)

            logger.info(f"Stopped streaming {symbol}")

    async def get_collection_status(self) -> Dict[str, Any]:
        """
        Get status of all collection tasks.

        Returns:
            Status dictionary
        """
        return {
            "streaming_symbols": list(self.streaming_symbols),
            "active_tasks": len(self.active_tasks),
            "tasks": [task.to_dict() for task in list(self.active_tasks.values())[-20:]]  # Last 20 tasks
        }

    async def backfill_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Backfill historical data for a date range.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Backfill result
        """
        task_id = f"backfill_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        task = CollectionTask(
            task_id=task_id,
            symbol=symbol,
            collection_type="backfill",
            start_time=start_date,
            end_time=end_date
        )

        self.active_tasks[task_id] = task

        try:
            task.status = CollectionStatus.COLLECTING
            num_days = (end_date - start_date).days + 1

            logger.info(
                f"Starting backfill for {symbol}",
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                days=num_days
            )

            # Collect data day by day
            current_date = start_date
            total_ticks = 0

            while current_date <= end_date:
                # Skip weekends
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue

                # Collect for this day
                tick_array = self.collector.get_tick_data(
                    symbol=symbol,
                    num_days=1,
                    reference_date=current_date
                )

                if len(tick_array) > 0:
                    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)
                    stored = self.tick_store.store_ticks(pydantic_ticks)
                    total_ticks += stored

                    task.ticks_collected = total_ticks
                    task.updated_at = datetime.now(self.et_tz)

                current_date += timedelta(days=1)

            task.status = CollectionStatus.COMPLETED
            task.updated_at = datetime.now(self.et_tz)

            return {
                "task_id": task_id,
                "status": "completed",
                "symbol": symbol,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "ticks_collected": total_ticks,
                "days_processed": num_days
            }

        except Exception as e:
            task.status = CollectionStatus.ERROR
            task.errors.append(str(e))

            logger.error("Backfill failed", error=str(e), symbol=symbol)

            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "ticks_collected": task.ticks_collected
            }

    async def check_data_health(
        self,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Check data health for symbols.

        Args:
            symbols: List of symbols to check

        Returns:
            Health check results
        """
        results = {}

        for symbol in symbols:
            try:
                # Get latest tick
                latest_tick = await self._get_latest_stored_tick(symbol)

                if latest_tick:
                    # Calculate delay
                    now = datetime.now(self.et_tz)
                    tick_time = pd.Timestamp(latest_tick["timestamp"]).tz_localize(self.et_tz)
                    delay_seconds = (now - tick_time).total_seconds()

                    results[symbol] = {
                        "status": "healthy" if delay_seconds < settings.MAX_TICK_DELAY_SECONDS else "stale",
                        "latest_tick": latest_tick["timestamp"],
                        "delay_seconds": delay_seconds,
                        "price": latest_tick.get("price")
                    }
                else:
                    results[symbol] = {
                        "status": "no_data",
                        "message": "No tick data found"
                    }

            except Exception as e:
                results[symbol] = {
                    "status": "error",
                    "error": str(e)
                }

        return {
            "timestamp": datetime.now(self.et_tz).isoformat(),
            "symbols_checked": len(symbols),
            "results": results
        }

    async def _get_latest_stored_tick(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Get latest tick from storage."""
        # This would query ArcticDB for the latest tick
        # Placeholder implementation
        return None