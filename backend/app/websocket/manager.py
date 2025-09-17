"""
WebSocket manager for real-time data streaming.
Handles client connections and broadcasts tick/bar updates.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Set, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import structlog
from enum import Enum

from app.core.config import settings

logger = structlog.get_logger()


class MessageType(Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    TICK = "tick"
    BAR = "bar"
    VALIDATION = "validation"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class ClientConnection:
    """Represents a WebSocket client connection."""

    def __init__(self, websocket: WebSocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.subscribed_symbols: Set[str] = set()
        self.subscribed_channels: Set[str] = set()
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()

    async def send_json(self, data: Dict[str, Any]):
        """Send JSON data to client."""
        try:
            await self.websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send to client {self.client_id}: {str(e)}")
            raise


class WebSocketManager:
    """
    Manages WebSocket connections and message broadcasting.
    Implements pub/sub pattern for real-time data distribution.
    """

    def __init__(self):
        self.active_connections: Dict[str, ClientConnection] = {}
        self.symbol_subscribers: Dict[str, Set[str]] = {}  # symbol -> set of client_ids
        self.channel_subscribers: Dict[str, Set[str]] = {}  # channel -> set of client_ids
        self._heartbeat_task = None
        self._cleanup_task = None

    async def initialize(self):
        """Initialize WebSocket manager and start background tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("WebSocket manager initialized")

    async def shutdown(self):
        """Shutdown WebSocket manager and cleanup."""
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Close all connections
        for client_id in list(self.active_connections.keys()):
            await self.disconnect(client_id)

        logger.info("WebSocket manager shutdown complete")

    async def connect(self, websocket: WebSocket, client_id: str) -> ClientConnection:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            client_id: Unique client identifier

        Returns:
            ClientConnection instance
        """
        await websocket.accept()
        client = ClientConnection(websocket, client_id)
        self.active_connections[client_id] = client

        # Send welcome message
        await client.send_json({
            "type": MessageType.STATUS.value,
            "message": "Connected to Fuzzy OSS20 WebSocket",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        logger.info(f"Client {client_id} connected")
        return client

    async def disconnect(self, client_id: str):
        """
        Disconnect a client and cleanup subscriptions.

        Args:
            client_id: Client identifier
        """
        if client_id not in self.active_connections:
            return

        client = self.active_connections[client_id]

        # Remove from all subscriptions
        for symbol in list(client.subscribed_symbols):
            await self.unsubscribe_symbol(client_id, symbol)

        for channel in list(client.subscribed_channels):
            await self.unsubscribe_channel(client_id, channel)

        # Close WebSocket
        try:
            await client.websocket.close()
        except:
            pass

        # Remove from active connections
        del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """
        Handle incoming WebSocket message from client.

        Args:
            client_id: Client identifier
            message: Parsed message dictionary
        """
        try:
            msg_type = message.get("type")

            if msg_type == MessageType.SUBSCRIBE.value:
                await self._handle_subscribe(client_id, message)
            elif msg_type == MessageType.UNSUBSCRIBE.value:
                await self._handle_unsubscribe(client_id, message)
            elif msg_type == MessageType.HEARTBEAT.value:
                await self._handle_heartbeat(client_id)
            else:
                await self.send_error(client_id, f"Unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {str(e)}")
            await self.send_error(client_id, str(e))

    async def _handle_subscribe(self, client_id: str, message: Dict[str, Any]):
        """Handle subscription request."""
        symbols = message.get("symbols", [])
        channels = message.get("channels", [])

        # Subscribe to symbols
        for symbol in symbols:
            await self.subscribe_symbol(client_id, symbol)

        # Subscribe to channels
        for channel in channels:
            await self.subscribe_channel(client_id, channel)

        # Send confirmation
        client = self.active_connections.get(client_id)
        if client:
            await client.send_json({
                "type": MessageType.STATUS.value,
                "action": "subscribed",
                "symbols": list(client.subscribed_symbols),
                "channels": list(client.subscribed_channels),
                "timestamp": datetime.utcnow().isoformat()
            })

    async def _handle_unsubscribe(self, client_id: str, message: Dict[str, Any]):
        """Handle unsubscription request."""
        symbols = message.get("symbols", [])
        channels = message.get("channels", [])

        # Unsubscribe from symbols
        for symbol in symbols:
            await self.unsubscribe_symbol(client_id, symbol)

        # Unsubscribe from channels
        for channel in channels:
            await self.unsubscribe_channel(client_id, channel)

        # Send confirmation
        client = self.active_connections.get(client_id)
        if client:
            await client.send_json({
                "type": MessageType.STATUS.value,
                "action": "unsubscribed",
                "symbols": list(client.subscribed_symbols),
                "channels": list(client.subscribed_channels),
                "timestamp": datetime.utcnow().isoformat()
            })

    async def _handle_heartbeat(self, client_id: str):
        """Handle heartbeat message."""
        if client_id in self.active_connections:
            self.active_connections[client_id].last_heartbeat = datetime.utcnow()

    async def subscribe_symbol(self, client_id: str, symbol: str):
        """Subscribe client to symbol updates."""
        if client_id not in self.active_connections:
            return

        client = self.active_connections[client_id]
        client.subscribed_symbols.add(symbol)

        if symbol not in self.symbol_subscribers:
            self.symbol_subscribers[symbol] = set()
        self.symbol_subscribers[symbol].add(client_id)

        logger.debug(f"Client {client_id} subscribed to {symbol}")

    async def unsubscribe_symbol(self, client_id: str, symbol: str):
        """Unsubscribe client from symbol updates."""
        if client_id not in self.active_connections:
            return

        client = self.active_connections[client_id]
        client.subscribed_symbols.discard(symbol)

        if symbol in self.symbol_subscribers:
            self.symbol_subscribers[symbol].discard(client_id)
            if not self.symbol_subscribers[symbol]:
                del self.symbol_subscribers[symbol]

        logger.debug(f"Client {client_id} unsubscribed from {symbol}")

    async def subscribe_channel(self, client_id: str, channel: str):
        """Subscribe client to a channel."""
        if client_id not in self.active_connections:
            return

        client = self.active_connections[client_id]
        client.subscribed_channels.add(channel)

        if channel not in self.channel_subscribers:
            self.channel_subscribers[channel] = set()
        self.channel_subscribers[channel].add(client_id)

        logger.debug(f"Client {client_id} subscribed to channel {channel}")

    async def unsubscribe_channel(self, client_id: str, channel: str):
        """Unsubscribe client from a channel."""
        if client_id not in self.active_connections:
            return

        client = self.active_connections[client_id]
        client.subscribed_channels.discard(channel)

        if channel in self.channel_subscribers:
            self.channel_subscribers[channel].discard(client_id)
            if not self.channel_subscribers[channel]:
                del self.channel_subscribers[channel]

        logger.debug(f"Client {client_id} unsubscribed from channel {channel}")

    async def broadcast_tick(self, symbol: str, tick_data: Dict[str, Any]):
        """
        Broadcast tick data to all subscribers.

        Args:
            symbol: Stock symbol
            tick_data: Tick data dictionary
        """
        if symbol not in self.symbol_subscribers:
            return

        message = {
            "type": MessageType.TICK.value,
            "symbol": symbol,
            "data": tick_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Send to all subscribers
        disconnected = []
        for client_id in self.symbol_subscribers[symbol]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except:
                    disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

    async def broadcast_bar(self, symbol: str, interval: str, bar_data: Dict[str, Any]):
        """
        Broadcast bar data to subscribers.

        Args:
            symbol: Stock symbol
            interval: Bar interval (1m, 5m, etc.)
            bar_data: Bar data dictionary
        """
        if symbol not in self.symbol_subscribers:
            return

        message = {
            "type": MessageType.BAR.value,
            "symbol": symbol,
            "interval": interval,
            "data": bar_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Send to symbol subscribers
        disconnected = []
        for client_id in self.symbol_subscribers[symbol]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except:
                    disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

    async def broadcast_to_channel(self, channel: str, data: Dict[str, Any]):
        """
        Broadcast data to all channel subscribers.

        Args:
            channel: Channel name
            data: Data to broadcast
        """
        if channel not in self.channel_subscribers:
            return

        message = {
            "type": "channel",
            "channel": channel,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }

        disconnected = []
        for client_id in self.channel_subscribers[channel]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except:
                    disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

    async def send_error(self, client_id: str, error_message: str):
        """Send error message to specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json({
                    "type": MessageType.ERROR.value,
                    "message": error_message,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except:
                pass

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to all connected clients."""
        while True:
            try:
                await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)

                message = {
                    "type": MessageType.HEARTBEAT.value,
                    "timestamp": datetime.utcnow().isoformat()
                }

                disconnected = []
                for client_id, client in self.active_connections.items():
                    try:
                        await client.send_json(message)
                    except:
                        disconnected.append(client_id)

                # Clean up disconnected clients
                for client_id in disconnected:
                    await self.disconnect(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {str(e)}")

    async def _cleanup_loop(self):
        """Clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()
                stale_clients = []

                for client_id, client in self.active_connections.items():
                    # Check if client hasn't responded to heartbeat in 2 minutes
                    if (now - client.last_heartbeat).total_seconds() > 120:
                        stale_clients.append(client_id)

                # Disconnect stale clients
                for client_id in stale_clients:
                    logger.info(f"Disconnecting stale client {client_id}")
                    await self.disconnect(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            "total_connections": len(self.active_connections),
            "total_symbol_subscriptions": sum(len(s) for s in self.symbol_subscribers.values()),
            "total_channel_subscriptions": sum(len(s) for s in self.channel_subscribers.values()),
            "active_symbols": list(self.symbol_subscribers.keys()),
            "active_channels": list(self.channel_subscribers.keys()),
            "clients": [
                {
                    "client_id": client_id,
                    "connected_at": client.connected_at.isoformat(),
                    "subscribed_symbols": list(client.subscribed_symbols),
                    "subscribed_channels": list(client.subscribed_channels)
                }
                for client_id, client in self.active_connections.items()
            ]
        }