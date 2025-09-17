# API Reference Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Common Headers](#common-headers)
5. [Error Responses](#error-responses)
6. [Endpoints](#endpoints)
   - [Health Check](#health-check)
   - [Data Endpoints](#data-endpoints)
   - [Validation Endpoints](#validation-endpoints)
   - [Collection Endpoints](#collection-endpoints)
   - [Backfill Endpoints](#backfill-endpoints)
   - [WebSocket Endpoints](#websocket-endpoints)

## Overview

The Fuzzy OSS20 Trading Platform API provides RESTful endpoints for accessing market data, managing data collection, performing validation, and real-time streaming via WebSockets.

### API Version
Current Version: `v1`

### Content Types
- Request: `application/json`
- Response: `application/json`
- WebSocket: `application/json` or `application/octet-stream`

## Authentication

### JWT Authentication
```http
Authorization: Bearer <jwt_token>
```

### Obtaining a Token
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## Base URL

```
Development: http://localhost:8000/api/v1
Production: https://api.fuzzyoss20.com/api/v1
```

## Common Headers

```http
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token>
X-Request-ID: <uuid>
X-Client-Version: 1.0.0
```

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "field": "Additional context"
    },
    "timestamp": "2025-01-17T10:30:00Z",
    "request_id": "uuid"
  }
}
```

### Common HTTP Status Codes
- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid request
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Access denied
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Endpoints

### Health Check

#### GET /health
Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-17T10:30:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "iqfeed": "connected",
    "arcticdb": "healthy"
  }
}
```

### Data Endpoints

#### GET /data/ticks/{symbol}
Retrieve tick data for a symbol.

**Parameters:**
- `symbol` (path) - Stock symbol (e.g., AAPL)
- `start_time` (query) - ISO 8601 timestamp
- `end_time` (query) - ISO 8601 timestamp
- `limit` (query) - Maximum records (default: 1000)
- `offset` (query) - Pagination offset

**Request:**
```http
GET /api/v1/data/ticks/AAPL?start_time=2025-01-17T09:30:00Z&end_time=2025-01-17T16:00:00Z&limit=100
```

**Response:**
```json
{
  "symbol": "AAPL",
  "count": 100,
  "data": [
    {
      "timestamp": "2025-01-17T09:30:00.123Z",
      "price": 150.25,
      "size": 100,
      "bid": 150.24,
      "ask": 150.26,
      "exchange": "NASDAQ",
      "conditions": "",
      "tick_id": 123456,
      "spread": 0.02,
      "spread_bps": 1.33
    }
  ],
  "metadata": {
    "start_time": "2025-01-17T09:30:00Z",
    "end_time": "2025-01-17T16:00:00Z",
    "total_count": 50000,
    "has_more": true
  }
}
```

#### GET /data/bars/{symbol}/{bar_type}/{interval}
Retrieve bar data for a symbol.

**Parameters:**
- `symbol` (path) - Stock symbol
- `bar_type` (path) - Bar type (time, tick, volume, dollar, renko, range)
- `interval` (path) - Interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- `start_time` (query) - ISO 8601 timestamp
- `end_time` (query) - ISO 8601 timestamp
- `limit` (query) - Maximum records

**Request:**
```http
GET /api/v1/data/bars/AAPL/time/1m?start_time=2025-01-17T09:30:00Z&limit=60
```

**Response:**
```json
{
  "symbol": "AAPL",
  "bar_type": "time",
  "interval": "1m",
  "count": 60,
  "data": [
    {
      "timestamp": "2025-01-17T09:30:00Z",
      "open": 150.20,
      "high": 150.35,
      "low": 150.18,
      "close": 150.30,
      "volume": 10000,
      "vwap": 150.28,
      "trade_count": 250
    }
  ]
}
```

#### GET /data/latest/{symbol}
Get the latest tick for a symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-01-17T15:59:59.999Z",
  "price": 151.50,
  "size": 200,
  "bid": 151.49,
  "ask": 151.51,
  "day_change": 1.25,
  "day_change_percent": 0.83
}
```

#### GET /data/metadata/{symbol}/{tier}
Get metadata for a symbol at specified tier.

**Parameters:**
- `symbol` (path) - Stock symbol
- `tier` (path) - Metadata tier (1, 2, or 3)
- `date` (query) - Date (YYYY-MM-DD)

**Response:**
```json
{
  "symbol": "AAPL",
  "tier": 1,
  "date": "2025-01-17",
  "metadata": {
    "spread_mean": 0.02,
    "spread_std": 0.005,
    "volume_profile": {...},
    "price_levels": {...},
    "computed_at": "2025-01-17T16:00:00Z"
  }
}
```

#### POST /data/search
Advanced tick data search with filters.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "filters": {
    "price_min": 150.00,
    "price_max": 152.00,
    "size_min": 100,
    "exchange": "NASDAQ",
    "start_time": "2025-01-17T09:30:00Z",
    "end_time": "2025-01-17T10:30:00Z"
  },
  "limit": 500,
  "sort": "timestamp",
  "order": "desc"
}
```

#### GET /data/symbols
Get list of available symbols.

**Response:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
  "count": 5,
  "updated_at": "2025-01-17T16:00:00Z"
}
```

#### GET /data/summary/{symbol}
Get data summary statistics for a symbol.

**Parameters:**
- `lookback_days` (query) - Number of days to analyze (default: 30)

**Response:**
```json
{
  "symbol": "AAPL",
  "date_range": {
    "start": "2024-12-18T00:00:00Z",
    "end": "2025-01-17T23:59:59Z"
  },
  "tick_statistics": {
    "total_ticks": 1500000,
    "daily_average": 50000,
    "unique_days": 21
  },
  "price_statistics": {
    "min": 148.50,
    "max": 152.75,
    "mean": 150.25,
    "std": 1.23
  },
  "volume_statistics": {
    "total_volume": 2000000000,
    "daily_average": 95238095,
    "mean_trade_size": 250
  }
}
```

### Validation Endpoints

#### POST /validation/validate
Run validation checks on data.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "start_date": "2025-01-16",
  "end_date": "2025-01-17",
  "validation_types": ["session", "storage", "timezone", "continuity"]
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-01-17T10:30:00Z",
  "results": [
    {
      "validation_type": "session",
      "status": "passed",
      "message": "All ticks within valid trading sessions",
      "details": {
        "total_ticks": 50000,
        "sessions_checked": ["premarket", "regular", "afterhours"]
      }
    },
    {
      "validation_type": "timezone",
      "status": "passed",
      "message": "All timestamps in Eastern Time"
    }
  ],
  "overall_status": "passed"
}
```

#### GET /validation/status/{symbol}
Get validation status for a symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "last_validated": "2025-01-17T10:00:00Z",
  "status": "healthy",
  "validations": {
    "session": "passed",
    "storage": "passed",
    "timezone": "passed",
    "continuity": "warning"
  }
}
```

#### GET /validation/summary
Get validation summary for all symbols.

**Parameters:**
- `lookback_days` (query) - Days to look back (default: 7)

**Response:**
```json
{
  "timestamp": "2025-01-17T10:30:00Z",
  "symbols_checked": 5,
  "overall_status": "warning",
  "results": {
    "AAPL": {
      "passed": 3,
      "warnings": 1,
      "failed": 0
    }
  }
}
```

### Collection Endpoints

#### POST /collection/historical
Collect historical tick data.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "num_days": 5,
  "max_ticks": 100000
}
```

**Response:**
```json
{
  "task_id": "hist_AAPL_20250117103000",
  "status": "started",
  "symbol": "AAPL",
  "estimated_time": 30,
  "message": "Historical collection initiated"
}
```

#### POST /collection/realtime/start
Start real-time data collection.

**Request Body:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT"]
}
```

**Response:**
```json
{
  "status": "started",
  "added_symbols": ["AAPL", "GOOGL"],
  "already_streaming": ["MSFT"],
  "total_streaming": 3
}
```

#### POST /collection/realtime/stop
Stop real-time data collection.

**Request Body:**
```json
{
  "symbols": ["AAPL"]
}
```

**Response:**
```json
{
  "status": "stopped",
  "stopped_symbols": ["AAPL"],
  "remaining_streaming": 2
}
```

#### GET /collection/status
Get collection status.

**Response:**
```json
{
  "streaming_symbols": ["GOOGL", "MSFT"],
  "active_tasks": 2,
  "tasks": [
    {
      "task_id": "hist_AAPL_20250117103000",
      "symbol": "AAPL",
      "collection_type": "historical",
      "status": "collecting",
      "ticks_collected": 45000,
      "created_at": "2025-01-17T10:30:00Z"
    }
  ]
}
```

#### GET /collection/task/{task_id}
Get specific task status.

**Response:**
```json
{
  "task_id": "hist_AAPL_20250117103000",
  "symbol": "AAPL",
  "status": "completed",
  "ticks_collected": 100000,
  "duration_seconds": 25,
  "completed_at": "2025-01-17T10:30:25Z"
}
```

### Backfill Endpoints

#### POST /backfill/detect-gaps
Detect data gaps for backfilling.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "start_date": "2025-01-01",
  "end_date": "2025-01-17"
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "gaps_found": 3,
  "gaps": [
    {
      "start": "2025-01-05T09:30:00Z",
      "end": "2025-01-05T10:15:00Z",
      "duration_minutes": 45,
      "severity": "high"
    }
  ]
}
```

#### POST /backfill/start
Start backfill process.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "start_date": "2025-01-05",
  "end_date": "2025-01-06"
}
```

**Response:**
```json
{
  "task_id": "backfill_AAPL_20250117103000",
  "status": "started",
  "estimated_ticks": 100000,
  "estimated_time_minutes": 5
}
```

#### GET /backfill/status/{task_id}
Get backfill task status.

**Response:**
```json
{
  "task_id": "backfill_AAPL_20250117103000",
  "status": "in_progress",
  "progress_percent": 65,
  "ticks_processed": 65000,
  "ticks_total": 100000,
  "current_date": "2025-01-05",
  "errors": []
}
```

### WebSocket Endpoints

#### WS /ws/{client_id}
WebSocket connection for real-time data.

**Connection:**
```javascript
ws://localhost:8000/ws/client_123
```

**Subscribe Message:**
```json
{
  "type": "subscribe",
  "symbols": ["AAPL", "GOOGL"],
  "channels": ["validation", "system"]
}
```

**Unsubscribe Message:**
```json
{
  "type": "unsubscribe",
  "symbols": ["AAPL"],
  "channels": ["validation"]
}
```

**Tick Data Message (Server → Client):**
```json
{
  "type": "tick",
  "symbol": "AAPL",
  "data": {
    "timestamp": "2025-01-17T10:30:00.123Z",
    "price": 150.25,
    "size": 100,
    "bid": 150.24,
    "ask": 150.26
  },
  "sequence": 12345
}
```

**Bar Data Message (Server → Client):**
```json
{
  "type": "bar",
  "symbol": "AAPL",
  "interval": "1m",
  "data": {
    "timestamp": "2025-01-17T10:30:00Z",
    "open": 150.20,
    "high": 150.35,
    "low": 150.18,
    "close": 150.30,
    "volume": 10000
  }
}
```

**Validation Alert (Server → Client):**
```json
{
  "type": "validation",
  "channel": "validation",
  "data": {
    "symbol": "AAPL",
    "validation_type": "continuity",
    "status": "warning",
    "message": "Data gap detected",
    "timestamp": "2025-01-17T10:30:00Z"
  }
}
```

**Heartbeat:**
```json
{
  "type": "heartbeat",
  "timestamp": "2025-01-17T10:30:00Z"
}
```

**Error Message:**
```json
{
  "type": "error",
  "message": "Invalid subscription request",
  "code": "INVALID_REQUEST",
  "timestamp": "2025-01-17T10:30:00Z"
}
```

## Rate Limiting

### Limits
- **REST API**: 1000 requests per minute
- **WebSocket**: 100 messages per second
- **Data Endpoints**: 100 requests per minute
- **Collection Endpoints**: 10 requests per minute

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642412400
```

### Rate Limit Exceeded Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "retry_after": 60
  }
}
```

## Pagination

### Query Parameters
- `limit` - Number of items per page (max: 1000)
- `offset` - Number of items to skip
- `cursor` - Cursor for cursor-based pagination

### Paginated Response
```json
{
  "data": [...],
  "pagination": {
    "limit": 100,
    "offset": 0,
    "total": 5000,
    "has_more": true,
    "next_cursor": "eyJpZCI6MTAwfQ=="
  }
}
```

## Webhooks

### Webhook Configuration
```http
POST /api/v1/webhooks
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["tick.new", "validation.failed", "collection.complete"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload
```json
{
  "event": "validation.failed",
  "timestamp": "2025-01-17T10:30:00Z",
  "data": {
    "symbol": "AAPL",
    "validation_type": "continuity",
    "message": "Large data gap detected"
  },
  "signature": "sha256=abcdef..."
}
```

## SDK Examples

### Python
```python
import requests

class FuzzyAPI:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

    def get_ticks(self, symbol, start_time, end_time):
        response = requests.get(
            f'{self.base_url}/data/ticks/{symbol}',
            params={
                'start_time': start_time,
                'end_time': end_time
            },
            headers=self.headers
        )
        return response.json()
```

### JavaScript/TypeScript
```typescript
class FuzzyAPI {
  constructor(
    private baseUrl: string,
    private token: string
  ) {}

  async getTicks(
    symbol: string,
    startTime: string,
    endTime: string
  ): Promise<TickData[]> {
    const response = await fetch(
      `${this.baseUrl}/data/ticks/${symbol}?` +
      `start_time=${startTime}&end_time=${endTime}`,
      {
        headers: {
          'Authorization': `Bearer ${this.token}`,
          'Content-Type': 'application/json'
        }
      }
    );
    return response.json();
  }
}
```

## Testing

### Test Environment
```
Base URL: https://api-test.fuzzyoss20.com/api/v1
Test API Key: test_key_123456789
```

### Postman Collection
Import the Postman collection from: `/docs/postman/fuzzy-api.json`

### cURL Examples
```bash
# Get latest tick
curl -H "Authorization: Bearer $TOKEN" \
  https://api.fuzzyoss20.com/api/v1/data/latest/AAPL

# Start collection
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"]}' \
  https://api.fuzzyoss20.com/api/v1/collection/realtime/start
```

---

*API Version: 1.0.0*
*Last Updated: January 17, 2025*