"""
Polymarket 15-Minute Bitcoin Market Data Collector
Collects real-time price data at 5-second intervals for ML analysis
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import websockets
import sqlite3
import requests
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """Represents a single price observation"""
    timestamp: int
    market_id: str
    token_id: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    volume: Optional[float] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None


class PolymarketDataCollector:
    """Collects Polymarket data at 5-second intervals"""

    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    WSS_MARKET = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, db_path: str = "polymarket_data.db"):
        self.db_path = db_path
        self.active_markets: Dict[str, Dict] = {}
        self.websocket = None
        self.is_collecting = False
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for storing price data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    market_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    price REAL NOT NULL,
                    bid REAL,
                    ask REAL,
                    spread REAL,
                    volume REAL,
                    window_start TEXT,
                    window_end TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON price_data(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_token
                ON price_data(market_id, token_id)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS markets (
                    market_id TEXT PRIMARY KEY,
                    question TEXT,
                    slug TEXT,
                    token_ids TEXT,
                    window_start TEXT,
                    window_end TEXT,
                    condition_id TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            logger.info(f"Database initialized at {self.db_path}")

    def get_active_15min_markets(self, limit: int = 100) -> List[Dict]:
        """
        Fetch currently active 15-minute Bitcoin markets

        Returns:
            List of active market dictionaries
        """
        try:
            # Search for Bitcoin markets
            response = requests.get(
                f"{self.GAMMA_API}/markets",
                params={
                    "tag_id": 235,  # Bitcoin tag
                    "closed": False,  # Only active markets
                    "limit": limit
                }
            )
            response.raise_for_status()
            markets = response.json()

            # Filter for 15-minute markets
            fifteen_min_markets = []
            for market in markets:
                # Check if it's a 15-minute market based on question/slug
                if any(term in market.get('question', '').lower() or
                      term in market.get('slug', '').lower()
                      for term in ['15m', '15 min', '15-min', '15 minute']):
                    fifteen_min_markets.append(market)

            logger.info(f"Found {len(fifteen_min_markets)} active 15-minute Bitcoin markets")
            return fifteen_min_markets

        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def store_market_info(self, market: Dict):
        """Store market metadata in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO markets
                (market_id, question, slug, token_ids, window_start, window_end, condition_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                market['id'],
                market.get('question'),
                market.get('slug'),
                json.dumps(market.get('clobTokenIds', [])),
                market.get('startDate'),
                market.get('endDate'),
                market.get('conditionId')
            ))

    def store_price_point(self, point: PricePoint):
        """Store a single price observation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO price_data
                (timestamp, market_id, token_id, price, bid, ask, spread, volume, window_start, window_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                point.timestamp,
                point.market_id,
                point.token_id,
                point.price,
                point.bid,
                point.ask,
                point.spread,
                point.volume,
                point.window_start,
                point.window_end
            ))

    async def connect_websocket(self, token_ids: List[str]):
        """
        Connect to Polymarket WebSocket for real-time price updates

        Args:
            token_ids: List of CLOB token IDs to monitor
        """
        try:
            logger.info(f"Connecting to WebSocket for {len(token_ids)} tokens...")

            async with websockets.connect(self.WSS_MARKET) as websocket:
                self.websocket = websocket

                # Subscribe to market data
                subscribe_msg = {
                    "assets_ids": token_ids,
                    "type": "MARKET"
                }

                await websocket.send(json.dumps(subscribe_msg))
                logger.info("WebSocket subscription sent")

                # Listen for messages
                async for message in websocket:
                    if not self.is_collecting:
                        break

                    await self.handle_websocket_message(message)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if self.is_collecting:
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
                await self.connect_websocket(token_ids)

    async def handle_websocket_message(self, message: str):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)

            # Extract price information based on message type
            if 'event_type' in data:
                event_type = data['event_type']

                if event_type in ['price_change', 'book', 'last_trade_price']:
                    timestamp = int(time.time())

                    # Extract relevant data
                    token_id = data.get('asset_id') or data.get('token_id')
                    market_id = data.get('market') or self._get_market_for_token(token_id)

                    price = data.get('price')
                    bid = data.get('bid')
                    ask = data.get('ask')

                    if price and token_id:
                        point = PricePoint(
                            timestamp=timestamp,
                            market_id=market_id,
                            token_id=token_id,
                            price=float(price),
                            bid=float(bid) if bid else None,
                            ask=float(ask) if ask else None,
                            spread=float(ask) - float(bid) if (ask and bid) else None
                        )

                        self.store_price_point(point)
                        logger.debug(f"Stored price point: {token_id} @ {price}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def _get_market_for_token(self, token_id: str) -> Optional[str]:
        """Get market ID for a given token ID"""
        for market_id, market in self.active_markets.items():
            if token_id in market.get('clobTokenIds', []):
                return market_id
        return None

    async def poll_prices(self, token_ids: List[str], interval: int = 5):
        """
        Fallback method: Poll prices at regular intervals using REST API
        Use this if WebSocket is not providing granular enough data

        Args:
            token_ids: List of token IDs to monitor
            interval: Polling interval in seconds (default: 5)
        """
        logger.info(f"Starting price polling every {interval} seconds")

        while self.is_collecting:
            timestamp = int(time.time())

            for token_id in token_ids:
                try:
                    # Get current price from CLOB API
                    response = requests.get(
                        f"{self.CLOB_API}/price",
                        params={"token_id": token_id},
                        timeout=3
                    )

                    if response.status_code == 200:
                        data = response.json()

                        point = PricePoint(
                            timestamp=timestamp,
                            market_id=self._get_market_for_token(token_id),
                            token_id=token_id,
                            price=float(data.get('price', 0)),
                            bid=float(data.get('bid')) if 'bid' in data else None,
                            ask=float(data.get('ask')) if 'ask' in data else None
                        )

                        self.store_price_point(point)

                except Exception as e:
                    logger.error(f"Error polling price for {token_id}: {e}")

            await asyncio.sleep(interval)

    async def collect_data(self, duration_days: int = 5, use_websocket: bool = True):
        """
        Main data collection loop

        Args:
            duration_days: Number of days to collect data
            use_websocket: Use WebSocket (True) or polling (False)
        """
        self.is_collecting = True

        # Get active markets
        markets = self.get_active_15min_markets()

        if not markets:
            logger.warning("No active 15-minute Bitcoin markets found")
            return

        # Store market info and collect token IDs
        token_ids = []
        for market in markets:
            self.active_markets[market['id']] = market
            self.store_market_info(market)
            token_ids.extend(market.get('clobTokenIds', []))

        logger.info(f"Monitoring {len(token_ids)} tokens across {len(markets)} markets")

        # Set end time
        end_time = datetime.now() + timedelta(days=duration_days)
        logger.info(f"Data collection will run until {end_time}")

        try:
            if use_websocket:
                # Use WebSocket for real-time data
                await self.connect_websocket(token_ids)
            else:
                # Use polling at 5-second intervals
                await self.poll_prices(token_ids, interval=5)

        except KeyboardInterrupt:
            logger.info("Collection stopped by user")
        finally:
            self.is_collecting = False

    def export_to_csv(self, output_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Export collected data to CSV for ML processing

        Args:
            output_path: Path to output CSV file
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
        """
        import pandas as pd

        query = "SELECT * FROM price_data WHERE 1=1"
        params = []

        if start_date:
            query += " AND datetime(timestamp, 'unixepoch') >= ?"
            params.append(start_date)

        if end_date:
            query += " AND datetime(timestamp, 'unixepoch') <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")

        return df


async def main():
    """Example usage"""
    collector = PolymarketDataCollector()

    # Collect data for 5 days
    await collector.collect_data(duration_days=5, use_websocket=True)


if __name__ == "__main__":
    asyncio.run(main())
