# Polymarket 15-Minute Bitcoin Market Data Collection & ML Analysis

Complete system for collecting, analyzing, and detecting patterns in Polymarket's 15-minute Bitcoin prediction markets.

## Overview

This system enables you to:
1. **Collect data** at 5-second intervals from live markets via WebSocket
2. **Store data** in SQLite database for efficient querying
3. **Preprocess data** with feature engineering for ML
4. **Train ML models** to detect patterns and anomalies
5. **Predict outcomes** and identify unusual market behavior

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Polymarket APIs                             │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐          │
│  │ Gamma API  │  │  CLOB API  │  │  WebSocket   │          │
│  └────────────┘  └────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               polymarket_collector.py                        │
│  • Fetches active 15-min markets                            │
│  • Connects to WebSocket for real-time prices               │
│  • Samples data every 5 seconds                             │
│  • Stores in SQLite database                                │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 SQLite Database                              │
│  • price_data table (timestamped observations)              │
│  • markets table (market metadata)                          │
│  • Indexed for fast queries                                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            data_preprocessing.py                             │
│  • Loads data from database                                 │
│  • Creates 15-minute windows                                │
│  • Engineers features (momentum, volatility, etc.)          │
│  • Detects statistical anomalies                            │
│  • Creates sequences for time-series models                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   ml_models.py                               │
│  • LSTM: Time-series pattern detection                      │
│  • Random Forest: Window outcome classification             │
│  • Isolation Forest: Anomaly detection                      │
│  • Training & evaluation pipelines                          │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```python
import tensorflow as tf
import pandas as pd
import websockets
print("All dependencies installed successfully!")
```

## Usage

### Phase 1: Data Collection (5 Days)

Collect data at 5-second intervals for all 15-minute Bitcoin markets over a 5-day period.

```python
import asyncio
from polymarket_collector import PolymarketDataCollector

# Initialize collector
collector = PolymarketDataCollector(db_path="bitcoin_15min_data.db")

# Start collecting data for 5 days
asyncio.run(collector.collect_data(duration_days=5, use_websocket=True))
```

**What happens:**
- Fetches all active 15-minute Bitcoin markets from Gamma API
- Connects to CLOB WebSocket for real-time price updates
- Samples prices every 5 seconds
- Stores data in SQLite database
- Automatically handles reconnections and errors

**Expected data volume:**
- 96 windows per day × 5 days = 480 windows
- 180 data points per window (15 min ÷ 5 sec)
- Total: ~86,400 data points

### Phase 2: Data Preprocessing

Transform raw price data into ML-ready features.

```python
from data_preprocessing import PolymarketDataPreprocessor

# Initialize preprocessor
preprocessor = PolymarketDataPreprocessor(db_path="bitcoin_15min_data.db")

# Export processed data for ML
preprocessor.export_for_ml(output_dir="ml_data")
```

**Generated features:**
- **Price dynamics:** changes, momentum, velocity, acceleration
- **Statistical:** rolling means, std dev, min/max over 60s windows
- **Market microstructure:** bid-ask spreads, spread changes
- **Temporal:** hour, day of week, cyclical time encodings
- **Window aggregates:** per-15min window statistics

**Output files:**
```
ml_data/
├── processed_data.csv          # Full dataset with all features
├── window_aggregates.csv       # One row per 15-min window
├── sequences_X.npy             # Time-series sequences for LSTM
├── sequences_y.npy             # Target values
├── scaler.pkl                  # Fitted StandardScaler
└── feature_columns.txt         # Feature names
```

### Phase 3: Model Training

Train machine learning models to detect patterns and anomalies.

```python
from ml_models import train_complete_pipeline

# Train all models
results = train_complete_pipeline(
    data_dir="ml_data",
    model_dir="models"
)

print(results)
```

**Trained models:**

1. **LSTM (Time-Series Predictor)**
   - Predicts price movements within 15-minute windows
   - Uses 3-minute sequences (36 × 5-second observations)
   - Outputs: predicted price change percentage

2. **Random Forest (Window Classifier)**
   - Classifies entire 15-minute windows as UP or DOWN
   - Uses aggregate features per window
   - Outputs: binary prediction + confidence

3. **Isolation Forest (Anomaly Detector)**
   - Identifies unusual market behavior
   - Detects price manipulation, flash crashes, etc.
   - Outputs: anomaly score (-1 = anomalous, 1 = normal)

### Phase 4: Analysis & Insights

Analyze patterns and anomalies discovered by the models.

```python
from ml_models import PolymarketPatternDetector, AnomalyDetector
import pandas as pd
import numpy as np

# Load test data
window_df = pd.read_csv("ml_data/window_aggregates.csv")
X_test = window_df.drop(['window_id', 'market_id', 'token_id', 'outcome'], axis=1)

# Load trained models
rf_model = PolymarketPatternDetector(model_type='random_forest')
rf_model.load_model("models/random_forest_classifier.pkl")

anomaly_model = AnomalyDetector()
anomaly_model.load_model("models/anomaly_detector.pkl")

# Make predictions
predictions = rf_model.predict(X_test)
anomalies = anomaly_model.predict_anomalies(X_test)

# Find anomalous windows
anomalous_windows = window_df[anomalies == -1]
print(f"Found {len(anomalous_windows)} anomalous windows")
print(anomalous_windows[['window_id', 'price_change_pct_total', 'price_std', 'spread_mean']])
```

## Data Collection Strategies

### Real-Time Collection (Recommended)

Uses WebSocket for instant updates:

```python
collector = PolymarketDataCollector()
await collector.collect_data(duration_days=5, use_websocket=True)
```

**Pros:**
- True 5-second granularity
- Low latency
- Efficient bandwidth usage

**Cons:**
- Requires stable internet connection
- Only works for currently active markets

### Polling-Based Collection (Fallback)

Uses REST API with 5-second polling:

```python
collector = PolymarketDataCollector()
await collector.collect_data(duration_days=5, use_websocket=False)
```

**Pros:**
- More robust to connection issues
- Works with any market

**Cons:**
- Higher API usage
- Potential rate limiting
- Slightly less precise timing

### Historical Backfill

For completed markets, use the CLOB API:

```python
import requests
from datetime import datetime

# Example: Get historical data for a specific market
market_token_id = "YOUR_TOKEN_ID"

# November 30, 2025, 12:00-12:15 PM
start_ts = int(datetime(2025, 11, 30, 12, 0).timestamp())
end_ts = int(datetime(2025, 11, 30, 12, 15).timestamp())

response = requests.get(
    "https://clob.polymarket.com/prices-history",
    params={
        "market": market_token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": 1  # 1-minute resolution (best available)
    }
)

history = response.json()
print(f"Retrieved {len(history['history'])} data points")
```

**Note:** Historical API only provides 1-minute fidelity, not 5-second.

## Understanding the Data

### Price Data Structure

Each observation contains:

```python
{
    "timestamp": 1733061605,      # Unix timestamp
    "market_id": "516926",        # Market identifier
    "token_id": "0x1234...",      # Outcome token
    "price": 0.547,               # Probability (0-1)
    "bid": 0.545,                 # Best bid price
    "ask": 0.549,                 # Best ask price
    "spread": 0.004,              # Bid-ask spread
    "window_start": "2025-11-30T12:00:00Z",
    "window_end": "2025-11-30T12:15:00Z"
}
```

### 15-Minute Window Lifecycle

```
12:00:00 - Market opens
12:00:05 - First observation
12:00:10 - Second observation
...
12:14:55 - Last observation (before close)
12:15:00 - Market resolves (determines outcome)
```

Each window has:
- **180 observations** (900 seconds ÷ 5 seconds)
- **Opening price** (at 12:00:00)
- **Closing price** (at 12:14:55)
- **Outcome** (UP if close > open, DOWN otherwise)

## Pattern Detection Examples

### 1. Momentum Patterns

Detect when price shows strong directional movement:

```python
# High positive momentum = strong upward trend
high_momentum = window_df[window_df['momentum_60s_mean'] > 0.05]

# Accelerating movement
accelerating = window_df[window_df['velocity_std'] > 0.01]
```

### 2. Volatility Patterns

Identify periods of high uncertainty:

```python
# High volatility windows
volatile = window_df[window_df['price_std'] > 0.15]

# Sudden volatility spikes
volatility_spikes = window_df[
    window_df['price_std'] > window_df['price_std'].mean() + 2 * window_df['price_std'].std()
]
```

### 3. Anomaly Patterns

Markets behaving unusually:

```python
# Get anomaly scores
scores = anomaly_model.get_anomaly_scores(X_test)

# Most anomalous windows (lowest scores = most anomalous)
top_anomalies = window_df.iloc[np.argsort(scores)[:10]]
```

## Performance Optimization

### Database Indexing

The database is pre-indexed on critical fields:
- `timestamp` - for time-range queries
- `(market_id, token_id)` - for market-specific queries

### Memory-Efficient Processing

For large datasets, process in chunks:

```python
# Process data in batches
chunk_size = 10000

for chunk in pd.read_sql_query(
    "SELECT * FROM price_data ORDER BY timestamp",
    conn,
    chunksize=chunk_size
):
    # Process chunk
    processed = preprocessor.calculate_features(chunk)
    # Save or analyze
```

### Parallel Processing

Speed up data collection for multiple markets:

```python
import asyncio

async def collect_multiple_markets(market_ids):
    tasks = []
    for market_id in market_ids:
        collector = PolymarketDataCollector(db_path=f"market_{market_id}.db")
        tasks.append(collector.collect_data(duration_days=5))

    await asyncio.gather(*tasks)
```

## Troubleshooting

### No Active Markets Found

```python
# Check manually
import requests
response = requests.get("https://gamma-api.polymarket.com/markets?tag_id=235&closed=false&limit=100")
markets = response.json()
print(f"Found {len(markets)} Bitcoin markets")
```

### WebSocket Connection Issues

```python
# Use polling instead
collector.collect_data(duration_days=5, use_websocket=False)
```

### Rate Limiting

Free tier: 1,000 requests/hour

```python
# Add delays between requests
import time
time.sleep(3.6)  # 1000 requests/hour = 1 per 3.6 seconds
```

### Missing Data Points

Check database:

```python
import sqlite3
conn = sqlite3.connect("bitcoin_15min_data.db")
cursor = conn.execute("""
    SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
    FROM price_data
""")
print(cursor.fetchone())
```

## Advanced Use Cases

### 1. Live Trading Signals

```python
# Real-time predictions
async def live_prediction_loop():
    while True:
        # Get latest data
        latest_data = get_latest_60_seconds()

        # Predict next movement
        prediction = lstm_model.predict(latest_data)

        if prediction > threshold:
            print("BUY signal")
        else:
            print("SELL signal")

        await asyncio.sleep(5)
```

### 2. Market Efficiency Analysis

```python
# Compare predicted vs actual outcomes
comparison = pd.DataFrame({
    'predicted': predictions,
    'actual': y_test
})

# Calculate prediction accuracy
accuracy = (comparison['predicted'].round() == comparison['actual']).mean()
print(f"Market efficiency: {accuracy:.2%}")
```

### 3. Cross-Market Analysis

```python
# Correlate multiple 15-min windows
correlations = window_df.groupby('window_id')['price_change_pct_total'].corr()
print("Cross-market correlation:", correlations.mean())
```

## API Reference

### Polymarket Endpoints Used

| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `https://gamma-api.polymarket.com/markets` | Find active markets | 1000/hour |
| `https://clob.polymarket.com/prices-history` | Historical prices | 1000/hour |
| `wss://ws-subscriptions-clob.polymarket.com/ws/market` | Real-time prices | Unlimited |

## Data Sources

- [Polymarket Documentation](https://docs.polymarket.com/)
- [Historical Timeseries Data](https://docs.polymarket.com/developers/CLOB/timeseries)
- [WebSocket Overview](https://docs.polymarket.com/developers/CLOB/websocket/wss-overview)
- [Gamma Markets API](https://docs.polymarket.com/developers/gamma-markets-api/get-markets)

## License

MIT License - Free to use for research and analysis

## Disclaimer

This tool is for educational and research purposes. Always comply with Polymarket's Terms of Service and API usage policies. Not financial advice.
