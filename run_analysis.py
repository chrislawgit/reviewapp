#!/usr/bin/env python3
"""
Complete Polymarket 15-Minute Bitcoin Market Analysis Pipeline
Run this script to execute the full data collection and analysis workflow
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from polymarket_collector import PolymarketDataCollector
from data_preprocessing import PolymarketDataPreprocessor
from ml_models import train_complete_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PolymarketAnalysisPipeline:
    """Complete pipeline for data collection, preprocessing, and ML analysis"""

    def __init__(self, db_path: str = "polymarket_data.db",
                 data_dir: str = "ml_data",
                 model_dir: str = "models"):
        self.db_path = db_path
        self.data_dir = data_dir
        self.model_dir = model_dir

        # Create directories
        Path(self.data_dir).mkdir(exist_ok=True)
        Path(self.model_dir).mkdir(exist_ok=True)

    async def run_data_collection(self, duration_days: int = 5, use_websocket: bool = True):
        """
        Step 1: Collect data from Polymarket

        Args:
            duration_days: Number of days to collect data
            use_websocket: Use WebSocket (True) or polling (False)
        """
        logger.info(f"\n{'='*60}")
        logger.info("STEP 1: DATA COLLECTION")
        logger.info(f"{'='*60}")
        logger.info(f"Duration: {duration_days} days")
        logger.info(f"Method: {'WebSocket' if use_websocket else 'Polling (5-sec intervals)'}")
        logger.info(f"Database: {self.db_path}")

        collector = PolymarketDataCollector(db_path=self.db_path)

        # Check for active markets first
        markets = collector.get_active_15min_markets()

        if not markets:
            logger.warning("No active 15-minute Bitcoin markets found!")
            logger.info("This could mean:")
            logger.info("  1. No 15-minute markets are currently active")
            logger.info("  2. All markets have closed")
            logger.info("  3. API is temporarily unavailable")
            logger.info("\nTrying to continue anyway...")

        # Start collection
        try:
            await collector.collect_data(
                duration_days=duration_days,
                use_websocket=use_websocket
            )
        except KeyboardInterrupt:
            logger.info("\nData collection stopped by user")
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            raise

    def run_preprocessing(self):
        """
        Step 2: Preprocess collected data

        Returns:
            Dataset dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info(f"{'='*60}")

        preprocessor = PolymarketDataPreprocessor(db_path=self.db_path)

        # Load and check data
        df = preprocessor.load_data()

        if len(df) == 0:
            logger.error("No data found in database!")
            logger.info("Make sure data collection completed successfully")
            return None

        logger.info(f"Loaded {len(df)} price observations")

        # Export processed data
        preprocessor.export_for_ml(output_dir=self.data_dir)

        logger.info(f"Preprocessed data exported to {self.data_dir}/")

        return True

    def run_training(self):
        """
        Step 3: Train ML models

        Returns:
            Training results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info("STEP 3: MODEL TRAINING")
        logger.info(f"{'='*60}")

        try:
            results = train_complete_pipeline(
                data_dir=self.data_dir,
                model_dir=self.model_dir
            )

            logger.info(f"\nTraining completed successfully!")
            logger.info(f"Models saved to {self.model_dir}/")

            return results

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def run_analysis(self):
        """
        Step 4: Analyze results and generate insights
        """
        import pandas as pd
        from ml_models import PolymarketPatternDetector, AnomalyDetector

        logger.info(f"\n{'='*60}")
        logger.info("STEP 4: ANALYSIS & INSIGHTS")
        logger.info(f"{'='*60}")

        # Load window aggregates
        window_df = pd.read_csv(f"{self.data_dir}/window_aggregates.csv")
        logger.info(f"Analyzing {len(window_df)} windows")

        # Prepare features
        feature_cols = [col for col in window_df.columns
                       if col not in ['window_id', 'market_id', 'token_id', 'outcome', 'timestamp_count']]
        X = window_df[feature_cols].fillna(0)

        # Load models
        logger.info("\nLoading trained models...")

        try:
            # Random Forest
            rf_model = PolymarketPatternDetector(model_type='random_forest')
            rf_model.load_model(f"{self.model_dir}/random_forest_classifier.pkl")

            # Anomaly detector
            anomaly_model = AnomalyDetector()
            anomaly_model.load_model(f"{self.model_dir}/anomaly_detector.pkl")

            # Make predictions
            logger.info("\nGenerating predictions...")
            predictions = rf_model.predict(X.values)
            anomalies = anomaly_model.predict_anomalies(X)
            anomaly_scores = anomaly_model.get_anomaly_scores(X)

            # Analysis
            logger.info(f"\n{'='*60}")
            logger.info("RESULTS SUMMARY")
            logger.info(f"{'='*60}")

            # Window outcomes
            actual_up = (window_df['outcome'] == 1).sum()
            predicted_up = (predictions > 0.5).sum()

            logger.info(f"\nWindow Outcomes:")
            logger.info(f"  Actual UP: {actual_up} ({actual_up/len(window_df)*100:.1f}%)")
            logger.info(f"  Predicted UP: {predicted_up} ({predicted_up/len(window_df)*100:.1f}%)")

            # Anomalies
            n_anomalies = (anomalies == -1).sum()
            logger.info(f"\nAnomalies Detected:")
            logger.info(f"  Count: {n_anomalies} ({n_anomalies/len(window_df)*100:.1f}%)")

            # Top anomalies
            logger.info(f"\nTop 5 Most Anomalous Windows:")
            top_anomaly_idx = anomaly_scores.argsort()[:5]

            for idx in top_anomaly_idx:
                window = window_df.iloc[idx]
                logger.info(f"  Window {window['window_id']}:")
                logger.info(f"    Price change: {window['price_change_pct_total']*100:.2f}%")
                logger.info(f"    Volatility: {window['price_std']:.4f}")
                logger.info(f"    Anomaly score: {anomaly_scores[idx]:.4f}")

            # Patterns
            logger.info(f"\nPattern Analysis:")

            # High volatility windows
            high_vol = window_df[window_df['price_std'] > window_df['price_std'].quantile(0.9)]
            logger.info(f"  High volatility windows: {len(high_vol)} (top 10%)")

            # Strong momentum
            if 'momentum_60s_mean' in window_df.columns:
                strong_momentum = window_df[
                    abs(window_df['momentum_60s_mean']) > window_df['momentum_60s_mean'].std()
                ]
                logger.info(f"  Strong momentum windows: {len(strong_momentum)}")

            # Save detailed results
            results_df = pd.DataFrame({
                'window_id': window_df['window_id'],
                'market_id': window_df['market_id'],
                'actual_outcome': window_df['outcome'],
                'predicted_outcome': (predictions > 0.5).astype(int).flatten(),
                'prediction_confidence': predictions.flatten(),
                'is_anomaly': (anomalies == -1).astype(int),
                'anomaly_score': anomaly_scores,
                'price_change_pct': window_df['price_change_pct_total']
            })

            output_path = f"{self.data_dir}/analysis_results.csv"
            results_df.to_csv(output_path, index=False)
            logger.info(f"\nDetailed results saved to {output_path}")

        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            logger.info("Make sure training completed successfully")

    async def run_complete_pipeline(self, duration_days: int = 5, use_websocket: bool = True):
        """
        Run the complete analysis pipeline

        Args:
            duration_days: Days to collect data
            use_websocket: Use WebSocket or polling
        """
        logger.info(f"\n{'#'*60}")
        logger.info("POLYMARKET 15-MINUTE BITCOIN MARKET ANALYSIS")
        logger.info(f"{'#'*60}")
        logger.info(f"Start time: {datetime.now()}")

        # Step 1: Data Collection
        await self.run_data_collection(duration_days, use_websocket)

        # Step 2: Preprocessing
        if not self.run_preprocessing():
            logger.error("Preprocessing failed. Exiting.")
            return

        # Step 3: Training
        self.run_training()

        # Step 4: Analysis
        self.run_analysis()

        logger.info(f"\n{'#'*60}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'#'*60}")
        logger.info(f"End time: {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket 15-Minute Bitcoin Market Analysis Pipeline"
    )

    parser.add_argument(
        '--mode',
        choices=['collect', 'preprocess', 'train', 'analyze', 'full'],
        default='full',
        help='Pipeline mode (default: full)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=5,
        help='Days to collect data (default: 5)'
    )

    parser.add_argument(
        '--use-polling',
        action='store_true',
        help='Use polling instead of WebSocket (default: False)'
    )

    parser.add_argument(
        '--db-path',
        default='polymarket_data.db',
        help='Database path (default: polymarket_data.db)'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = PolymarketAnalysisPipeline(db_path=args.db_path)

    # Run based on mode
    if args.mode == 'collect':
        asyncio.run(pipeline.run_data_collection(
            duration_days=args.duration,
            use_websocket=not args.use_polling
        ))

    elif args.mode == 'preprocess':
        pipeline.run_preprocessing()

    elif args.mode == 'train':
        pipeline.run_training()

    elif args.mode == 'analyze':
        pipeline.run_analysis()

    elif args.mode == 'full':
        asyncio.run(pipeline.run_complete_pipeline(
            duration_days=args.duration,
            use_websocket=not args.use_polling
        ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nPipeline stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
