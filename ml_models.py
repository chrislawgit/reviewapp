"""
Machine Learning Models for Polymarket Pattern and Anomaly Detection
Includes LSTM for time-series prediction and Isolation Forest for anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import joblib

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logging.warning("TensorFlow not available. LSTM models will not work.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolymarketPatternDetector:
    """Detects patterns in Polymarket 15-minute market data using ML"""

    def __init__(self, model_type: str = 'lstm'):
        """
        Initialize pattern detector

        Args:
            model_type: Type of model ('lstm', 'random_forest', 'isolation_forest')
        """
        self.model_type = model_type
        self.model = None
        self.history = None

    def build_lstm_model(self, sequence_length: int, n_features: int) -> keras.Model:
        """
        Build LSTM model for time-series prediction

        Args:
            sequence_length: Number of time steps in each sequence
            n_features: Number of features per time step

        Returns:
            Compiled Keras model
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for LSTM models")

        model = models.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
            layers.Dropout(0.2),

            # Second LSTM layer
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),

            # Third LSTM layer
            layers.LSTM(32),
            layers.Dropout(0.2),

            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),

            # Output layer (regression for price prediction)
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )

        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        return model

    def build_lstm_classifier(self, sequence_length: int, n_features: int, n_classes: int = 2) -> keras.Model:
        """
        Build LSTM classifier for predicting market direction (up/down)

        Args:
            sequence_length: Number of time steps
            n_features: Number of features
            n_classes: Number of classes (default: 2 for binary up/down)

        Returns:
            Compiled Keras model
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for LSTM models")

        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.LSTM(32),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),

            layers.Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Built LSTM classifier with {model.count_params()} parameters")
        return model

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train LSTM model

        Args:
            X_train: Training sequences (n_samples, sequence_length, n_features)
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Dictionary with training history
        """
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]

        # Build model
        self.model = self.build_lstm_model(sequence_length, n_features)

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        logger.info("LSTM training completed")
        return self.history.history

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_estimators: int = 100) -> Dict:
        """
        Train Random Forest model for classification

        Args:
            X_train: Training features (2D array)
            y_train: Training labels
            n_estimators: Number of trees

        Returns:
            Training metrics
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Calculate feature importance
        feature_importance = self.model.feature_importances_

        logger.info("Random Forest training completed")
        return {'feature_importance': feature_importance}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input data

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)

        if self.model_type == 'lstm':
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)

            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse)
            }

            logger.info(f"LSTM Evaluation - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

        else:  # Classification
            # Convert to binary predictions
            pred_binary = (predictions > 0.5).astype(int).flatten()
            y_binary = (y_test > 0.5).astype(int).flatten() if y_test.ndim > 1 else y_test

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            metrics = {
                'accuracy': float(accuracy_score(y_binary, pred_binary)),
                'precision': float(precision_score(y_binary, pred_binary, average='binary')),
                'recall': float(recall_score(y_binary, pred_binary, average='binary')),
                'f1': float(f1_score(y_binary, pred_binary, average='binary'))
            }

            logger.info(f"Classification - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        return metrics

    def save_model(self, path: str):
        """Save trained model"""
        if self.model_type == 'lstm':
            self.model.save(path)
        else:
            joblib.dump(self.model, path)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        if self.model_type == 'lstm':
            self.model = keras.models.load_model(path)
        else:
            self.model = joblib.load(path)

        logger.info(f"Model loaded from {path}")


class AnomalyDetector:
    """Detects anomalies in market behavior using Isolation Forest"""

    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector

        Args:
            contamination: Expected proportion of anomalies (0.1 = 10%)
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        self.feature_names = None

    def fit(self, X: pd.DataFrame):
        """
        Fit anomaly detection model

        Args:
            X: Training data (DataFrame with features)
        """
        self.feature_names = X.columns.tolist()

        # Handle NaN values
        X_clean = X.fillna(0)

        self.model.fit(X_clean)
        logger.info(f"Isolation Forest trained on {len(X)} samples")

    def predict_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """
        Detect anomalies

        Args:
            X: Input data

        Returns:
            Array of predictions (1 = normal, -1 = anomaly)
        """
        X_clean = X[self.feature_names].fillna(0)
        predictions = self.model.predict(X_clean)

        anomaly_count = np.sum(predictions == -1)
        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(predictions)*100:.2f}%)")

        return predictions

    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous)

        Args:
            X: Input data

        Returns:
            Array of anomaly scores
        """
        X_clean = X[self.feature_names].fillna(0)
        scores = self.model.score_samples(X_clean)
        return scores

    def save_model(self, path: str):
        """Save anomaly detector"""
        joblib.dump(self.model, path)
        logger.info(f"Anomaly detector saved to {path}")

    def load_model(self, path: str):
        """Load anomaly detector"""
        self.model = joblib.load(path)
        logger.info(f"Anomaly detector loaded from {path}")


def train_complete_pipeline(data_dir: str = "ml_data", model_dir: str = "models"):
    """
    Complete training pipeline for both pattern detection and anomaly detection

    Args:
        data_dir: Directory with preprocessed data
        model_dir: Directory to save trained models
    """
    import os
    os.makedirs(model_dir, exist_ok=True)

    # Load preprocessed data
    logger.info("Loading preprocessed data...")

    # Load window aggregates for Random Forest
    window_df = pd.read_csv(f"{data_dir}/window_aggregates.csv")

    # Prepare features and target for window-level prediction
    feature_cols = [col for col in window_df.columns
                   if col not in ['window_id', 'market_id', 'token_id', 'outcome', 'timestamp_count']]

    X_windows = window_df[feature_cols].fillna(0)
    y_windows = window_df['outcome']

    # Train/test split for Random Forest
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_windows, y_windows, test_size=0.2, random_state=42
    )

    # Train Random Forest Classifier
    logger.info("\n=== Training Random Forest Classifier ===")
    rf_detector = PolymarketPatternDetector(model_type='random_forest')
    rf_detector.train_random_forest(X_train_rf.values, y_train_rf.values)
    rf_metrics = rf_detector.evaluate(X_test_rf.values, y_test_rf.values)
    rf_detector.save_model(f"{model_dir}/random_forest_classifier.pkl")

    # Train Anomaly Detector
    logger.info("\n=== Training Anomaly Detector ===")
    anomaly_detector = AnomalyDetector(contamination=0.05)
    anomaly_detector.fit(X_train_rf)
    anomalies = anomaly_detector.predict_anomalies(X_test_rf)
    anomaly_detector.save_model(f"{model_dir}/anomaly_detector.pkl")

    # Load sequences for LSTM (if TensorFlow available)
    if HAS_TENSORFLOW:
        logger.info("\n=== Training LSTM Model ===")
        X_seq = np.load(f"{data_dir}/sequences_X.npy")
        y_seq = np.load(f"{data_dir}/sequences_y.npy")

        # Remove NaN values
        valid_idx = ~np.isnan(y_seq)
        X_seq = X_seq[valid_idx]
        y_seq = y_seq[valid_idx]

        # Train/validation split
        X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )

        # Train LSTM
        lstm_detector = PolymarketPatternDetector(model_type='lstm')
        lstm_detector.train_lstm(
            X_train_lstm, y_train_lstm,
            X_val_lstm, y_val_lstm,
            epochs=30,
            batch_size=64
        )
        lstm_metrics = lstm_detector.evaluate(X_val_lstm, y_val_lstm)
        lstm_detector.save_model(f"{model_dir}/lstm_model.h5")

    logger.info("\n=== Training Complete ===")
    logger.info(f"Models saved to {model_dir}/")

    return {
        'random_forest_metrics': rf_metrics,
        'lstm_metrics': lstm_metrics if HAS_TENSORFLOW else None
    }


if __name__ == "__main__":
    # Train all models
    results = train_complete_pipeline()
    print("\nTraining Results:")
    print(results)
