# -*- coding: utf-8 -*-
# lstm_prediction_pipeline.py
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import logging
from typing import Dict, Any, Optional, Tuple
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import your DatabaseWriter
from LSTM_Database_writer import DatabaseWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTMPredictionPipeline:
    """Complete pipeline for reading data, making predictions, and writing results"""
    
    def __init__(self, connection_string: str, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize the prediction pipeline
        
        Args:
            connection_string: Database connection string
            model_path: Path to saved LSTM model (.h5 or .keras file)
            scaler_path: Path to saved scaler (optional, .pkl file)
        """
        self.connection_string = connection_string
        self.db_writer = DatabaseWriter(connection_string)
        self.connection = None
        
        # Load model and scaler
        logger.info("Loading LSTM model...")
        try:
            # Try loading with compile=False to avoid metric deserialization issues
            self.model = keras.models.load_model(model_path, compile=False)
            logger.info(f"✓ Model loaded from {model_path}")
            
            # Recompile the model with fresh metrics
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            logger.info("✓ Model recompiled successfully")
        except Exception as e:
            logger.error(f"Failed to load with compile=False, trying custom objects...")
            # Try with custom objects for backward compatibility
            try:
                from tensorflow.keras.losses import MeanSquaredError
                from tensorflow.keras.metrics import MeanAbsoluteError
                
                custom_objects = {
                    'mse': MeanSquaredError(),
                    'mae': MeanAbsoluteError()
                }
                self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
                logger.info(f"✓ Model loaded with custom objects from {model_path}")
            except Exception as e2:
                logger.error(f"All loading attempts failed: {e2}")
                raise
        
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✓ Scaler loaded from {scaler_path}")
        else:
            self.scaler = StandardScaler()
            logger.warning("⚠ No scaler provided, will fit on data")
        
        # Model configuration
        self.sequence_length = self.model.input_shape[1]  # Get from model
        self.n_features = self.model.input_shape[2]
        
        # Feature names (adjust based on your actual features)
        # IMPORTANT: Must match the features used during training (11 features)
        self.feature_columns = [
            'filter_dp',
            'seal_gas_flow',
            'seal_gas_diff_pressure',
            'seal_gas_temp',
            'primary_vent_flow',
            'primary_vent_pressure',
            'secondary_seal_gas_flow',
            'separation_seal_gas_flow',
            'separation_seal_gas_pressure',
            'seal_gas_to_vent_diff_pressure',
            'encoding'
        ]
        
        self._connect()
    
    def _connect(self):
        """Establish database connection for reading"""
        try:
            self.connection = pyodbc.connect(
                self.connection_string,
                timeout=30
            )
            logger.info("✓ Database connection established for reading")
        except Exception as e:
            logger.error(f"✗ Failed to connect to database: {str(e)}")    
            raise
    
    def read_data_from_db(self, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         compressor_id: Optional[str] = None,
                         table_name: str = "dbo.[compressor_EandA_dataset (v1)]") -> pd.DataFrame:  # ← UPDATE THIS  
        """
        Read raw data from database
        
        Args:
            start_date: Start of date range (default: last 7 days)
            end_date: End of date range (default: now)
            compressor_id: Specific compressor to filter (optional)
            table_name: Name of source data table (UPDATE THIS TO YOUR ACTUAL TABLE NAME)
            
        Returns:
            DataFrame with raw sensor data
        """
        try:
            # Default to last 7 days if not specified
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=7)
            
            # Build query
            query = f"""
                SELECT 
                    datetime,
                    compressor_id,
                    filter_dp,
                    seal_gas_flow,
                    seal_gas_diff_pressure,
                    seal_gas_temp,
                    primary_vent_flow,
                    primary_vent_pressure,
                    secondary_seal_gas_flow,
                    separation_seal_gas_flow,
                    separation_seal_gas_pressure,
                    seal_gas_to_vent_diff_pressure,
                    encoding
                FROM {table_name}
                WHERE datetime BETWEEN ? AND ?
            """
            
            params = [start_date, end_date]
            
            if compressor_id:
                query += " AND compressor_id = ?"
                params.append(compressor_id)
            
            query += " ORDER BY compressor_id, datetime"
            logger.info(f"Reading data from table: {table_name}")
            logger.info(f"Reading data from {start_date} to {end_date}...")
            df = pd.read_sql(query, self.connection, params=params)
            
            if len(df) == 0:
                logger.warning("⚠ No data found for specified criteria")
                return pd.DataFrame()
            
            logger.info(f"✓ Read {len(df)} records from database")
            logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            logger.info(f"  Compressors: {df['compressor_id'].nunique()}")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error reading data from database: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Preprocess data for LSTM model
        
        Args:
            df: Raw dataframe with sensor data
            
        Returns:
            Tuple of (sequences array, metadata dataframe)
        """
        try:
            logger.info("Preprocessing data...")
            
            # DEBUG: Check what columns are actually in the dataframe
            logger.info(f"Columns in dataframe: {df.columns.tolist()}")
            logger.info(f"Expected features: {self.feature_columns}")
            
            # Check for missing features
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                logger.error(f"✗ Missing features in data: {missing_features}")
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Handle missing values
            df = df.copy()
            for col in self.feature_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())
            
            # Group by compressor to maintain temporal order
            sequences = []
            metadata = []
            
            for compressor_id in df['compressor_id'].unique():
                comp_data = df[df['compressor_id'] == compressor_id].sort_values('datetime')
                
                # Extract features
                features = comp_data[self.feature_columns].values
                
                # Scale features
                if not hasattr(self.scaler, 'mean_'):
                    # Fit scaler if not already fitted
                    self.scaler.fit(features)
                
                features_scaled = self.scaler.transform(features)
                
                # Create sequences
                for i in range(len(features_scaled) - self.sequence_length + 1):
                    seq = features_scaled[i:i + self.sequence_length]
                    sequences.append(seq)
                    
                    # Store metadata for this sequence
                    end_idx = i + self.sequence_length - 1
                    metadata.append({
                        'sequence_index': len(sequences) - 1,
                        'compressor_id': compressor_id,
                        'datetime': comp_data.iloc[end_idx]['datetime'],
                        'start_datetime': comp_data.iloc[i]['datetime'],
                        'end_datetime': comp_data.iloc[end_idx]['datetime']
                    })
            
            sequences_array = np.array(sequences)
            metadata_df = pd.DataFrame(metadata)
            
            logger.info(f"✓ Created {len(sequences_array)} sequences")
            logger.info(f"  Sequence shape: {sequences_array.shape}")
            
            return sequences_array, metadata_df
            
        except Exception as e:
            logger.error(f"✗ Error preprocessing data: {str(e)}")
            raise
    
    def make_predictions(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions using LSTM model
        
        Args:
            sequences: Preprocessed sequences array
            
        Returns:
            Reconstructions array
        """
        try:
            logger.info("Making predictions...")
            
            # Predict in batches to avoid memory issues
            batch_size = 256
            reconstructions = self.model.predict(sequences, batch_size=batch_size, verbose=0)
            
            logger.info(f"✓ Generated {len(reconstructions)} predictions")
            
            return reconstructions
            
        except Exception as e:
            logger.error(f"✗ Error making predictions: {str(e)}")
            raise
    
    def calculate_anomaly_scores(self, 
                                 sequences: np.ndarray,
                                 reconstructions: np.ndarray,
                                 metadata_df: pd.DataFrame,
                                 threshold_percentile: float = 95.0,
                                 consecutive_threshold: int = 3) -> pd.DataFrame:
        """
        Calculate reconstruction errors and detect anomalies
        
        Args:
            sequences: Original sequences
            reconstructions: Reconstructed sequences
            metadata_df: Metadata for each sequence
            threshold_percentile: Percentile for threshold calculation
            consecutive_threshold: Number of consecutive anomalies to trigger alert
            
        Returns:
            DataFrame with all results and anomaly scores
        """
        try:
            logger.info("Calculating anomaly scores...")
            
            results_df = metadata_df.copy()
            
            # Calculate reconstruction error (MSE per sequence)
            mse_per_sequence = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
            results_df['reconstruction_error'] = mse_per_sequence
            
            # Calculate per-feature errors
            feature_errors = np.mean(np.square(sequences - reconstructions), axis=1)  # Average over time steps
            
            for i, feature_name in enumerate(self.feature_columns):
                results_df[f'error_{feature_name}'] = feature_errors[:, i]
            
            # Add encoding column (optional: latent space representation if encoder available)
            results_df['error_encoding'] = 0.0  # Placeholder
            
            # Calculate threshold
            threshold = np.percentile(mse_per_sequence, threshold_percentile)
            results_df['threshold'] = threshold
            
            # Detect point anomalies
            results_df['is_anomaly_point'] = results_df['reconstruction_error'] > threshold
            
            # Calculate anomaly score (normalized distance above threshold)
            results_df['anomaly_score'] = results_df['reconstruction_error'] / threshold
            
            # Detect consecutive anomalies (triggered anomalies)
            results_df['consecutive_count'] = 0
            results_df['triggered_anomaly'] = False
            
            # Process by compressor to maintain sequence
            for compressor_id in results_df['compressor_id'].unique():
                mask = results_df['compressor_id'] == compressor_id
                comp_indices = results_df[mask].index
                
                consecutive_count = 0
                for idx in comp_indices:
                    if results_df.loc[idx, 'is_anomaly_point']:
                        consecutive_count += 1
                        results_df.loc[idx, 'consecutive_count'] = consecutive_count
                        
                        if consecutive_count >= consecutive_threshold:
                            results_df.loc[idx, 'triggered_anomaly'] = True
                    else:
                        consecutive_count = 0
            
            # Summary statistics
            total_sequences = len(results_df)
            anomaly_points = results_df['is_anomaly_point'].sum()
            triggered_anomalies = results_df['triggered_anomaly'].sum()
            
            logger.info(f"✓ Anomaly detection complete:")
            logger.info(f"  Total sequences: {total_sequences}")
            logger.info(f"  Anomaly points: {anomaly_points} ({anomaly_points/total_sequences*100:.2f}%)")
            logger.info(f"  Triggered anomalies: {triggered_anomalies} ({triggered_anomalies/total_sequences*100:.2f}%)")
            logger.info(f"  Threshold (MSE): {threshold:.6f}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"✗ Error calculating anomaly scores: {str(e)}")
            raise
    
    def run_pipeline(self,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    compressor_id: Optional[str] = None,
                    threshold_percentile: float = 95.0,
                    consecutive_threshold: int = 3,
                    source_table: str = "dbo.[compressor_EandA_dataset (v1)]") -> str:  # ← ADD THIS
        """
        Run the complete prediction pipeline
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            compressor_id: Specific compressor (optional)
            threshold_percentile: Percentile for anomaly threshold
            consecutive_threshold: Consecutive anomalies to trigger alert
            source_table: Name of the source data table in database
            
        Returns:
            Batch ID for this run
        """
        try:
            batch_id = str(uuid.uuid4())
            logger.info(f"Starting pipeline run - Batch ID: {batch_id}")
            
            # Step 1: Read data
            df = self.read_data_from_db(start_date, end_date, compressor_id, source_table)
            
            if len(df) == 0:
                logger.warning("⚠ No data to process")
                return batch_id
            
            # Step 2: Preprocess
            sequences, metadata_df = self.preprocess_data(df)
            
            # Step 3: Make predictions
            reconstructions = self.make_predictions(sequences)
            
            # Step 4: Calculate anomaly scores
            results_df = self.calculate_anomaly_scores(
                sequences, 
                reconstructions, 
                metadata_df,
                threshold_percentile,
                consecutive_threshold
            )
            
            # Step 5: Write to database
            logger.info("Writing results to database...")
            
            success = self.db_writer.write_predictions(results_df, batch_id)
            
            if success:
                # Write summary
                model_info = {
                    'model_version': 'v1.0',
                    'threshold': float(results_df['threshold'].iloc[0]),
                    'sequence_length': self.sequence_length
                }
                self.db_writer.write_batch_summary(results_df, batch_id, model_info)
                
                # Write feature statistics
                self.db_writer.write_feature_statistics(results_df, batch_id)
                
                logger.info(f"✓ Pipeline completed successfully - Batch ID: {batch_id}")
            else:
                logger.error("✗ Failed to write results to database")
            
            return batch_id
            
        except Exception as e:
            logger.error(f"✗ Pipeline failed: {str(e)}")
            raise
    
    def close(self):
        """Close all connections"""
        if self.connection:
            self.connection.close()
        self.db_writer.close()
        logger.info("✓ All connections closed")


# Main execution
if __name__ == "__main__":
    # Configuration
    # IMPORTANT: Update the DRIVER to match what's installed on your system
    # Run: print(pyodbc.drivers()) to see available drivers
    
    # Try these in order until one works:
    # Option 1: ODBC Driver 18
    # CONNECTION_STRING = (
    #     "DRIVER={ODBC Driver 18 for SQL Server};"
    #     "SERVER=yokogawa-sqlserver-x2g77kzajj6gy.database.windows.net;"
    #     "DATABASE=AOM_Dev;"
    #     "UID=xmadmin;"
    #     "PWD=Yokogawa1234;"
    #     "Encrypt=yes;"
    #     "TrustServerCertificate=yes;"
    # )
    
    # Option 2: ODBC Driver 17 (most common)
    CONNECTION_STRING = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=yokogawa-sqlserver-x2g77kzajj6gy.database.windows.net;"
        "DATABASE=AOM-Dev;"
        "UID=xmadmin;"
        "PWD=NnDmyJEfQkw9;"
    )
    
    # Option 3: ODBC Driver 13
    # CONNECTION_STRING = (
    #     "DRIVER={ODBC Driver 13 for SQL Server};"
    #     "SERVER=yokogawa-sqlserver-x2g77kzajj6gy.database.windows.net;"
    #     "DATABASE=AOM_Dev;"
    #     "UID=xmadmin;"
    #     "PWD=Yokogawa1234;"
    # )
    
    # Option 4: SQL Server (older, generic driver)
    # CONNECTION_STRING = (
    #     "DRIVER={SQL Server};"
    #     "SERVER=yokogawa-sqlserver-x2g77kzajj6gy.database.windows.net;"
    #     "DATABASE=AOM_Dev;"
    #     "UID=xmadmin;"
    #     "PWD=Yokogawa1234;"
    # )
    
    # Option 5: SQL Server Native Client (older)
    # CONNECTION_STRING = (
    #     "DRIVER={SQL Server Native Client 11.0};"
    #     "SERVER=yokogawa-sqlserver-x2g77kzajj6gy.database.windows.net;"
    #     "DATABASE=AOM_Dev;"
    #     "UID=xmadmin;"
    #     "PWD=Yokogawa1234;"
    # )
    
    MODEL_PATH = "C:/Users/adminuser/Projects/LSTM_Autoencoder_detection/models/compressor_autoencoder2.h5"  # Path to your saved model
    SCALER_PATH = "C:/Users/adminuser/Projects/LSTM_Autoencoder_detection/models/scaler2.pkl"  # Path to your saved scaler (optional)
    
    try:
        # Initialize pipeline
        pipeline = LSTMPredictionPipeline(
            connection_string=CONNECTION_STRING,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH
        )
        
        # Run pipeline for last 7 days
        # IMPORTANT: Replace 'dbo.YourActualTableName' with your real table name!
        batch_id = pipeline.run_pipeline(
            start_date=datetime.now() - timedelta(days=21),
            end_date=datetime.now(),
            threshold_percentile=95.0,
            consecutive_threshold=3,
            source_table="dbo.[compressor_EandA_dataset (v1)]"  # ← UPDATE THIS!
        )
        
        print(f"\n{'='*60}")
        print(f"Pipeline completed successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"{'='*60}\n")
        
        # Optional: Retrieve and display summary
        recent_batches = pipeline.db_writer.get_recent_batches(limit=5)
        if len(recent_batches) > 0:
            print("\nRecent Batches:")
            print(recent_batches.to_string(index=False))
        
        # Close connections
        pipeline.close()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise