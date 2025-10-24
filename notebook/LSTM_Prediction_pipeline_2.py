# -*- coding: utf-8 -*-
# lstm_prediction_pipeline_compressor_aware_final.py
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import logging
from typing import Dict, Any, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from LSTM_Database_writer import DatabaseWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer


# Define SelectDecoderOutput custom layer (must match training script)
class SelectDecoderOutput(Layer):
    """Custom layer to select decoder output based on compressor ID"""
    
    def __init__(self, **kwargs):
        super(SelectDecoderOutput, self).__init__(**kwargs)
    
    def call(self, inputs):
        outputs = inputs[:-1]
        comp_id = inputs[-1]
        stacked = tf.stack(outputs, axis=1)
        batch_size = tf.shape(comp_id)[0]
        comp_id = tf.squeeze(comp_id, axis=-1)
        batch_indices = tf.range(batch_size)
        indices = tf.stack([batch_indices, comp_id], axis=1)
        selected = tf.gather_nd(stacked, indices)
        return selected
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
    
    def get_config(self):
        config = super(SelectDecoderOutput, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CompressorAwareLSTMPredictionPipeline:
    """Complete pipeline for compressor-aware LSTM predictions"""
    
    def __init__(self, connection_string: str, model_path: str, 
                 scaler_path: str, encoder_path: str, thresholds_path: str):
        """
        Initialize the compressor-aware prediction pipeline
        
        Args:
            connection_string: Database connection string
            model_path: Path to saved compressor-aware LSTM model (no Lambda)
            scaler_path: Path to saved feature scaler
            encoder_path: Path to saved compressor ID encoder
            thresholds_path: Path to saved per-compressor thresholds
        """
        self.connection_string = connection_string
        self.db_writer = DatabaseWriter(connection_string)
        self.connection = None
        
        # Load compressor-aware model
        logger.info("Loading compressor-aware LSTM model (no Lambda version)...")
        
        try:
            # Custom objects for loading
            custom_objects = {
                'SelectDecoderOutput': SelectDecoderOutput,
                'mse': 'mse',  # Handle old metric names
                'mae': 'mae'
            }
            
            # Load model with compile=False to avoid metric issues
            self.model = keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            
            # Recompile with fresh metrics
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"✓ Model loaded and recompiled from {model_path}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            logger.error("\nMake sure:")
            logger.error("1. Model was trained with SelectDecoderOutput (no Lambda)")
            logger.error("2. Model file exists at the specified path")
            logger.error(f"3. Path: {model_path}")
            raise
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        logger.info(f"✓ Scaler loaded from {scaler_path}")
        
        # Load compressor encoder
        self.compressor_encoder = joblib.load(encoder_path)
        logger.info(f"✓ Compressor encoder loaded from {encoder_path}")
        logger.info(f"  Known compressors: {list(self.compressor_encoder.classes_)}")
        
        # Load per-compressor thresholds
        self.thresholds = joblib.load(thresholds_path)
        logger.info(f"✓ Thresholds loaded from {thresholds_path}")
        for comp, thresh in self.thresholds.items():
            logger.info(f"    {comp}: {thresh:.6f}")
        
        # Model configuration
        self.sequence_length = self.model.input[0].shape[1]
        self.n_features = self.model.input[0].shape[2]
        
        # Feature columns (must match training)
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
    
    def inspect_table(self, table_name: str):
        """Inspect table structure and content"""
        try:
            # Clean table name - remove dbo. prefix if present
            clean_table = table_name.replace('dbo.', '', 1).strip()
            escaped_table = f"[{clean_table}]" if not clean_table.startswith('[') else clean_table
            
            print(f"\n{'='*60}")
            print(f"Inspecting table: {clean_table}")
            print(f"{'='*60}")
            
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM dbo.{escaped_table}"
            count_df = pd.read_sql(count_query, self.connection)
            print(f"\n📊 Total rows: {count_df['count'].iloc[0]:,}")
            
            # Get column info
            column_query = f"""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """
            # Extract just the table name without schema
            table_name_only = clean_table.replace('[', '').replace(']', '')
            columns_df = pd.read_sql(column_query, self.connection, params=(table_name_only,))
            
            print(f"\n📋 Columns ({len(columns_df)}):")
            print(columns_df.to_string(index=False))
            
            # Get sample data
            sample_query = f"SELECT TOP 5 * FROM dbo.{escaped_table}"
            sample_df = pd.read_sql(sample_query, self.connection)
            
            print(f"\n🔍 Sample data (first 5 rows):")
            print(sample_df.to_string(index=False))
            
            # Get date range
            if 'datetime' in sample_df.columns:
                date_query = f"""
                    SELECT 
                        MIN(datetime) as min_date,
                        MAX(datetime) as max_date
                    FROM dbo.{escaped_table}
                """
                date_df = pd.read_sql(date_query, self.connection)
                print(f"\n📅 Date range:")
                print(f"  Min: {date_df['min_date'].iloc[0]}")
                print(f"  Max: {date_df['max_date'].iloc[0]}")
            
        except Exception as e:
            print(f"❌ Error inspecting table: {str(e)}")
            raise
    
    def read_data_from_db(self, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         compressor_id: Optional[str] = None,
                         table_name: str = "dbo.compressor_normal_dataset3") -> pd.DataFrame:
        """Read raw data from database"""
        try:
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=7)
            
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
            
            logger.info(f"📌 Reading data from {start_date} to {end_date}...")
            df = pd.read_sql(query, self.connection, params=params)
            
            if len(df) == 0:
                logger.warning("⚠ No data found for specified criteria")
                return pd.DataFrame()
            
            logger.info(f"✓ Read {len(df):,} records from database")
            logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            logger.info(f"  Compressors: {df['compressor_id'].nunique()}")
            
            # Check for unknown compressors
            unknown_compressors = set(df['compressor_id'].unique()) - set(self.compressor_encoder.classes_)
            if unknown_compressors:
                logger.warning(f"⚠ Unknown compressors found (will be skipped): {unknown_compressors}")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error reading data from database: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Preprocess data for compressor-aware LSTM model
        
        Returns:
            Tuple of (sequences array, compressor_ids array, metadata dataframe)
        """
        try:
            logger.info("🔧 Preprocessing data...")
            
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
            
            # Filter out unknown compressors
            known_compressors = set(self.compressor_encoder.classes_)
            df = df[df['compressor_id'].isin(known_compressors)]
            
            if len(df) == 0:
                logger.error("✗ No data remaining after filtering unknown compressors")
                raise ValueError("No valid compressor data to process")
            
            # Encode compressor IDs
            df['compressor_id_encoded'] = self.compressor_encoder.transform(df['compressor_id'])
            
            sequences = []
            compressor_ids = []
            metadata = []
            
            for compressor_id in df['compressor_id'].unique():
                comp_data = df[df['compressor_id'] == compressor_id].sort_values('datetime')
                
                logger.info(f"🔧 Processing {compressor_id}: {len(comp_data)} records")
                
                # Extract and scale features
                features = comp_data[self.feature_columns].values
                features_scaled = self.scaler.transform(features)
                
                # Get encoded compressor ID
                comp_id_encoded = comp_data['compressor_id_encoded'].iloc[0]
                
                # Create sequences
                for i in range(len(features_scaled) - self.sequence_length + 1):
                    seq = features_scaled[i:i + self.sequence_length]
                    sequences.append(seq)
                    compressor_ids.append(comp_id_encoded)
                    
                    end_idx = i + self.sequence_length - 1
                    metadata.append({
                        'sequence_index': len(sequences) - 1,
                        'compressor_id': compressor_id,
                        'compressor_id_encoded': comp_id_encoded,
                        'datetime': comp_data.iloc[end_idx]['datetime'],
                        'start_datetime': comp_data.iloc[i]['datetime'],
                        'end_datetime': comp_data.iloc[end_idx]['datetime']
                    })
            
            sequences_array = np.array(sequences)
            compressor_ids_array = np.array(compressor_ids)
            metadata_df = pd.DataFrame(metadata)
            
            logger.info(f"✓ Created {len(sequences_array):,} sequences")
            logger.info(f"  Sequence shape: {sequences_array.shape}")
            logger.info(f"  Compressor IDs shape: {compressor_ids_array.shape}")
            
            return sequences_array, compressor_ids_array, metadata_df
            
        except Exception as e:
            logger.error(f"✗ Error preprocessing data: {str(e)}")
            raise
    
    def make_predictions(self, sequences: np.ndarray, compressor_ids: np.ndarray) -> np.ndarray:
        """Make predictions using compressor-aware LSTM model"""
        try:
            logger.info("🤖 Making predictions with compressor-aware model...")
            
            batch_size = 256
            reconstructions = self.model.predict(
                [sequences, compressor_ids], 
                batch_size=batch_size, 
                verbose=0
            )
            
            logger.info(f"✓ Generated {len(reconstructions):,} predictions")
            
            return reconstructions
            
        except Exception as e:
            logger.error(f"✗ Error making predictions: {str(e)}")
            raise
    
    def calculate_anomaly_scores(self, 
                                 sequences: np.ndarray,
                                 reconstructions: np.ndarray,
                                 metadata_df: pd.DataFrame,
                                 consecutive_threshold: int = 3) -> pd.DataFrame:
        """Calculate reconstruction errors using per-compressor thresholds"""
        try:
            logger.info("📊 Calculating anomaly scores with per-compressor thresholds...")
            
            results_df = metadata_df.copy()
            
            # Calculate reconstruction error (MSE per sequence)
            mse_per_sequence = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
            results_df['reconstruction_error'] = mse_per_sequence
            
            # Calculate per-feature errors
            feature_errors = np.mean(np.square(sequences - reconstructions), axis=1)
            
            for i, feature_name in enumerate(self.feature_columns):
                results_df[f'error_{feature_name}'] = feature_errors[:, i]
            
            # Apply per-compressor thresholds
            results_df['threshold'] = results_df['compressor_id'].map(self.thresholds)
            
            # Handle any compressors without thresholds
            if results_df['threshold'].isna().any():
                global_threshold = np.percentile(mse_per_sequence, 95)
                results_df['threshold'].fillna(global_threshold, inplace=True)
                logger.warning(f"⚠ Some compressors missing thresholds, using global: {global_threshold:.6f}")
            
            # Detect point anomalies
            results_df['is_anomaly_point'] = results_df['reconstruction_error'] > results_df['threshold']
            
            # Calculate anomaly score
            results_df['anomaly_score'] = results_df['reconstruction_error'] / results_df['threshold']
            
            # Detect consecutive anomalies
            results_df['consecutive_count'] = 0
            results_df['triggered_anomaly'] = False
            
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
            
            logger.info(f"\n{'='*80}")
            logger.info(f"✓ ANOMALY DETECTION SUMMARY:")
            logger.info(f"{'='*80}")
            logger.info(f"   Total sequences: {total_sequences:,}")
            logger.info(f"   Anomaly points: {anomaly_points:,} ({anomaly_points/total_sequences*100:.2f}%)")
            logger.info(f"   Triggered anomalies: {triggered_anomalies:,} ({triggered_anomalies/total_sequences*100:.2f}%)")
            
            # Per-compressor summary
            logger.info(f"\n📊 Per-Compressor Results:")
            for comp in results_df['compressor_id'].unique():
                comp_mask = results_df['compressor_id'] == comp
                comp_anomalies = results_df[comp_mask]['is_anomaly_point'].sum()
                comp_total = comp_mask.sum()
                comp_threshold = self.thresholds.get(comp, 0)
                logger.info(f"   {comp}: {comp_anomalies}/{comp_total} anomalies ({comp_anomalies/comp_total*100:.1f}%), threshold={comp_threshold:.6f}")
            
            logger.info(f"{'='*80}\n")
            
            return results_df
            
        except Exception as e:
            logger.error(f"✗ Error calculating anomaly scores: {str(e)}")
            raise
    
    def run_pipeline(self,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    compressor_id: Optional[str] = None,
                    consecutive_threshold: int = 3,
                    source_table: str = "dbo.compressor_normal_dataset3") -> str:
        """Run the complete compressor-aware prediction pipeline"""
        try:
            batch_id = str(uuid.uuid4())
            
            logger.info(f"\n{'#'*80}")
            logger.info(f"🚀 STARTING COMPRESSOR-AWARE PIPELINE")
            logger.info(f"{'#'*80}")
            logger.info(f"📋 Batch ID: {batch_id}")
            logger.info(f"📅 Date Range: {start_date} to {end_date}")
            logger.info(f"{'#'*80}\n")
            
            # Step 1: Read data
            logger.info(f"{'='*80}")
            logger.info(f"STEP 1: READING DATA FROM DATABASE")
            logger.info(f"{'='*80}")
            df = self.read_data_from_db(start_date, end_date, compressor_id, source_table)
            
            if len(df) == 0:
                logger.warning("⚠ No data to process - PIPELINE ABORTED")
                return batch_id
            
            # Step 2: Preprocess
            logger.info(f"\n{'='*80}")
            logger.info(f"STEP 2: PREPROCESSING DATA")
            logger.info(f"{'='*80}")
            sequences, compressor_ids, metadata_df = self.preprocess_data(df)
            
            # Step 3: Make predictions
            logger.info(f"\n{'='*80}")
            logger.info(f"STEP 3: MAKING PREDICTIONS")
            logger.info(f"{'='*80}")
            reconstructions = self.make_predictions(sequences, compressor_ids)
            
            # Step 4: Calculate anomaly scores
            logger.info(f"\n{'='*80}")
            logger.info(f"STEP 4: CALCULATING ANOMALY SCORES")
            logger.info(f"{'='*80}")
            results_df = self.calculate_anomaly_scores(
                sequences, 
                reconstructions, 
                metadata_df,
                consecutive_threshold
            )
            
            # Step 5: Write to database
            logger.info(f"\n{'='*80}")
            logger.info(f"STEP 5: WRITING RESULTS TO DATABASE")
            logger.info(f"{'='*80}")
            
            success = self.db_writer.write_predictions(results_df, batch_id)
            
            if success:
                model_info = {
                    'model_version': 'compressor_aware_v1.0',
                    'model_type': 'compressor_aware_lstm',
                    'sequence_length': self.sequence_length,
                    'known_compressors': list(self.compressor_encoder.classes_),
                    'per_compressor_thresholds': self.thresholds
                }
                self.db_writer.write_batch_summary(results_df, batch_id, model_info)
                self.db_writer.write_feature_statistics(results_df, batch_id)
                
                logger.info(f"\n{'#'*80}")
                logger.info(f"✓ PIPELINE COMPLETED SUCCESSFULLY")
                logger.info(f"{'#'*80}")
                logger.info(f"📋 Batch ID: {batch_id}")
                logger.info(f"{'#'*80}\n")
            else:
                logger.error("✗ Failed to write results to database")
            
            return batch_id
            
        except Exception as e:
            logger.error(f"\n{'#'*80}")
            logger.error(f"✗ PIPELINE FAILED")
            logger.error(f"{'#'*80}")
            logger.error(f"Error: {str(e)}")
            logger.error(f"{'#'*80}\n")
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
    CONNECTION_STRING = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=yokogawa-sqlserver-x2g77kzajj6gy.database.windows.net;"
        "DATABASE=AOM-Dev;"
        "UID=xmadmin;"
        "PWD=NnDmyJEfQkw9;"
    )
    
    # Paths for compressor-aware model files (NO LAMBDA VERSION)
    MODEL_DIR = "models"
    MODEL_PATH = f"{MODEL_DIR}/compressor_aware_model_no_lambda2.h5"
    SCALER_PATH = f"{MODEL_DIR}/feature_scaler_no_lambda2.pkl"
    ENCODER_PATH = f"{MODEL_DIR}/compressor_encoder_no_lambda2.pkl"
    THRESHOLDS_PATH = f"{MODEL_DIR}/thresholds_no_lambda2.pkl"
    
    # Source table (don't include 'dbo.' - it's added automatically)
    SOURCE_TABLE = "compressor_normal_dataset3"
    
    try:
        logger.info(f"\n{'#'*80}")
        logger.info(f"🔍 COMPRESSOR-AWARE LSTM PREDICTION PIPELINE")
        logger.info(f"{'#'*80}\n")
        
        # Initialize pipeline
        logger.info("Initializing compressor-aware pipeline...")
        pipeline = CompressorAwareLSTMPredictionPipeline(
            connection_string=CONNECTION_STRING,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            encoder_path=ENCODER_PATH,
            thresholds_path=THRESHOLDS_PATH
        )
        
        # Optional: Inspect table first
        logger.info(f"\n{'='*80}")
        logger.info(f"PRE-FLIGHT CHECK: INSPECTING TABLE")
        logger.info(f"{'='*80}\n")
        pipeline.inspect_table(SOURCE_TABLE)
        
        # Set date range
        start_date = datetime.now() - timedelta(days=19)
        end_date = datetime.now()
        
        # Run pipeline
        batch_id = pipeline.run_pipeline(
            start_date=start_date,
            end_date=end_date,
            consecutive_threshold=3,
            source_table=SOURCE_TABLE
        )
        
        # Display recent batches
        logger.info(f"{'='*80}")
        logger.info(f"📊 RECENT BATCH SUMMARIES")
        logger.info(f"{'='*80}")
        recent_batches = pipeline.db_writer.get_recent_batches(limit=5)
        if len(recent_batches) > 0:
            print("\nRecent Batches:")
            print(recent_batches.to_string(index=False))
        
        # Close connections
        pipeline.close()
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"✓✓✓ ALL OPERATIONS COMPLETED SUCCESSFULLY ✓✓✓")
        logger.info(f"{'#'*80}\n")
        
    except Exception as e:
        logger.error(f"\n{'#'*80}")
        logger.error(f"❌ FATAL ERROR")
        logger.error(f"{'#'*80}")
        logger.error(f"Error: {str(e)}")
        logger.error(f"{'#'*80}\n")
        
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        raise