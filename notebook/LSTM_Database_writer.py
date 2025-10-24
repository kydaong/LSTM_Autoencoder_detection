# -*- coding: utf-8 -*-
# LSTM_Database_writer.py
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseWriter:
    """Handles all database operations for LSTM predictions"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = pyodbc.connect(
                self.connection_string,
                timeout=30
            )
            self.connection.autocommit = False
            logger.info("✓ Database connection established")
        except Exception as e:
            logger.error(f"✗ Failed to connect to database: {str(e)}")
            raise
    
    def _ensure_connection(self):
        """Ensure database connection is active"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        except:
            logger.warning("Database connection lost, reconnecting...")
            self._connect()
    
    def write_predictions(self, results: pd.DataFrame, batch_id: str) -> bool:
        """
        Write prediction results to database matching your exact column structure
        
        Args:
            results: DataFrame with all your columns
            batch_id: Unique identifier for this batch
            
        Returns:
            bool: Success status
        """
        try:
            self._ensure_connection()
            cursor = self.connection.cursor()
            
            # Prepare data for bulk insert using itertuples for better performance
            predictions_data = []
            for row in results.itertuples(index=False):
                # Handle timestamp column (might be 'timestamp' or 'datetime')
                timestamp_value = None
                if hasattr(row, 'timestamp'):
                    timestamp_value = pd.to_datetime(row.timestamp)
                elif hasattr(row, 'datetime'):
                    timestamp_value = pd.to_datetime(row.datetime)
                elif hasattr(row, 'end_timestamp'):
                    timestamp_value = pd.to_datetime(row.end_timestamp)
                else:
                    timestamp_value = datetime.now()
                
                # Handle compressor_id
                compressor_id_value = 'Unknown'
                if hasattr(row, 'compressor_id'):
                    compressor_id_value = str(row.compressor_id)
                elif hasattr(row, 'CompressorID'):
                    compressor_id_value = str(row.CompressorID)
                
                prediction_record = (
                    batch_id,
                    datetime.now(),
                    int(row.sequence_index),
                    float(row.reconstruction_error),
                    float(row.threshold),
                    bool(row.is_anomaly_point),
                    bool(row.triggered_anomaly),
                    float(row.anomaly_score),
                    int(row.consecutive_count),
                    float(row.error_filter_dp) if hasattr(row, 'error_filter_dp') and pd.notna(row.error_filter_dp) else None,
                    float(row.error_seal_gas_flow) if hasattr(row, 'error_seal_gas_flow') and pd.notna(row.error_seal_gas_flow) else None,
                    float(row.error_seal_gas_diff_pressure) if hasattr(row, 'error_seal_gas_diff_pressure') and pd.notna(row.error_seal_gas_diff_pressure) else None,
                    float(row.error_seal_gas_temp) if hasattr(row, 'error_seal_gas_temp') and pd.notna(row.error_seal_gas_temp) else None,
                    float(row.error_primary_vent_flow) if hasattr(row, 'error_primary_vent_flow') and pd.notna(row.error_primary_vent_flow) else None,
                    float(row.error_primary_vent_pressure) if hasattr(row, 'error_primary_vent_pressure') and pd.notna(row.error_primary_vent_pressure) else None,
                    float(row.error_secondary_seal_gas_flow) if hasattr(row, 'error_secondary_seal_gas_flow') and pd.notna(row.error_secondary_seal_gas_flow) else None,
                    float(row.error_separation_seal_gas_flow) if hasattr(row, 'error_separation_seal_gas_flow') and pd.notna(row.error_separation_seal_gas_flow) else None,
                    float(row.error_separation_seal_gas_pressure) if hasattr(row, 'error_separation_seal_gas_pressure') and pd.notna(row.error_separation_seal_gas_pressure) else None,
                    float(row.error_seal_gas_to_vent_diff_pressure) if hasattr(row, 'error_seal_gas_to_vent_diff_pressure') and pd.notna(row.error_seal_gas_to_vent_diff_pressure) else None,
                    float(row.error_encoding) if hasattr(row, 'error_encoding') and pd.notna(row.error_encoding) else None,
                    timestamp_value,
                    compressor_id_value
                )
                predictions_data.append(prediction_record)
            
            # Bulk insert using fast_executemany
            insert_sql = """
                INSERT INTO dbo.LSTM_Predictions (
                    BatchID, ProcessedAt,
                    sequence_index, reconstruction_error, threshold,
                    is_anomaly_point, triggered_anomaly, anomaly_score, consecutive_count,
                    error_filter_dp, error_seal_gas_flow, error_seal_gas_diff_pressure,
                    error_seal_gas_temp, error_primary_vent_flow, error_primary_vent_pressure,
                    error_secondary_seal_gas_flow, error_separation_seal_gas_flow,
                    error_separation_seal_gas_pressure, error_seal_gas_to_vent_diff_pressure,
                    error_encoding, timestamp, compressor_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.fast_executemany = True
            cursor.executemany(insert_sql, predictions_data)
            
            logger.info(f"✓ Inserted {len(predictions_data)} predictions")
            
            # Write top anomalies
            self._write_top_anomalies(cursor, results, batch_id)
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"✗ Error writing predictions: {str(e)}")
            logger.error(f"Results columns: {results.columns.tolist()}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def _write_top_anomalies(self, cursor, results: pd.DataFrame, batch_id: str):
        """Write top anomalies with feature contributions"""
        try:
            # Get top 10 anomalies
            anomalies = results[results['triggered_anomaly'] == True].copy()
            if len(anomalies) == 0:
                logger.info("✓ No anomalies detected in this batch")
                return
            
            top_anomalies = anomalies.nlargest(10, 'anomaly_score')
            
            feature_error_cols = [
                'error_filter_dp', 'error_seal_gas_flow', 'error_seal_gas_diff_pressure',
                'error_seal_gas_temp', 'error_primary_vent_flow', 'error_primary_vent_pressure',
                'error_secondary_seal_gas_flow', 'error_separation_seal_gas_flow',
                'error_separation_seal_gas_pressure', 'error_seal_gas_to_vent_diff_pressure',
                'error_encoding'
            ]
            
            top_anomalies_data = []
            for rank, (idx, row) in enumerate(top_anomalies.iterrows(), 1):
                # Get top 3 contributing features
                feature_errors = {}
                for col in feature_error_cols:
                    if col in row and pd.notna(row[col]):
                        feature_name = col.replace('error_', '')
                        feature_errors[feature_name] = float(row[col])
                
                # Sort by error value
                sorted_features = sorted(feature_errors.items(), key=lambda x: x[1], reverse=True)
                
                top_feature1 = sorted_features[0][0] if len(sorted_features) > 0 else None
                top_feature1_error = sorted_features[0][1] if len(sorted_features) > 0 else None
                top_feature2 = sorted_features[1][0] if len(sorted_features) > 1 else None
                top_feature2_error = sorted_features[1][1] if len(sorted_features) > 1 else None
                top_feature3 = sorted_features[2][0] if len(sorted_features) > 2 else None
                top_feature3_error = sorted_features[2][1] if len(sorted_features) > 2 else None
                
                # Handle timestamp column (might be 'timestamp' or 'datetime')
                timestamp_value = datetime.now()
                if 'timestamp' in row:
                    timestamp_value = pd.to_datetime(row['timestamp'])
                elif 'datetime' in row:
                    timestamp_value = pd.to_datetime(row['datetime'])
                elif 'end_timestamp' in row:
                    timestamp_value = pd.to_datetime(row['end_timestamp'])
                
                # Handle compressor_id
                compressor_id_value = 'Unknown'
                if 'compressor_id' in row:
                    compressor_id_value = str(row['compressor_id'])
                elif 'CompressorID' in row:
                    compressor_id_value = str(row['CompressorID'])
                
                anomaly_record = (
                    batch_id,
                    compressor_id_value,
                    timestamp_value,
                    int(row['sequence_index']),
                    float(row['anomaly_score']),
                    float(row['reconstruction_error']),
                    rank,
                    top_feature1,
                    top_feature1_error,
                    top_feature2,
                    top_feature2_error,
                    top_feature3,
                    top_feature3_error,
                    datetime.now()
                )
                top_anomalies_data.append(anomaly_record)
            
            if top_anomalies_data:
                anomaly_sql = """
                    INSERT INTO dbo.LSTM_Top_Anomalies (
                        BatchID, CompressorID, Timestamp, SequenceIndex,
                        AnomalyScore, ReconstructionError, Rank,
                        TopFeature1, TopFeature1Error,
                        TopFeature2, TopFeature2Error,
                        TopFeature3, TopFeature3Error,
                        ProcessedAt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.executemany(anomaly_sql, top_anomalies_data)
                logger.info(f"✓ Inserted {len(top_anomalies_data)} top anomalies")
                
        except Exception as e:
            logger.error(f"✗ Error writing top anomalies: {str(e)}")
    
    def write_batch_summary(self, results: pd.DataFrame, batch_id: str, 
                           model_info: Dict[str, Any]) -> bool:
        """
        Write batch processing summary statistics
        
        Args:
            results: DataFrame with prediction results
            batch_id: Unique identifier for this batch
            model_info: Dictionary with model metadata
            
        Returns:
            bool: Success status
        """
        try:
            self._ensure_connection()
            cursor = self.connection.cursor()
            
            # Calculate summary statistics
            total_sequences = len(results)
            anomalies_detected = int(results['triggered_anomaly'].sum())
            anomaly_rate = (anomalies_detected / total_sequences * 100) if total_sequences > 0 else 0
            avg_reconstruction_error = float(results['reconstruction_error'].mean())
            max_reconstruction_error = float(results['reconstruction_error'].max())
            
            # Handle case where no anomalies detected
            if anomalies_detected > 0:
                avg_anomaly_score = float(results[results['triggered_anomaly'] == True]['anomaly_score'].mean())
            else:
                avg_anomaly_score = 0.0
            
            # Get time range - handle different column names
            timestamp_col = None
            if 'timestamp' in results.columns:
                timestamp_col = 'timestamp'
            elif 'datetime' in results.columns:
                timestamp_col = 'datetime'
            elif 'end_timestamp' in results.columns:
                timestamp_col = 'end_timestamp'
            
            if timestamp_col:
                min_timestamp = results[timestamp_col].min()
                max_timestamp = results[timestamp_col].max()
            else:
                min_timestamp = datetime.now()
                max_timestamp = datetime.now()
            
            # Get unique compressors
            compressor_col = None
            if 'compressor_id' in results.columns:
                compressor_col = 'compressor_id'
            elif 'CompressorID' in results.columns:
                compressor_col = 'CompressorID'
            
            if compressor_col:
                unique_compressors = int(results[compressor_col].nunique())
            else:
                unique_compressors = 1
            
            summary_data = (
                batch_id,
                datetime.now(),
                total_sequences,
                anomalies_detected,
                float(anomaly_rate),
                avg_reconstruction_error,
                max_reconstruction_error,
                avg_anomaly_score,
                pd.to_datetime(min_timestamp),
                pd.to_datetime(max_timestamp),
                unique_compressors,
                model_info.get('model_version', 'Unknown'),
                float(model_info.get('threshold', 0.0)),
                int(model_info.get('sequence_length', 0))
            )
            
            summary_sql = """
                INSERT INTO dbo.LSTM_Batch_Summary (
                    BatchID, ProcessedAt, TotalSequences, AnomaliesDetected,
                    AnomalyRate, AvgReconstructionError, MaxReconstructionError,
                    AvgAnomalyScore, DataStartTime, DataEndTime,
                    UniqueCompressors, ModelVersion, Threshold, SequenceLength
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(summary_sql, summary_data)
            logger.info(f"✓ Batch summary written: {anomalies_detected}/{total_sequences} anomalies ({anomaly_rate:.2f}%)")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"✗ Error writing batch summary: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def write_feature_statistics(self, results: pd.DataFrame, batch_id: str) -> bool:
        """
        Write per-feature error statistics for analysis
        
        Args:
            results: DataFrame with prediction results
            batch_id: Unique identifier for this batch
            
        Returns:
            bool: Success status
        """
        try:
            self._ensure_connection()
            cursor = self.connection.cursor()
            
            feature_error_cols = [
                'error_filter_dp', 'error_seal_gas_flow', 'error_seal_gas_diff_pressure',
                'error_seal_gas_temp', 'error_primary_vent_flow', 'error_primary_vent_pressure',
                'error_secondary_seal_gas_flow', 'error_separation_seal_gas_flow',
                'error_separation_seal_gas_pressure', 'error_seal_gas_to_vent_diff_pressure',
                'error_encoding'
            ]
            
            feature_stats = []
            for col in feature_error_cols:
                if col in results.columns:
                    feature_name = col.replace('error_', '')
                    valid_errors = results[col].dropna()
                    
                    if len(valid_errors) > 0:
                        stat_record = (
                            batch_id,
                            feature_name,
                            float(valid_errors.mean()),
                            float(valid_errors.std()),
                            float(valid_errors.min()),
                            float(valid_errors.max()),
                            float(valid_errors.median()),
                            float(valid_errors.quantile(0.95)),
                            int(len(valid_errors)),
                            datetime.now()
                        )
                        feature_stats.append(stat_record)
            
            if feature_stats:
                stats_sql = """
                    INSERT INTO dbo.LSTM_Feature_Statistics (
                        BatchID, FeatureName, MeanError, StdError,
                        MinError, MaxError, MedianError, P95Error,
                        SampleCount, ProcessedAt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.executemany(stats_sql, feature_stats)
                logger.info(f"✓ Feature statistics written for {len(feature_stats)} features")
                
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"✗ Error writing feature statistics: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def get_recent_batches(self, limit: int = 10) -> pd.DataFrame:
        """
        Retrieve recent batch summaries
        
        Args:
            limit: Number of recent batches to retrieve
            
        Returns:
            DataFrame with batch information
        """
        try:
            self._ensure_connection()
            
            query = """
                SELECT TOP (?) 
                    BatchID, ProcessedAt, TotalSequences, AnomaliesDetected,
                    AnomalyRate, AvgReconstructionError, MaxReconstructionError,
                    ModelVersion, Threshold
                FROM dbo.LSTM_Batch_Summary
                ORDER BY ProcessedAt DESC
            """
            
            df = pd.read_sql(query, self.connection, params=(limit,))
            logger.info(f"✓ Retrieved {len(df)} recent batches")
            return df
            
        except Exception as e:
            logger.error(f"✗ Error retrieving recent batches: {str(e)}")
            return pd.DataFrame()
    
    def get_anomalies_by_compressor(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get anomaly counts grouped by compressor for a date range
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            DataFrame with anomaly counts per compressor
        """
        try:
            self._ensure_connection()
            
            query = """
                SELECT 
                    compressor_id,
                    COUNT(*) as total_anomalies,
                    AVG(anomaly_score) as avg_anomaly_score,
                    MAX(anomaly_score) as max_anomaly_score,
                    MIN(timestamp) as first_anomaly,
                    MAX(timestamp) as last_anomaly
                FROM dbo.LSTM_Predictions
                WHERE triggered_anomaly = 1
                    AND timestamp BETWEEN ? AND ?
                GROUP BY compressor_id
                ORDER BY total_anomalies DESC
            """
            
            df = pd.read_sql(query, self.connection, params=(start_date, end_date))
            
            if len(df) == 0:
                logger.info(f"✓ No anomalies found between {start_date} and {end_date}")
            else:
                logger.info(f"✓ Retrieved anomaly data for {len(df)} compressors")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error retrieving compressor anomalies: {str(e)}")
            return pd.DataFrame()
    
    def cleanup_old_predictions(self, days_to_keep: int = 90) -> int:
        """
        Delete prediction records older than specified days
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of records deleted
        """
        try:
            self._ensure_connection()
            cursor = self.connection.cursor()
            
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            # Delete old predictions
            delete_sql = """
                DELETE FROM dbo.LSTM_Predictions
                WHERE ProcessedAt < ?
            """
            
            cursor.execute(delete_sql, cutoff_date)
            deleted_count = cursor.rowcount
            
            # Also cleanup related tables
            cursor.execute("DELETE FROM dbo.LSTM_Top_Anomalies WHERE ProcessedAt < ?", cutoff_date)
            cursor.execute("DELETE FROM dbo.LSTM_Batch_Summary WHERE ProcessedAt < ?", cutoff_date)
            cursor.execute("DELETE FROM dbo.LSTM_Feature_Statistics WHERE ProcessedAt < ?", cutoff_date)
            
            self.connection.commit()
            logger.info(f"✓ Cleaned up {deleted_count} old prediction records (older than {days_to_keep} days)")
            return deleted_count
            
        except Exception as e:
            logger.error(f"✗ Error cleaning up old predictions: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return 0
    
    def close(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                logger.info("✓ Database connection closed")
            except Exception as e:
                logger.error(f"✗ Error closing connection: {str(e)}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Connection string example
    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=yokogawa-sqlserver-x2g77kzajj6gy.database.windows.net;"
        "DATABASE=AOM-Dev;"
        "UID=xmadmin;"
        "PWD=NnDmyJEfQkw9;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )
    
    # Initialize writer
    db_writer = DatabaseWriter(conn_str)
    
    # Example: Write predictions
    # results_df = pd.DataFrame(...)  # Your predictions dataframe
    # batch_id = str(uuid.uuid4())
    # 
    # success = db_writer.write_predictions(results_df, batch_id)
    # 
    # if success:
    #     db_writer.write_batch_summary(results_df, batch_id, {
    #         'model_version': 'v1.0',
    #         'threshold': 0.05,
    #         'sequence_length': 10
    #     })
    #     db_writer.write_feature_statistics(results_df, batch_id)
    
    # Close connection when done
    db_writer.close()