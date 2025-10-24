# -*- coding: utf-8 -*-
# LSTM_Database_Writer_CompressorAware.py
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CompressorAwareDatabaseWriter:
    """Handles database operations for compressor-aware LSTM predictions"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
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
        Write compressor-aware prediction results to database
        
        Args:
            results: DataFrame with predictions including compressor_id_encoded
            batch_id: Unique identifier for this batch
            
        Returns:
            bool: Success status
        """
        try:
            self._ensure_connection()
            cursor = self.connection.cursor()
            
            # Prepare data for bulk insert
            predictions_data = []
            for row in results.itertuples(index=False):
                # Handle timestamp columns
                timestamp_value = None
                if hasattr(row, 'timestamp'):
                    timestamp_value = pd.to_datetime(row.timestamp)
                elif hasattr(row, 'datetime'):
                    timestamp_value = pd.to_datetime(row.datetime)
                elif hasattr(row, 'end_datetime'):
                    timestamp_value = pd.to_datetime(row.end_datetime)
                else:
                    timestamp_value = datetime.now()
                
                # Get compressor ID (text)
                compressor_id_value = 'Unknown'
                if hasattr(row, 'compressor_id'):
                    compressor_id_value = str(row.compressor_id)
                elif hasattr(row, 'CompressorID'):
                    compressor_id_value = str(row.CompressorID)
                
                # Get compressor ID encoded (integer) - NEW for compressor-aware
                compressor_id_encoded = None
                if hasattr(row, 'compressor_id_encoded'):
                    compressor_id_encoded = int(row.compressor_id_encoded)
                
                prediction_record = (
                    batch_id,
                    datetime.now(),
                    int(row.sequence_index),
                    float(row.reconstruction_error),
                    float(row.threshold),  # This is now per-compressor threshold
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
                    compressor_id_value,
                    compressor_id_encoded  # NEW column
                )
                predictions_data.append(prediction_record)
            
            # Bulk insert - NOTE: Added compressor_id_encoded column
            insert_sql = """
                INSERT INTO dbo.LSTM_Predictions (
                    BatchID, ProcessedAt,
                    sequence_index, reconstruction_error, threshold,
                    is_anomaly_point, triggered_anomaly, anomaly_score, consecutive_count,
                    error_filter_dp, error_seal_gas_flow, error_seal_gas_diff_pressure,
                    error_seal_gas_temp, error_primary_vent_flow, error_primary_vent_pressure,
                    error_secondary_seal_gas_flow, error_separation_seal_gas_flow,
                    error_separation_seal_gas_pressure, error_seal_gas_to_vent_diff_pressure,
                    error_encoding, timestamp, compressor_id, compressor_id_encoded
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            # Get top anomalies PER COMPRESSOR
            anomalies = results[results['triggered_anomaly'] == True].copy()
            if len(anomalies) == 0:
                logger.info("✓ No anomalies detected in this batch")
                return
            
            # Get top 5 anomalies per compressor
            top_anomalies_list = []
            for compressor in anomalies['compressor_id'].unique():
                comp_anomalies = anomalies[anomalies['compressor_id'] == compressor]
                top_comp = comp_anomalies.nlargest(5, 'anomaly_score')
                top_anomalies_list.append(top_comp)
            
            if not top_anomalies_list:
                return
                
            top_anomalies = pd.concat(top_anomalies_list).sort_values('anomaly_score', ascending=False)
            
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
                
                sorted_features = sorted(feature_errors.items(), key=lambda x: x[1], reverse=True)
                
                top_feature1 = sorted_features[0][0] if len(sorted_features) > 0 else None
                top_feature1_error = sorted_features[0][1] if len(sorted_features) > 0 else None
                top_feature2 = sorted_features[1][0] if len(sorted_features) > 1 else None
                top_feature2_error = sorted_features[1][1] if len(sorted_features) > 1 else None
                top_feature3 = sorted_features[2][0] if len(sorted_features) > 2 else None
                top_feature3_error = sorted_features[2][1] if len(sorted_features) > 2 else None
                
                # Handle timestamp
                timestamp_value = datetime.now()
                if 'timestamp' in row:
                    timestamp_value = pd.to_datetime(row['timestamp'])
                elif 'datetime' in row:
                    timestamp_value = pd.to_datetime(row['datetime'])
                elif 'end_datetime' in row:
                    timestamp_value = pd.to_datetime(row['end_datetime'])
                
                # Handle compressor_id
                compressor_id_value = str(row.get('compressor_id', 'Unknown'))
                
                # Get compressor-specific threshold used
                threshold_used = float(row.get('threshold', 0.0))
                
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
                    datetime.now(),
                    threshold_used  # NEW: Store the threshold used for this compressor
                )
                top_anomalies_data.append(anomaly_record)
            
            if top_anomalies_data:
                # NOTE: Added ThresholdUsed column
                anomaly_sql = """
                    INSERT INTO dbo.LSTM_Top_Anomalies (
                        BatchID, CompressorID, Timestamp, SequenceIndex,
                        AnomalyScore, ReconstructionError, Rank,
                        TopFeature1, TopFeature1Error,
                        TopFeature2, TopFeature2Error,
                        TopFeature3, TopFeature3Error,
                        ProcessedAt, ThresholdUsed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.executemany(anomaly_sql, top_anomalies_data)
                logger.info(f"✓ Inserted {len(top_anomalies_data)} top anomalies")
                
        except Exception as e:
            logger.error(f"✗ Error writing top anomalies: {str(e)}")
    
    def write_batch_summary(self, results: pd.DataFrame, batch_id: str, 
                           model_info: Dict[str, Any]) -> bool:
        """
        Write batch summary with compressor-aware details
        
        Args:
            results: DataFrame with prediction results
            batch_id: Unique identifier for this batch
            model_info: Dictionary with model metadata including per-compressor thresholds
            
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
            
            if anomalies_detected > 0:
                avg_anomaly_score = float(results[results['triggered_anomaly'] == True]['anomaly_score'].mean())
            else:
                avg_anomaly_score = 0.0
            
            # Get time range
            timestamp_col = None
            if 'timestamp' in results.columns:
                timestamp_col = 'timestamp'
            elif 'datetime' in results.columns:
                timestamp_col = 'datetime'
            elif 'end_datetime' in results.columns:
                timestamp_col = 'end_datetime'
            
            if timestamp_col:
                min_timestamp = results[timestamp_col].min()
                max_timestamp = results[timestamp_col].max()
            else:
                min_timestamp = datetime.now()
                max_timestamp = datetime.now()
            
            # Get unique compressors
            compressor_col = 'compressor_id' if 'compressor_id' in results.columns else 'CompressorID'
            unique_compressors = int(results[compressor_col].nunique())
            
            # Get model version and type
            model_version = model_info.get('model_version', 'Unknown')
            model_type = model_info.get('model_type', 'compressor_aware_lstm')
            
            # Calculate average threshold (across all compressors)
            avg_threshold = float(results['threshold'].mean())
            
            # Serialize per-compressor thresholds to JSON
            per_compressor_thresholds = model_info.get('per_compressor_thresholds', {})
            thresholds_json = json.dumps(per_compressor_thresholds) if per_compressor_thresholds else None
            
            # Get known compressors list
            known_compressors = model_info.get('known_compressors', [])
            known_compressors_json = json.dumps(list(known_compressors)) if known_compressors else None
            
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
                model_version,
                avg_threshold,  # Store average threshold
                int(model_info.get('sequence_length', 0)),
                model_type,  # NEW: Model type
                thresholds_json,  # NEW: Per-compressor thresholds as JSON
                known_compressors_json  # NEW: Known compressors list
            )
            
            # NOTE: Added ModelType, PerCompressorThresholds, and KnownCompressors columns
            summary_sql = """
                INSERT INTO dbo.LSTM_Batch_Summary (
                    BatchID, ProcessedAt, TotalSequences, AnomaliesDetected,
                    AnomalyRate, AvgReconstructionError, MaxReconstructionError,
                    AvgAnomalyScore, DataStartTime, DataEndTime,
                    UniqueCompressors, ModelVersion, Threshold, SequenceLength,
                    ModelType, PerCompressorThresholds, KnownCompressors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(summary_sql, summary_data)
            logger.info(f"✓ Batch summary written: {anomalies_detected}/{total_sequences} anomalies ({anomaly_rate:.2f}%)")
            logger.info(f"  Model type: {model_type}")
            logger.info(f"  Processed {unique_compressors} compressors")
            
            # Write per-compressor summary stats
            self._write_per_compressor_summary(cursor, results, batch_id)
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"✗ Error writing batch summary: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def _write_per_compressor_summary(self, cursor, results: pd.DataFrame, batch_id: str):
        """Write summary statistics per compressor - NEW METHOD"""
        try:
            compressor_stats = []
            
            for compressor_id in results['compressor_id'].unique():
                comp_data = results[results['compressor_id'] == compressor_id]
                
                total_sequences = len(comp_data)
                anomalies = int(comp_data['triggered_anomaly'].sum())
                anomaly_rate = (anomalies / total_sequences * 100) if total_sequences > 0 else 0
                avg_error = float(comp_data['reconstruction_error'].mean())
                max_error = float(comp_data['reconstruction_error'].max())
                threshold = float(comp_data['threshold'].iloc[0])  # Per-compressor threshold
                
                stat_record = (
                    batch_id,
                    str(compressor_id),
                    total_sequences,
                    anomalies,
                    float(anomaly_rate),
                    avg_error,
                    max_error,
                    threshold,
                    datetime.now()
                )
                compressor_stats.append(stat_record)
            
            if compressor_stats:
                stats_sql = """
                    INSERT INTO dbo.LSTM_Compressor_Summary (
                        BatchID, CompressorID, TotalSequences, AnomaliesDetected,
                        AnomalyRate, AvgReconstructionError, MaxReconstructionError,
                        Threshold, ProcessedAt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.executemany(stats_sql, compressor_stats)
                logger.info(f"✓ Per-compressor summary written for {len(compressor_stats)} compressors")
                
        except Exception as e:
            logger.error(f"✗ Error writing per-compressor summary: {str(e)}")
    
    def write_feature_statistics(self, results: pd.DataFrame, batch_id: str) -> bool:
        """Write per-feature error statistics"""
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
            
            # Overall feature statistics
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
            
            # Per-compressor feature statistics - NEW
            self._write_per_compressor_feature_stats(cursor, results, batch_id, feature_error_cols)
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"✗ Error writing feature statistics: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def _write_per_compressor_feature_stats(self, cursor, results: pd.DataFrame, 
                                           batch_id: str, feature_error_cols: list):
        """Write feature statistics per compressor - NEW METHOD"""
        try:
            comp_feature_stats = []
            
            for compressor_id in results['compressor_id'].unique():
                comp_data = results[results['compressor_id'] == compressor_id]
                
                for col in feature_error_cols:
                    if col in comp_data.columns:
                        feature_name = col.replace('error_', '')
                        valid_errors = comp_data[col].dropna()
                        
                        if len(valid_errors) > 0:
                            stat_record = (
                                batch_id,
                                str(compressor_id),
                                feature_name,
                                float(valid_errors.mean()),
                                float(valid_errors.std()),
                                float(valid_errors.max()),
                                float(valid_errors.quantile(0.95)),
                                int(len(valid_errors)),
                                datetime.now()
                            )
                            comp_feature_stats.append(stat_record)
            
            if comp_feature_stats:
                stats_sql = """
                    INSERT INTO dbo.LSTM_Compressor_Feature_Stats (
                        BatchID, CompressorID, FeatureName, MeanError,
                        StdError, MaxError, P95Error, SampleCount, ProcessedAt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.executemany(stats_sql, comp_feature_stats)
                logger.info(f"✓ Per-compressor feature stats written: {len(comp_feature_stats)} records")
                
        except Exception as e:
            logger.error(f"✗ Error writing per-compressor feature stats: {str(e)}")
    
    def get_recent_batches(self, limit: int = 10) -> pd.DataFrame:
        """Retrieve recent batch summaries"""
        try:
            self._ensure_connection()
            
            query = """
                SELECT TOP (?) 
                    BatchID, ProcessedAt, TotalSequences, AnomaliesDetected,
                    AnomalyRate, AvgReconstructionError, MaxReconstructionError,
                    ModelVersion, ModelType, Threshold, UniqueCompressors
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
        """Get anomaly counts grouped by compressor"""
        try:
            self._ensure_connection()
            
            query = """
                SELECT 
                    compressor_id,
                    COUNT(*) as total_anomalies,
                    AVG(anomaly_score) as avg_anomaly_score,
                    MAX(anomaly_score) as max_anomaly_score,
                    AVG(threshold) as avg_threshold,
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
        """Delete prediction records older than specified days"""
        try:
            self._ensure_connection()
            cursor = self.connection.cursor()
            
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            # Delete from all related tables
            cursor.execute("DELETE FROM dbo.LSTM_Predictions WHERE ProcessedAt < ?", cutoff_date)
            deleted_count = cursor.rowcount
            
            cursor.execute("DELETE FROM dbo.LSTM_Top_Anomalies WHERE ProcessedAt < ?", cutoff_date)
            cursor.execute("DELETE FROM dbo.LSTM_Batch_Summary WHERE ProcessedAt < ?", cutoff_date)
            cursor.execute("DELETE FROM dbo.LSTM_Feature_Statistics WHERE ProcessedAt < ?", cutoff_date)
            cursor.execute("DELETE FROM dbo.LSTM_Compressor_Summary WHERE ProcessedAt < ?", cutoff_date)
            cursor.execute("DELETE FROM dbo.LSTM_Compressor_Feature_Stats WHERE ProcessedAt < ?", cutoff_date)
            
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