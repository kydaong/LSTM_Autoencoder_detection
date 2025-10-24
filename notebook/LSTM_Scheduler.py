# -*- coding: utf-8 -*-
# LSTM_Scheduler.py
"""
Automated scheduler for LSTM anomaly detection pipeline
Runs at specified intervals to check for new data and process it
"""
import schedule
import time
import pyodbc
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path

# Import your pipeline
from LSTM_Prediction_pipeline_compaware import CompressorAwareLSTMPredictionPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



class LSTMScheduler:
    """Scheduler for automated LSTM pipeline execution"""
    
    def __init__(self, connection_string: str, model_path: str, scaler_path: str,
                 encoder_path: str, thresholds_path: str, source_table: str,
                 min_new_records: int = 100, lookback_hours: int = 2):
        """
        Initialize the scheduler
        
        Args:
            connection_string: Database connection string
            model_path: Path to LSTM model
            scaler_path: Path to scaler
            encoder_path: Path to encoder
            thresholds_path: Path to thresholds
            source_table: Source data table name
            min_new_records: Minimum new records required to trigger processing
            lookback_hours: How many hours back to check for new data
        """
        self.connection_string = connection_string
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.thresholds_path = thresholds_path
        self.source_table = source_table
        self.min_new_records = min_new_records
        self.lookback_hours = lookback_hours
        
        # Track last processed time
        self.last_processed_time = None
        self.load_last_processed_time()
        
        logger.info("="*80)
        logger.info("LSTM SCHEDULER INITIALIZED")
        logger.info("="*80)
        logger.info(f"Source table: {source_table}")
        logger.info(f"Min records to trigger: {min_new_records}")
        logger.info(f"Lookback window: {lookback_hours} hours")
        logger.info(f"Last processed: {self.last_processed_time}")
        logger.info("="*80)
    
    def load_last_processed_time(self):
        """Load the last processed timestamp from file"""
        checkpoint_file = Path("lstm_scheduler_checkpoint.txt")
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    self.last_processed_time = datetime.fromisoformat(timestamp_str)
                logger.info(f"✓ Loaded checkpoint: {self.last_processed_time}")
            except Exception as e:
                logger.warning(f"⚠ Could not load checkpoint: {e}")
                self.last_processed_time = datetime.now() - timedelta(hours=self.lookback_hours)
        else:
            # First run - start from lookback_hours ago
            self.last_processed_time = datetime.now() - timedelta(hours=self.lookback_hours)
            logger.info(f"✓ First run - starting from {self.last_processed_time}")
    
    def save_last_processed_time(self, timestamp: datetime):
        """Save the last processed timestamp to file"""
        checkpoint_file = Path("lstm_scheduler_checkpoint.txt")
        try:
            with open(checkpoint_file, 'w') as f:
                f.write(timestamp.isoformat())
            self.last_processed_time = timestamp
            logger.info(f"✓ Saved checkpoint: {timestamp}")
        except Exception as e:
            logger.error(f"✗ Could not save checkpoint: {e}")
    
    def check_new_data(self):
        """Check if there's new data to process"""
        try:
            connection = pyodbc.connect(self.connection_string, timeout=10)
            
            # Query for new records since last processed time
            query = f"""
                SELECT COUNT(*) as new_records,
                       MIN(datetime) as min_time,
                       MAX(datetime) as max_time
                FROM dbo.{self.source_table}
                WHERE datetime > ?
            """
            
            df = pd.read_sql(query, connection, params=[self.last_processed_time])
            connection.close()
            
            new_records = df['new_records'].iloc[0]
            min_time = df['min_time'].iloc[0]
            max_time = df['max_time'].iloc[0]
            
            logger.info(f"📊 Data check: {new_records} new records found")
            if new_records > 0:
                logger.info(f"   Time range: {min_time} to {max_time}")
            
            return new_records, min_time, max_time
            
        except Exception as e:
            logger.error(f"✗ Error checking for new data: {e}")
            return 0, None, None
    
    def run_pipeline(self):
        """Run the LSTM pipeline if conditions are met"""
        logger.info("\n" + "="*80)
        logger.info(f"🕐 SCHEDULED CHECK: {datetime.now()}")
        logger.info("="*80)
        
        try:
            # Check for new data
            new_records, min_time, max_time = self.check_new_data()
            
            if new_records < self.min_new_records:
                logger.info(f"⏭ Skipping - only {new_records} new records (need {self.min_new_records})")
                return
            
            logger.info(f"✓ Processing {new_records} new records...")
            
            # Initialize pipeline
            logger.info("Initializing pipeline...")
            pipeline = CompressorAwareLSTMPredictionPipeline(
                connection_string=self.connection_string,
                model_path=self.model_path,
                scaler_path=self.scaler_path,
                encoder_path=self.encoder_path,
                thresholds_path=self.thresholds_path
            )
            
            # Run pipeline on new data
            start_date = self.last_processed_time
            end_date = datetime.now()
            
            logger.info(f"📅 Processing date range: {start_date} to {end_date}")
            
            batch_id = pipeline.run_pipeline(
                start_date=start_date,
                end_date=end_date,
                consecutive_threshold=3,
                source_table=self.source_table
            )
            
            # Close pipeline
            pipeline.close()
            
            # Update checkpoint to max_time from data (not current time)
            if max_time:
                self.save_last_processed_time(pd.to_datetime(max_time))
            else:
                self.save_last_processed_time(end_date)
            
            logger.info(f"✓ Pipeline completed - Batch ID: {batch_id}")
            
        except Exception as e:
            logger.error(f"✗ Pipeline execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def run_scheduled(self):
        """Run the scheduler (blocking call)"""
        logger.info("\n" + "#"*80)
        logger.info("🚀 LSTM SCHEDULER STARTED")
        logger.info("#"*80)
        logger.info("Press Ctrl+C to stop")
        logger.info("#"*80 + "\n")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\n" + "#"*80)
            logger.info("⏹ SCHEDULER STOPPED BY USER")
            logger.info("#"*80)


# Configuration
def setup_hourly_schedule():
    """Setup to run every hour"""
    CONNECTION_STRING = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=yokogawa-sqlserver-x2g77kzajj6gy.database.windows.net;"
        "DATABASE=AOM-Dev;"
        "UID=xmadmin;"
        "PWD=NnDmyJEfQkw9;"
    )
    
    MODEL_DIR = "models"
    MODEL_PATH = f"{MODEL_DIR}/compressor_aware_model_no_lambda2.h5"
    SCALER_PATH = f"{MODEL_DIR}/feature_scaler_no_lambda2.pkl"
    ENCODER_PATH = f"{MODEL_DIR}/compressor_encoder_no_lambda2.pkl"
    THRESHOLDS_PATH = f"{MODEL_DIR}/thresholds_no_lambda2.pkl"
    SOURCE_TABLE = "compressor_normal_dataset3"
    
    # Initialize scheduler
    scheduler = LSTMScheduler(
        connection_string=CONNECTION_STRING,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        encoder_path=ENCODER_PATH,
        thresholds_path=THRESHOLDS_PATH,
        source_table=SOURCE_TABLE,
        min_new_records=100,  # Only run if at least 100 new records
        lookback_hours=2       # Check last 2 hours of data
    )
    
    # Schedule to run every hour
    schedule.every().hour.at(":00").do(scheduler.run_pipeline)
    
    # Also run immediately on startup
    logger.info("🚀 Running initial check...")
    scheduler.run_pipeline()
    
    return scheduler


def setup_custom_schedule(interval_minutes: int = 60):
    """Setup custom interval (in minutes)"""
    CONNECTION_STRING = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=yokogawa-sqlserver-x2g77kzajj6gy.database.windows.net;"
        "DATABASE=AOM-Dev;"
        "UID=xmadmin;"
        "PWD=NnDmyJEfQkw9;"
    )
    
    MODEL_DIR = "C:/Users/adminuser/Projects/LSTM_Autoencoder_detection/models"    # C:/Users/adminuser/Projects/LSTM_Autoencoder_detection/models
    MODEL_PATH = f"{MODEL_DIR}/compressor_aware_model_no_lambda2.h5"
    SCALER_PATH = f"{MODEL_DIR}/feature_scaler_no_lambda2.pkl"
    ENCODER_PATH = f"{MODEL_DIR}/compressor_encoder_no_lambda2.pkl"
    THRESHOLDS_PATH = f"{MODEL_DIR}/thresholds_no_lambda2.pkl"
    SOURCE_TABLE = "compressor_normal_dataset3"
    
    scheduler = LSTMScheduler(
        connection_string=CONNECTION_STRING,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        encoder_path=ENCODER_PATH,
        thresholds_path=THRESHOLDS_PATH,
        source_table=SOURCE_TABLE,
        min_new_records=100,
        lookback_hours=2
    )
    
    # Schedule to run every X minutes
    schedule.every(interval_minutes).minutes.do(scheduler.run_pipeline)
    
    # Run immediately on startup
    logger.info("🚀 Running initial check...")
    scheduler.run_pipeline()
    
    return scheduler


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM Anomaly Detection Scheduler')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Run interval in minutes (default: 60)')
    parser.add_argument('--min-records', type=int, default=100,
                       help='Minimum new records to trigger processing (default: 100)')
    
    args = parser.parse_args()
    
    # Setup and run scheduler
    if args.interval == 60:
        # Use hourly schedule (cleaner logs)
        scheduler = setup_hourly_schedule()
    else:
        # Use custom interval
        scheduler = setup_custom_schedule(interval_minutes=args.interval)
    
    # Start scheduler (blocking)
    scheduler.run_scheduled()