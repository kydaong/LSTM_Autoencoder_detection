import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import pyodbc
import sys
import os


# Add project root to path (same as before)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tools.sql_connect import SQLConnector

def load_model_components(model_path='compressor_autoencoder2.h5',
                          scaler_path='scaler2.pkl',
                          config_path='model_config2.pkl'):
    """Load the saved model, scaler, and configuration"""
    print("="*70)
    print("LOADING MODEL COMPONENTS")
    print("="*70)

    try:
        # Get the directory where this script is located (notebook folder)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level to project root, then into models folder
        project_root = os.path.dirname(script_dir)
        models_dir = os.path.join(project_root, 'models')
        
        # Build full paths - if relative paths are given, look in models folder
        full_model_path = os.path.join(models_dir, model_path) if not os.path.isabs(model_path) else model_path
        full_scaler_path = os.path.join(models_dir, scaler_path) if not os.path.isabs(scaler_path) else scaler_path
        full_config_path = os.path.join(models_dir, config_path) if not os.path.isabs(config_path) else config_path
        
        print(f"\n📂 Looking for files:")
        print(f"   Model:  {full_model_path}")
        print(f"   Scaler: {full_scaler_path}")
        print(f"   Config: {full_config_path}")
        
        # Check if files exist
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model file not found: {full_model_path}")
        if not os.path.exists(full_scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {full_scaler_path}")
        if not os.path.exists(full_config_path):
            raise FileNotFoundError(f"Config file not found: {full_config_path}")
        
        # Load model
        model = load_model(full_model_path, compile=False)
        print("   ✓ Model loaded successfully")
        
        # Recompile model
        from tensorflow.keras.optimizers import Adam
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        print("   ✓ Model recompiled")
        
        # Load scaler
        scaler = joblib.load(full_scaler_path)
        print("   ✓ Scaler loaded successfully")
        
        # Load config
        config = joblib.load(full_config_path)
        threshold = config['threshold']
        feature_columns = config['features']
        print("   ✓ Config loaded successfully")
    
        print(f"\n📋 Model Configuration:")
        print(f"   Threshold: {threshold:.6f}")
        print(f"   Features ({len(feature_columns)}):")
        for i, col in enumerate(feature_columns, 1):
            print(f"      {i}. {col}")
    
        return model, scaler, threshold, feature_columns
    
    except Exception as e:
        print(f"\n❌ Error loading model components: {str(e)}")
        return None, None, None, None


from sqlalchemy import text

def load_data_from_db(database='AOM-Dev', db_profile='AzureSQL', 
                      table_name='compressor_normal_dataset3', 
                      compressor_id=None, limit=None):
    """Load test data from Azure SQL Server database"""
    print("\n" + "="*70)
    print("LOADING DATA FROM AZURE SQL DATABASE")
    print("="*70)
    
    try:
        print(f"\n🔌 Connecting to Azure SQL:")
        print(f"   Database: {database}")
        print(f"   Profile: {db_profile}")
        
        sql_conn = SQLConnector(database='mssql', db_profile=db_profile)
        
        if sql_conn is None or not hasattr(sql_conn, 'engine') or sql_conn.engine is None:
            print("❌ Failed to create database connection")
            return None
        
        print(f"   ✓ Connected successfully")
        
        # Build query
        where_clause = ""
        if compressor_id:
            where_clause = f" WHERE compressor_id = '{compressor_id}'"
        
        if limit:
            query = f"SELECT TOP {limit} * FROM {table_name}{where_clause} ORDER BY datetime"
        else:
            query = f"SELECT * FROM {table_name}{where_clause} ORDER BY datetime"
        
        query = ' '.join(query.split())
        print(f"\n📊 Executing query:")
        print(f"   {query}")
        
        df = pd.read_sql(query, sql_conn.engine)
        print(f"\n✓ Loaded {len(df)} rows from Azure SQL database")
        
        if len(df) == 0:
            print("   ⚠️ Warning: Query returned 0 rows")
            return df
            
        print(f"   Columns: {list(df.columns)}")
        
        if 'compressor_id' in df.columns:
            unique_compressors = df['compressor_id'].unique()
            print(f"   Compressors: {list(unique_compressors)}")
            for comp_id in unique_compressors:
                count = (df['compressor_id'] == comp_id).sum()
                print(f"      - {comp_id}: {count} samples")
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error loading data from database: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def write_results_to_db(results, database='AOM-Dev', db_profile='AzureSQL',
                        table_name='anomaly_detection_results',
                        if_exists='replace'):
    """Write anomaly detection results to Azure SQL database"""
    print("\n" + "="*70)
    print("WRITING RESULTS TO AZURE SQL DATABASE")
    print("="*70)
    
    try:
        print(f"\n🔌 Connecting to Azure SQL:")
        print(f"   Database: {database}")
        print(f"   Profile: {db_profile}")
        
        sql_conn = SQLConnector(database='mssql', db_profile=db_profile)
        
        if sql_conn is None or not hasattr(sql_conn, 'engine') or sql_conn.engine is None:
            print("❌ Failed to create database connection")
            return False
        
        df = results.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Convert boolean columns to integers
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        if bool_cols:
            print(f"\n🔄 Converting boolean columns to integers: {bool_cols}")
            for col in bool_cols:
                df[col] = df[col].astype(int)
        
        # Handle timestamp columns - create SQL-safe versions
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        if timestamp_cols:
            print(f"\n📅 Processing timestamp columns: {timestamp_cols}")
            for col in timestamp_cols:
                df[col] = pd.to_datetime(df[col])
                # Create SQL-safe column name (avoid 'timestamp' reserved keyword)
                if col == 'timestamp':
                    df['event_timestamp'] = df[col]
                elif col == 'timestamp_start':
                    df['event_timestamp_start'] = df[col]
        
        # Add analysis timestamp
        df['analysis_timestamp'] = datetime.now()
        
        # Replace inf values and fill NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Clean column names
        df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        print(f"\n💾 Preparing to write {len(df)} rows to table: {table_name}")
        print(f"   Mode: {if_exists}")
        print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
        
        # Drop table if exists and mode is 'replace'
        from sqlalchemy import inspect
        inspector = inspect(sql_conn.engine)
        
        if table_name in inspector.get_table_names():
            if if_exists == 'replace':
                print(f"\n⚠️ Table '{table_name}' exists. Dropping it...")
                with sql_conn.engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE {table_name}"))
                    conn.commit()
                print(f"   ✓ Table dropped")
        
        # Write to database
        print(f"\n💾 Writing data to Azure SQL...")
        df.to_sql(
            name=table_name,
            con=sql_conn.engine,
            if_exists='append',
            index=False,
            method=None,
            chunksize=1000
        )
        
        print(f"   ✓ Data written successfully")
        
        # Verify write
        verify_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        with sql_conn.engine.connect() as conn:
            result = conn.execute(text(verify_query))
            count = result.fetchone()[0]
            print(f"   ✓ Verified: {count} rows in table '{table_name}'")
        
        # Show sample with timestamps
        sample_query = f"SELECT TOP 3 sequence_index, event_timestamp, reconstruction_error, anomaly_score, triggered_anomaly FROM {table_name} ORDER BY sequence_index"
        try:
            sample_df = pd.read_sql(sample_query, sql_conn.engine)
            print(f"\n📊 Sample of written data:")
            print(sample_df.to_string(index=False))
        except:
            print(f"\n📊 Data written successfully (sample query failed)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error writing to database: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_test_data(df, scaler, feature_columns):
    """Preprocess test data using saved scaler"""
    print("\n" + "="*70)
    print("PREPROCESSING TEST DATA")
    print("="*70)
    
    # Check for datetime/timestamp column
    time_col = None
    if 'datetime' in df.columns:
        time_col = 'datetime'
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        print(f"   ✓ Using 'datetime' column for time ordering")
    elif 'timestamp' in df.columns:
        time_col = 'timestamp'
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        print(f"   ✓ Using 'timestamp' column for time ordering")
    
    # Check for missing features
    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        print(f"\n⚠️ Warning: Missing features: {missing_features}")
        for feat in missing_features:
            df[feat] = 0
    
    # Select and order features
    df_features = df[feature_columns].copy()
    
    # Handle missing values
    if df_features.isnull().any().any():
        print(f"\n⚠️ Warning: Found missing values, filling...")
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')
    
    # Scale data
    scaled_data = scaler.transform(df_features)
    print(f"   ✓ Data scaled successfully")
    
    return scaled_data, df


def create_sequences(data, sequence_length):
    """Create sequences for LSTM"""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:(i + sequence_length)])
    return np.array(sequences)


def test_model_from_db(database='AOM-Dev',
                       db_profile='AzureSQL',
                       source_table='compressor_normal_dataset3',
                       results_table='anomaly_detection_results',
                       compressor_id=None,
                       limit=None,
                       model_path='compressor_autoencoder2.h5',
                       scaler_path='scaler2.pkl',
                       config_path='model_config2.pkl',
                       sequence_length=10,
                       consecutive_threshold=3):
    """Complete pipeline: Load from Azure SQL -> Test -> Write results to Azure SQL"""
    print("\n" + "="*70)
    print("LSTM AUTOENCODER - AZURE SQL DATABASE TESTING")
    print("="*70)
    
    # Load model components
    model, scaler, threshold, feature_columns = load_model_components()
    
    if model is None:
        return None
    
    # Load data from database
    df = load_data_from_db(database, db_profile, source_table, compressor_id, limit)
    
    if df is None or len(df) == 0:
        print("\n❌ No data loaded from database")
        return None
    
    # Preprocess
    scaled_data, df_original = preprocess_test_data(df, scaler, feature_columns)
    
    # Create sequences
    print(f"\n🔄 Creating sequences (length={sequence_length})...")
    sequences = create_sequences(scaled_data, sequence_length)
    print(f"   ✓ Created {len(sequences)} sequences")
    
    if len(sequences) == 0:
        print(f"\n❌ Not enough data for sequences")
        return None
    
    # Predict
    print(f"\n🤖 Running predictions...")
    predictions = model.predict(sequences, verbose=0)
    
    # Calculate reconstruction errors
    mse = np.mean(np.power(sequences - predictions, 2), axis=(1, 2))
    feature_mse = np.mean(np.power(sequences - predictions, 2), axis=1)
    
    # Detect anomalies
    is_anomaly_point = mse > threshold
    triggered_anomaly = np.zeros(len(mse), dtype=bool)
    consecutive_count = np.zeros(len(mse), dtype=int)
    
    consecutive = 0
    for i in range(len(is_anomaly_point)):
        if is_anomaly_point[i]:
            consecutive += 1
            if consecutive >= consecutive_threshold:
                start_idx = max(0, i - consecutive + 1)
                triggered_anomaly[start_idx:i+1] = True
        else:
            consecutive = 0
        consecutive_count[i] = consecutive
    
    # Map sequences to actual data indices and timestamps
    sequence_end_indices = np.arange(sequence_length - 1, 
                                     sequence_length - 1 + len(sequences))
    
    # Create results DataFrame
    results = pd.DataFrame({
        'sequence_index': range(len(mse)),
        'data_point_index': sequence_end_indices,
        'reconstruction_error': mse,
        'threshold': threshold,
        'is_anomaly_point': is_anomaly_point,
        'triggered_anomaly': triggered_anomaly,
        'anomaly_score': mse / threshold,
        'consecutive_count': consecutive_count
    })
    
    # Add feature errors
    for i, col in enumerate(feature_columns):
        results[f'error_{col}'] = feature_mse[:, i]
    
    # Add timestamp mapping
    time_col = None
    if 'datetime' in df_original.columns:
        time_col = 'datetime'
    elif 'timestamp' in df_original.columns:
        time_col = 'timestamp'
    
    if time_col:
        # Add END timestamp (when the sequence completes)
        results['timestamp'] = df_original[time_col].iloc[sequence_end_indices].values
        
        # Add START timestamp (when the sequence begins)
        sequence_start_indices = sequence_end_indices - (sequence_length - 1)
        results['timestamp_start'] = df_original[time_col].iloc[sequence_start_indices].values
        
        print(f"   ✓ Added timestamp columns (sequence start and end times)")
    
    # Add compressor_id if available
    if 'compressor_id' in df_original.columns:
        results['compressor_id'] = df_original['compressor_id'].iloc[sequence_end_indices].values
    
    # Print summary with timestamp information
    total = len(results)
    anomaly_points = results['is_anomaly_point'].sum()
    triggered = results['triggered_anomaly'].sum()
    
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"   Total sequences: {total}")
    print(f"   Points exceeding threshold: {anomaly_points} ({100*anomaly_points/total:.2f}%)")
    print(f"   Triggered anomalies: {triggered} ({100*triggered/total:.2f}%)")
    
    # Show triggered anomalies with timestamps
    if triggered > 0 and time_col:
        print(f"\n🚨 TRIGGERED ANOMALIES WITH TIMESTAMPS:")
        print("-" * 70)
        
        triggered_df = results[results['triggered_anomaly']].copy()
        
        # Group consecutive anomalies
        triggered_df['anomaly_group'] = (triggered_df['sequence_index'].diff() != 1).cumsum()
        
        for group_id, group in triggered_df.groupby('anomaly_group'):
            start_time = group['timestamp_start'].iloc[0]
            end_time = group['timestamp'].iloc[-1]
            duration = len(group)
            max_score = group['anomaly_score'].max()
            
            print(f"\n   Event {group_id}:")
            print(f"      Sequence Range:   {group['sequence_index'].iloc[0]} → {group['sequence_index'].iloc[-1]}")
            print(f"      Time Start:       {start_time}")
            print(f"      Time End:         {end_time}")
            print(f"      Duration:         {duration} sequences ({end_time - start_time})")
            print(f"      Max Anomaly Score: {max_score:.3f}x")
    
    # Write results to database
    success = write_results_to_db(results, database, db_profile, results_table, if_exists='replace')
    
    if success:
        print("\n" + "="*70)
        print("✓ TESTING COMPLETE - RESULTS SAVED TO AZURE SQL DATABASE")
        print("="*70)
    
    return results


# Example usage
if __name__ == "__main__":
    # Test with Azure SQL database
    results = test_model_from_db(
        database='AOM-Dev',
        db_profile='AzureSQL',
        source_table='compressor_normal_dataset3',
        results_table='anomaly_detection_results',
        compressor_id='Compressor_A',
        limit=10000,
        sequence_length=10,
        consecutive_threshold=3
    )