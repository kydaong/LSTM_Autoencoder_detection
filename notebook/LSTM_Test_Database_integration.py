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

def load_model_components(model_path="models/compressor_autoencoder2.h5",
                          scaler_path="models/scaler2.pkl",
                          config_path="models/model_config2.pkl"):
    """Load the saved model, scaler, and configuration"""
    print("="*70)
    print("LOADING MODEL COMPONENTS")
    print("="*70)

    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_model_path = os.path.join(PROJECT_ROOT, model_path) if not os.path.isabs(model_path) else model_path
        full_scaler_path = os.path.join(PROJECT_ROOT, scaler_path) if not os.path.isabs(scaler_path) else scaler_path
        full_config_path = os.path.join(PROJECT_ROOT, config_path) if not os.path.isabs(config_path) else config_path

        print(f"\n📂 PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"📂 model_path (relative): {model_path}")
        print(f"📂 full_model_path (absolute): {full_model_path}")
        print(f"📂 File exists? {os.path.exists(full_model_path)}")
        print(f"📂 Models folder exists? {os.path.exists(os.path.join(PROJECT_ROOT, 'models'))}")
        print(f"📂 Files in models folder: {os.listdir(os.path.join(PROJECT_ROOT, 'models')) if os.path.exists(os.path.join(PROJECT_ROOT, 'models')) else 'FOLDER NOT FOUND'}")
        
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


from sqlalchemy import text  # Add this import at the top

def load_data_from_db(db_profile='AzureSQL', table_name='compressor_normal_dataset3', 
                      compressor_id=None, limit=None):
    """
    Load test data from SQL Server database
    
    Parameters:
    - db_profile: Database profile name from config
    - table_name: Name of the table to read from
    - compressor_id: Optional filter for specific compressor
    - limit: Optional limit on number of rows to read
    """
    print("\n" + "="*70)
    print("LOADING DATA FROM DATABASE")
    print("="*70)
    
    try:
        # Initialize SQL connection
        print(f"\n📌 Connecting to database profile: {db_profile}")
        
        sql_conn = SQLConnector(database='AOM-Dev', db_profile=db_profile)
        
        # Check if connection was successful
        if sql_conn is None or not hasattr(sql_conn, 'engine') or sql_conn.engine is None:
            print("❌ Failed to create database connection")
            print("   Check your sql_config.yaml file")
            return None
        
        print(f"   ✓ Connected successfully")
        
        # Test the connection with text()
        try:
            with sql_conn.engine.connect() as conn:
                test_result = conn.execute(text("SELECT 1 AS test"))
                print(f"   ✓ Connection test successful")
        except Exception as conn_error:
            print(f"❌ Connection test failed: {conn_error}")
            return None
        
        # Build query
        where_clause = ""
        if compressor_id:
            where_clause = f" WHERE compressor_id = '{compressor_id}'"
        
        # Build complete query
        if limit:
            query = f"""
            SELECT TOP {limit} * 
            FROM {table_name}
            {where_clause}
            ORDER BY datetime
            """
        else:
            query = f"""
            SELECT * 
            FROM {table_name}
            {where_clause}
            ORDER BY datetime
            """
        
        # Clean up the query
        query = ' '.join(query.split())
        
        print(f"\n📊 Executing query:")
        print(f"   {query}")
        
        # Execute query - pandas handles the text() wrapper internally
        df = pd.read_sql(query, sql_conn.engine)
        
        print(f"\n✓ Loaded {len(df)} rows from database")
        
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
        print("\nFull traceback:")
        traceback.print_exc()
        return None
    
def write_results_to_db(results, db_profile='AzureSQL', 
                        table_name='anomaly_detection_results',
                        if_exists='replace'):  # Changed default to 'replace' for testing
    """
    Write anomaly detection results to database
    """
    print("\n" + "="*70)
    print("WRITING RESULTS TO DATABASE")
    print("="*70)
    
    try:
        # Initialize SQL connection
        sql_conn = SQLConnector(database='AOM-Dev', db_profile=db_profile)  
        
        if sql_conn is None or not hasattr(sql_conn, 'engine') or sql_conn.engine is None:
            print("❌ Failed to create database connection")
            return False
        
        # Create a clean copy
        df = results.copy()
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Convert boolean columns to integers (SQL Server compatibility)
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        if bool_cols:
            print(f"\n🔄 Converting boolean columns to integers: {bool_cols}")
            for col in bool_cols:
                df[col] = df[col].astype(int)
        
        # Rename 'timestamp' column if it exists (it's a reserved keyword in some SQL dialects)
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'timestamp_value'})
            print(f"   ℹ️ Renamed 'timestamp' to 'timestamp_value' (reserved keyword)")
        
        # Add analysis timestamp
        df['analysis_timestamp'] = datetime.now()
        
        # Replace inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Ensure all column names are valid SQL identifiers (no spaces, special chars)
        df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        print(f"\n💾 Preparing to write {len(df)} rows to table: {table_name}")
        print(f"   Mode: {if_exists}")
        print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"   Column types:")
        for col in df.columns:
            print(f"      - {col}: {df[col].dtype}")
        
        # If table exists and mode is 'append', check if we need to drop it first
        from sqlalchemy import inspect
        inspector = inspect(sql_conn.engine)
        
        if table_name in inspector.get_table_names():
            if if_exists == 'replace':
                print(f"\n⚠️ Table '{table_name}' exists. Dropping it...")
                with sql_conn.engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE {table_name}"))
                    conn.commit()
                print(f"   ✓ Table dropped")
        
        # Write to database using method=None (single inserts, more reliable)
        print(f"\n💾 Writing data...")
        df.to_sql(
            name=table_name,
            con=sql_conn.engine,
            if_exists='append',  # Always append since we dropped if needed
            index=False,
            method=None,  # Use single inserts (slower but more reliable)
            chunksize=None
        )
        
        print(f"   ✓ Data written successfully")
        
        # Verify write
        verify_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        with sql_conn.engine.connect() as conn:
            result = conn.execute(text(verify_query))
            count = result.fetchone()[0]
            print(f"   ✓ Verified: {count} rows in table '{table_name}'")
        
        # Show sample of written data
        sample_query = f"SELECT TOP 3 * FROM {table_name}"
        sample_df = pd.read_sql(sample_query, sql_conn.engine)
        print(f"\n📊 Sample of written data:")
        print(sample_df.to_string())
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error writing to database: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to provide helpful debugging info
        print("\n🔍 Debugging info:")
        print(f"   DataFrame shape: {df.shape if 'df' in locals() else 'N/A'}")
        print(f"   DataFrame columns: {list(df.columns) if 'df' in locals() else 'N/A'}")
        
        return False
    
def preprocess_test_data(df, scaler, feature_columns):
    """Preprocess test data using saved scaler"""
    print("\n" + "="*70)
    print("PREPROCESSING TEST DATA")
    print("="*70)
    
    # Check for timestamp
    has_timestamp = 'timestamp' in df.columns
    if has_timestamp:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    
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


def test_model_from_db(db_profile='AzureSQL',
                       source_table='compressor_normal_dataset3',
                       results_table='anomaly_detection_results',
                       compressor_id=None,
                       limit=None,
                       model_path='compressor_autoencoder2.h5',
                       scaler_path='scaler2.pkl',
                       config_path='model_config2.pkl',
                       sequence_length=10,
                       consecutive_threshold=3):
    """
    Complete pipeline: Load from DB -> Test -> Write results to DB
    """
    print("\n" + "="*70)
    print("LSTM AUTOENCODER - DATABASE TESTING")
    print("="*70)
    
    # Load model components
    model, scaler, threshold, feature_columns = load_model_components()

    
    if model is None:
        return None
    
    # Load data from database
    df = load_data_from_db(db_profile, source_table, compressor_id, limit)
    
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
    
    # Create results DataFrame
    results = pd.DataFrame({
        'sequence_index': range(len(mse)),
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
    
    # Add timestamps if available
    if 'timestamp' in df_original.columns:
        start_idx = sequence_length - 1
        results['timestamp'] = df_original['timestamp'].iloc[start_idx:start_idx + len(results)].values
    
    # Add compressor_id if available
    if 'compressor_id' in df_original.columns:
        start_idx = sequence_length - 1
        results['compressor_id'] = df_original['compressor_id'].iloc[start_idx:start_idx + len(results)].values
    
    # Print summary
    total = len(results)
    anomaly_points = results['is_anomaly_point'].sum()
    triggered = results['triggered_anomaly'].sum()
    
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"   Total sequences: {total}")
    print(f"   Points exceeding threshold: {anomaly_points} ({100*anomaly_points/total:.2f}%)")
    print(f"   Triggered anomalies: {triggered} ({100*triggered/total:.2f}%)")
    
    # Write results to database
    success = write_results_to_db(results, db_profile, results_table, if_exists='append')
    
    if success:
        print("\n" + "="*70)
        print("✓ TESTING COMPLETE - RESULTS SAVED TO DATABASE")
        print("="*70)
    
    return results


# Example usage
if __name__ == "__main__":
    # Test with database
    results = test_model_from_db(
        db_profile='AzureSQL',
        source_table='compressor_normal_dataset3',
        results_table='anomaly_detection_results',
        compressor_id='Compressor_A',  # Optional: filter specific compressor
        limit=10000,  # Optional: limit rows for testing
        sequence_length=10,
        consecutive_threshold=3
    )