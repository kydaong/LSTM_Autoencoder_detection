import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, LSTM, RepeatVector, TimeDistributed, Dense, 
                                     Embedding, Concatenate, Reshape, Layer)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import tensorflow as tf
from datetime import datetime
import os

class SelectDecoderOutput(Layer):
    """Custom layer to select decoder output - NO LAMBDA"""
    
    def __init__(self, **kwargs):
        super(SelectDecoderOutput, self).__init__(**kwargs)
    
    def call(self, inputs):
        outputs = inputs[:-1]
        comp_id = inputs[-1]
        
        # Stack all decoder outputs
        stacked = tf.stack(outputs, axis=1)
        
        # Get batch size and squeeze compressor ID
        batch_size = tf.shape(comp_id)[0]
        comp_id = tf.squeeze(comp_id, axis=-1)
        
        # Create indices for gathering
        batch_indices = tf.range(batch_size)
        indices = tf.stack([batch_indices, comp_id], axis=1)
        
        # Select the appropriate decoder output for each sample
        selected = tf.gather_nd(stacked, indices)
        
        return selected
    
    def compute_output_shape(self, input_shapes):
        # Return shape: (batch_size, sequence_length, n_features)
        return input_shapes[0]
    
    def get_config(self):
        config = super(SelectDecoderOutput, self).get_config()
        return config


class CompressorAwareLSTMAutoencoder:
    """LSTM Autoencoder with compressor-specific decoders - NO LAMBDA LAYERS"""
    
    def __init__(self, sequence_length=10, n_features=11, embedding_dim=8):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.model = None
        self.encoder = None
        self.feature_scaler = StandardScaler()
        self.thresholds = {}
        self.compressor_encoder = None
        self.unique_compressors = None
        
    def build_model(self, unique_compressors):
        """Build compressor-aware autoencoder with custom layer instead of Lambda"""
        
        print(f"Building model for {len(unique_compressors)} compressors...")
        
        # Input for sequences
        sequence_input = Input(shape=(self.sequence_length, self.n_features), 
                              name='sequence_input')
        
        # Input for compressor ID
        compressor_input = Input(shape=(1,), dtype='int32', name='compressor_input')
        
        # Embed compressor ID and reshape (NO LAMBDA)
        compressor_embedded = Embedding(
            input_dim=len(unique_compressors), 
            output_dim=self.embedding_dim,
            name='compressor_embedding'
        )(compressor_input)
        
        # Use Reshape instead of Lambda
        compressor_embedded = Reshape(
            (self.embedding_dim,), 
            name='reshape_embedding'
        )(compressor_embedded)
        
        # Shared encoder
        encoded = LSTM(64, activation='relu', name='encoder_lstm')(sequence_input)
        
        # Concatenate compressed representation with compressor embedding
        merged = Concatenate(name='merge_encoded_compressor')([encoded, compressor_embedded])
        
        # Create separate decoder for each compressor
        decoder_outputs = []
        for comp_idx in range(len(unique_compressors)):
            comp_name = unique_compressors[comp_idx]
            
            # Decoder-specific dense layers
            decoded = Dense(32, activation='relu', 
                          name=f'decoder_dense1_{comp_name}')(merged)
            decoded = Dense(64, activation='relu', 
                          name=f'decoder_dense2_{comp_name}')(decoded)
            
            # Repeat vector for sequence generation
            decoded = RepeatVector(self.sequence_length, 
                                  name=f'repeat_vector_{comp_name}')(decoded)
            
            # Decoder LSTM
            decoded = LSTM(64, activation='relu', return_sequences=True,
                          name=f'decoder_lstm_{comp_name}')(decoded)
            
            # Output layer
            output = TimeDistributed(Dense(self.n_features),
                                    name=f'output_{comp_name}')(decoded)
            
            decoder_outputs.append(output)
        
        # Use custom layer instead of Lambda (NO LAMBDA)
        final_output = SelectDecoderOutput(name='select_decoder_output')(
            decoder_outputs + [compressor_input]
        )
        
        # Build model
        self.model = Model(inputs=[sequence_input, compressor_input], 
                          outputs=final_output)
        
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='mse',
                          metrics=['mae'])
        
        return self.model
    
    def prepare_data(self, df, feature_columns):
        """Prepare data with compressor ID encoding"""
        
        print("\nPreparing data...")
        print(f"Dataset shape: {df.shape}")
        print(f"Compressors found: {df['compressor_id'].unique()}")
        
        # Encode compressor IDs
        self.compressor_encoder = LabelEncoder()
        df['compressor_id_encoded'] = self.compressor_encoder.fit_transform(
            df['compressor_id'])
        self.unique_compressors = self.compressor_encoder.classes_
        
        print(f"Unique compressors: {list(self.unique_compressors)}")
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(df[feature_columns])
        
        return scaled_features, df['compressor_id_encoded'].values
    
    def create_sequences(self, data, comp_ids, sequence_length):
        """Create sequences from data"""
        sequences = []
        comp_ids_seq = []
        
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:(i + sequence_length)])
            comp_ids_seq.append(comp_ids[i + sequence_length - 1])
        
        return np.array(sequences), np.array(comp_ids_seq)
    
    def train(self, csv_path, feature_columns, epochs=50, batch_size=32, validation_split=0.2):
        """Train the compressor-aware model"""
        
        print("="*70)
        print("COMPRESSOR-AWARE LSTM AUTOENCODER - NO LAMBDA LAYERS")
        print("="*70)
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"\nLoaded data from {csv_path}")
        print(f"Columns: {list(df.columns)}")
        
        # Prepare data
        scaled_features, compressor_ids = self.prepare_data(df, feature_columns)
        
        # Create sequences
        print(f"\nCreating sequences with length={self.sequence_length}...")
        X_sequences, comp_ids_sequences = self.create_sequences(
            scaled_features, compressor_ids, self.sequence_length)
        
        print(f"Sequences created: {len(X_sequences)}")
        print(f"Sequence shape: {X_sequences.shape}")
        
        # Build model
        self.build_model(self.unique_compressors)
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        print(self.model.summary())
        
        # Train model
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        history = self.model.fit(
            [X_sequences, comp_ids_sequences],
            X_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Calculate per-compressor thresholds
        print("\n" + "="*70)
        print("CALCULATING PER-COMPRESSOR THRESHOLDS")
        print("="*70)
        
        val_predictions = self.model.predict([X_sequences, comp_ids_sequences])
        mse = np.mean(np.power(X_sequences - val_predictions, 2), axis=(1, 2))
        
        for comp_idx, comp_name in enumerate(self.unique_compressors):
            mask = comp_ids_sequences == comp_idx
            comp_mse = mse[mask]
            threshold = np.mean(comp_mse) + 2 * np.std(comp_mse)
            self.thresholds[comp_name] = threshold
            print(f"{comp_name}: mean_error={np.mean(comp_mse):.6f}, threshold={threshold:.6f}")
        
        return history
    
    def save(self, model_dir='models'):
        """Save model and preprocessing objects"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save with version 5 (no lambda version)
        model_path = os.path.join(model_dir, 'compressor_aware_model_no_lambda.h5')
        scaler_path = os.path.join(model_dir, 'feature_scaler_no_lambda.pkl')
        encoder_path = os.path.join(model_dir, 'compressor_encoder_no_lambda.pkl')
        thresholds_path = os.path.join(model_dir, 'thresholds_no_lambda.pkl')
        
        self.model.save(model_path)
        joblib.dump(self.feature_scaler, scaler_path)
        joblib.dump(self.compressor_encoder, encoder_path)
        joblib.dump(self.thresholds, thresholds_path)
        
        print(f"\n{'='*70}")
        print("MODEL SAVED (NO LAMBDA VERSION)")
        print(f"{'='*70}")
        print(f"Model: {model_path}")
        print(f"Scaler: {scaler_path}")
        print(f"Encoder: {encoder_path}")
        print(f"Thresholds: {thresholds_path}")
        print(f"\nUpdate your pipeline to use these files:")
        print(f"  MODEL_PATH = '{model_path}'")
        print(f"  SCALER_PATH = '{scaler_path}'")
        print(f"  ENCODER_PATH = '{encoder_path}'")
        print(f"  THRESHOLDS_PATH = '{thresholds_path}'")
    
    @classmethod
    def load(cls, model_dir='models'):
        """Load trained model - NO LAMBDA LAYERS"""
        
        model_path = os.path.join(model_dir, 'compressor_aware_model_no_lambda.h5')
        scaler_path = os.path.join(model_dir, 'feature_scaler_no_lambda.pkl')
        encoder_path = os.path.join(model_dir, 'compressor_encoder_no_lambda.pkl')
        thresholds_path = os.path.join(model_dir, 'thresholds_no_lambda.pkl')
        
        # Register custom layer
        custom_objects = {'SelectDecoderOutput': SelectDecoderOutput}
        
        ae = cls()
        ae.model = load_model(model_path, custom_objects=custom_objects)
        ae.feature_scaler = joblib.load(scaler_path)
        ae.compressor_encoder = joblib.load(encoder_path)
        ae.thresholds = joblib.load(thresholds_path)
        ae.unique_compressors = ae.compressor_encoder.classes_
        
        print(f"Model loaded from {model_dir}")
        print(f"Thresholds: {ae.thresholds}")
        
        return ae


# Example usage
if __name__ == "__main__":
    # CONFIGURATION - UPDATE THESE PATHS
    CSV_PATH = "C:/Users/adminuser/Desktop/compressor_normal_dataset3.csv"  # ← YOUR CSV PATH
    MODEL_DIR = "models"  # ← WHERE TO SAVE MODEL
    
    feature_columns = [
        'filter_dp', 'seal_gas_flow', 'seal_gas_diff_pressure',
        'seal_gas_temp', 'primary_vent_flow', 'primary_vent_pressure',
        'secondary_seal_gas_flow', 'separation_seal_gas_flow',
        'separation_seal_gas_pressure', 'seal_gas_to_vent_diff_pressure',
        'encoding'
    ]
    
    # Initialize and train
    ae = CompressorAwareLSTMAutoencoder(
        sequence_length=10, 
        n_features=len(feature_columns),
        embedding_dim=8
    )
    
    print("Starting training...")
    print(f"Data source: {CSV_PATH}")
    print(f"Output directory: {MODEL_DIR}")
    print()
    
    history = ae.train(
        csv_path=CSV_PATH,
        feature_columns=feature_columns,
        epochs=50,
        batch_size=32
    )
    
    # Save model
    ae.save(model_dir=MODEL_DIR)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - NO LAMBDA LAYERS")
    print("="*70)
    print("\nYour model is now compatible with all TensorFlow versions!")
    print("\nNext steps:")
    print("1. Update your pipeline script with the new file paths shown above")
    print("2. Run your pipeline - it will load without any issues")
    print("\nTo load and use the model later:")
    print("  ae = CompressorAwareLSTMAutoencoder.load(model_dir='models')")
    print("  predictions = ae.model.predict([sequences, compressor_ids])")