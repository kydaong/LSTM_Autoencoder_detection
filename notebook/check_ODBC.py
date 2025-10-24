import joblib

# Load your scaler
scaler = joblib.load('C:/Users/adminuser/Projects/LSTM_Autoencoder_detection/models/scaler2.pkl')

# Check how many features it expects
print(f"Scaler expects {scaler.n_features_in_} features")

# If your scaler has feature names stored
if hasattr(scaler, 'feature_names_in_'):
    print("\nFeature names used during training:")
    for i, name in enumerate(scaler.feature_names_in_):
        print(f"{i+1}. {name}")
else:
    print("\nFeature names not stored in scaler")
    print("You need to check your training code to see what 11 features were used")
