import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# 1. Configuration
WINDOW_SIZE = 3
FEATURES = 4  # s11, s12, s13, s15
MODEL_PATH = "models/future_model.keras"

print(f"🚀 Training new LSTM model with sequence window = {WINDOW_SIZE}...")

# 2. Generate some dummy data for training
# In a real scenario, you would load your NASA CMAPSS dataset here
# and use a sliding window of size 3 to create X_train and y_train.
num_samples = 1000
X_train = np.random.rand(num_samples, WINDOW_SIZE, FEATURES)
# Dummy target: predicting future risk score (0 to 1)
y_train = np.random.rand(num_samples, 1)

# 3. Build the LSTM Model
model = Sequential([
    LSTM(32, activation='relu', input_shape=(WINDOW_SIZE, FEATURES), return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # Outputting a continuous risk score
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4. Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# 5. Save the model
if not os.path.exists("models"):
    os.makedirs("models")
    
model.save(MODEL_PATH)
print(f"✅ Successfully saved new model to {MODEL_PATH}")
