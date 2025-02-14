import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

def prepare_data(data, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length - 12):  # Predicting 1-year ahead (12 months)
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length+12].values)  # Predict all asset classes
    return np.array(X), np.array(y)

# Generate random dataset
np.random.seed(42)
n_samples = 1000
n_features = 10
data = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'Feature_{i}' for i in range(n_features)])
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Prepare dataset
X, y = prepare_data(data_scaled, seq_length=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(y.shape[1])  # Output layer for multiple assets
])

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Predict forward 1-year returns
y_pred = model.predict(X_test)

# Explainability with SHAP
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# Plot summary of feature importance
shap.summary_plot(shap_values, feature_names=data.columns)
plt.show()
