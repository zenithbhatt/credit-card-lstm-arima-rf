import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load ETL Output
data = pd.read_excel("Card Patterns/Profitability_Metrics.xlsx")

# ----------------------------
# ARIMA Forecasting
# ----------------------------
series = data["Annual_Spend"]

# Hyperparameter tuning for ARIMA
best_aic = float("inf")
best_order = None
for p in range(1, 6):
    for d in range(1, 2):
        for q in range(1, 6):
            try:
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, d, q)
            except:
                continue

# Fit ARIMA model with best parameters
model = ARIMA(series, order=best_order)
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)

# Plot ARIMA results
plt.figure(figsize=(10, 6))
plt.plot(series, label="Actual")
plt.plot(range(len(series), len(series) + 12), forecast, label="Forecast")
plt.legend()
plt.title(f"ARIMA Forecast (Order: {best_order})")
plt.show()

# ----------------------------
# Random Forest Regression
# ----------------------------
X = data[["Lifetime_Value", "Risk_Score", "Churn_Probability"]]
y = data["Annual_Spend"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_squared_error")
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Evaluate Random Forest
y_pred_rf = best_rf.predict(X_test)
print("Random Forest Best Parameters:", grid_search.best_params_)
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R^2:", r2_score(y_test, y_pred_rf))

# ----------------------------
# LSTM Forecasting
# ----------------------------
# Scaling data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))

# Prepare LSTM data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 12
X_lstm, y_lstm = create_sequences(series_scaled, seq_length)

# Split LSTM data
X_train_lstm, X_test_lstm = X_lstm[:-12], X_lstm[-12:]
y_train_lstm, y_test_lstm = y_lstm[:-12], y_lstm[-12:]

# Build LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation="relu", input_shape=(seq_length, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam", loss="mse")

# Fit LSTM model
history = model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=16, verbose=0)

# Evaluate LSTM
y_pred_lstm = model_lstm.predict(X_test_lstm)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
y_test_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

print("LSTM MSE:", mean_squared_error(y_test_lstm, y_pred_lstm))
print("LSTM R^2:", r2_score(y_test_lstm, y_pred_lstm))

# Plot LSTM results
plt.figure(figsize=(10, 6))
plt.plot(range(len(series)), scaler.inverse_transform(series_scaled), label="Actual")
plt.plot(range(len(series) - 12, len(series)), y_pred_lstm, label="LSTM Forecast")
plt.legend()
plt.title("LSTM Forecast")
plt.show()
