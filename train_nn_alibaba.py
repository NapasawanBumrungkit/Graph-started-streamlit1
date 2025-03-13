import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


df = pd.read_csv("Ali_Baba_Stock_Data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

data = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

look_back = 60


def create_sequences(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, 0])  
        y.append(data[i, 0])  
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, look_back)

X_lstm = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

lstm_model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mean_squared_error")

lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

def evaluate_and_plot(model, X_test, y_test, title):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"{title} Test RMSE: {rmse}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

evaluate_and_plot(lstm_model, X_test_lstm, y_test, "Alibaba Stock Price Prediction using LSTM")
