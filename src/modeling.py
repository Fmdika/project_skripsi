import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from tensorflow import keras



# ==========================
# 1️⃣ Train-Test Split
# ==========================
def train_test_split_time_series(df, train_size=0.8):
    split_index = int(len(df) * train_size)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


# ==========================
# 2️⃣ Scaling
# ==========================
def scale_data(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train)

    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled, scaler


# ==========================
# 3️⃣ Windowing
# ==========================
def create_sequences(data, window_size=30):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])  # kolom pertama = IHSG (target)

    return np.array(X), np.array(y)


# ==========================
# 4️⃣ Build LSTM Model
# ==========================
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=False, input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


# ==========================
# 5️⃣ Evaluation
# ==========================
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mape, r2


# ==========================
# Test Model Multi Output
# ==========================
def create_sequences_multi_output(data, window_size=30):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])  # semua variabel jadi target

    return np.array(X), np.array(y)

def build_lstm_multi_output(input_shape, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(n_features)  # output 5 variabel
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

# ==========================
# Test Multi-Step Windowing
# ==========================
def create_sequences_multi_step(data, window_size=30, horizon=30):
    X, y = [], []

    for i in range(window_size, len(data) - horizon + 1):
        X.append(data[i-window_size:i])
        y.append(data[i:i+horizon, 0])  # hanya IHSG sebagai target

    return np.array(X), np.array(y)

# ==========================
# Build Multi-Step LSTM
# ==========================
def build_lstm_multi_step(input_shape, horizon):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(horizon)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

# ==========================
# Test Model Return
# ==========================

# Fungsi hitung log return
def add_log_return(df):
    df = df.copy()
    df["ihsg_return"] = np.log(df["ihsg"] / df["ihsg"].shift(1))
    df = df.dropna()
    return df

# Sequence return (single target)
def create_sequences_return(data, window_size=30):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])  # kolom pertama = ihsg_return

    return np.array(X), np.array(y)

# Build model return
def build_lstm_return(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

