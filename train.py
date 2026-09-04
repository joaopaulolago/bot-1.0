import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import pandas as pd
from indicators import add_indicators
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib

parser = argparse.ArgumentParser()

parser.add_argument("--ativo", default="bitcoin", 
                    choices=["bitcoin", "ethereum", "solana", "nvidia", "tesla", "ouro", "sp500", "petroleo"],
                    help="Escolha o ativo para treinar")

parser.add_argument("--intervalo", default="1h", help="Intervalo: 5m ou 1h")
parser.add_argument("--epochs", type=int, default=50) 
parser.add_argument("--window", type=int, default=60)
args = parser.parse_args()

os.makedirs("models", exist_ok=True)

caminho = f"data/{args.ativo}_{args.intervalo}.csv"
if not os.path.exists(caminho):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho}\nExecute primeiro: python dataset.py")

df = pd.read_csv(caminho)
print(f" Dados carregados para {args.ativo}: {len(df)} linhas")

df = add_indicators(df)
df = df.dropna()


FEATURES = ["SMA20", "EMA20", "RSI", "MACD", "ADX","BB_width","Vol_Log"] 
data = df[FEATURES].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

window = args.window
X, y = [], []


for i in range(window, len(data_scaled) - 1):
    X.append(data_scaled[i - window:i])
    y.append(1 if df["Close"].iloc[i + 1] > df["Close"].iloc[i] else 0)

X = np.array(X) 
y = np.array(y)

print(f"📈 Shape de treino: X={X.shape} y={y.shape}")

model = Sequential([
    Input(shape=(window, X.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

print(f"🧠 Treinando {args.ativo.upper()}...")
model.fit(
    X, y,
    epochs=args.epochs,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=1
)

model_path  = f"models/lstm_model_{args.ativo}_{args.intervalo}.keras"
scaler_path = f"models/scaler_{args.ativo}_{args.intervalo}.pkl"

model.save(model_path)
joblib.dump(scaler, scaler_path)

print(f" Concluído")
print(f" Modelo: {model_path}")
print(f" Scaler: {scaler_path}")