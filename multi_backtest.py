import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import os
import numpy as np
import pandas as pd
from indicators import add_indicators
from tensorflow.keras.models import load_model
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--intervalo", default="1h")
parser.add_argument("--capital",   type=float, default=1000.0)
parser.add_argument("--window",    type=int,   default=20)
parser.add_argument("--compra",    type=float, default=0.65)
parser.add_argument("--venda",     type=float, default=0.35)
parser.add_argument("--stop",      type=float, default=0.97)
parser.add_argument("--gain",      type=float, default=1.05)
args = parser.parse_args()

from config import ATIVOS
FEATURES = ["SMA20", "EMA20", "RSI", "MACD", "ADX", "BB_width", "Vol_Log"]

def rodar_backtest(nome):
    data_path   = f"data/{nome}_{args.intervalo}.csv"
    model_path  = f"models/lstm_model_{args.ativo}_{args.intervalo}.keras"
    scaler_path = f"models/scaler_{nome}_{args.intervalo}.pkl"

    for p in [data_path, model_path, scaler_path]:
        if not os.path.exists(p):
            print(f" Pulando {nome}: {p} não encontrado")
            return None

    df = pd.read_csv(data_path)
    df = add_indicators(df)
    df = df.dropna()

    if len(df) < 60:
        print(f" {nome}: dados insuficientes ({len(df)} linhas)")
        return None

    model  = load_model(model_path,  compile=False)
    scaler = joblib.load(scaler_path)

    model.compile(optimizer="adam", loss="binary_crossentropy")

    data = scaler.transform(df[FEATURES].values)

    capital     = args.capital
    position    = 0.0
    entry_price = 0.0
    n_ops       = 0
    window      = args.window

    for i in range(window, len(data) - 1):
        X    = np.array([data[i - window:i]])
        pred = model.predict(X, verbose=0)[0][0]
        price = float(df["Close"].iloc[i])
        rsi   = float(df["RSI"].iloc[i])

        if pred > args.compra and rsi < 65 and position == 0:
            position    = capital / price
            entry_price = price
            capital     = 0.0
            n_ops      += 1

        elif pred < args.venda and position > 0:
            capital  = position * price
            position = 0.0

        if position > 0:
            if price <= entry_price * args.stop:
                capital  = position * price
                position = 0.0
            elif price >= entry_price * args.gain:
                capital  = position * price
                position = 0.0

    final = capital if position == 0 else position * float(df["Close"].iloc[-1])
    lucro = final - args.capital
    pct   = (lucro / args.capital) * 100

    return {"lucro": lucro, "pct": pct, "ops": n_ops, "final": final}

resultados = {}

for ativo in ATIVOS:
    print(f"\n Testando {ativo}...")
    r = rodar_backtest(ativo)
    if r is not None:
        resultados[ativo] = r
        sinal = "📈" if r["lucro"] >= 0 else "📉"
        print(f"  {sinal} Lucro: R$ {r['lucro']:+,.2f} ({r['pct']:+.1f}%)  |  {r['ops']} operações")

print("\n" + "="*45)
print("  RANKING FINAL")
print("="*45)

if not resultados:
    print(" Nenhum ativo com modelo treinado.")
    print("   Execute: python train.py --ativo <nome>")
else:
    ranking = sorted(resultados.items(), key=lambda x: x[1]["lucro"], reverse=True)
    for pos, (ativo, r) in enumerate(ranking, 1):
        emoji = "" if pos == 1 else ("" if pos == 2 else "" if pos == 3 else "  ")
        print(f"  {emoji} {pos}. {ativo.upper():<12} R$ {r['lucro']:>+9,.2f}  ({r['pct']:+.1f}%)")

    melhor = ranking[0][0]
    print(f"\n Melhor ativo: {melhor.upper()} → R$ {resultados[melhor]['final']:,.2f}")