import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
backtest.py — Testa a estratégia lendo do banco SQLite e salva o histórico.

Uso:
    python backtest.py --ativo bitcoin --intervalo 1h
    python backtest.py --ativo bitcoin --intervalo 1h --salvar   ← salva no banco
"""

import argparse
import os
import numpy as np
from indicators import add_indicators
from tensorflow.keras.models import load_model
from database import criar_tabelas, buscar_prices, registrar_trade, registrar_equity
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--ativo",     default="bitcoin")
parser.add_argument("--intervalo", default="1h")
parser.add_argument("--capital",   type=float, default=1000.0)
parser.add_argument("--window",    type=int,   default=20)
parser.add_argument("--compra",    type=float, default=0.65)
parser.add_argument("--venda",     type=float, default=0.35)
parser.add_argument("--stop",      type=float, default=0.97)
parser.add_argument("--gain",      type=float, default=1.05)
parser.add_argument("--salvar",    action="store_true",
                    help="Salva operações no banco")
args = parser.parse_args()

criar_tabelas()

model_path  = f"models/lstm_model_{args.ativo}_{args.intervalo}.keras"
scaler_path = f"models/scaler_{args.ativo}_{args.intervalo}.pkl"

for p in [model_path, scaler_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f" Não encontrado: {p}\n   Execute train.py primeiro."
        )

print(f"Carregando {args.ativo} ({args.intervalo}) do banco...")
df = buscar_prices(args.ativo, args.intervalo)

if df.empty:
    raise ValueError(
        f" Sem dados no banco para {args.ativo} {args.intervalo}.\n"
        f"   Execute: python updater.py --once --intervalo {args.intervalo}"
    )

df = add_indicators(df)
df = df.dropna().reset_index(drop=True)
print(f" {len(df)} candles carregadas")

FEATURES = ["SMA20", "EMA20", "RSI", "MACD", "ADX", "BB_width", "Vol_Log"]
model     = load_model(model_path)
scaler    = joblib.load(scaler_path)
data      = scaler.transform(df[FEATURES].values)

capital     = args.capital
position    = 0.0
entry_price = 0.0
window      = args.window
stats       = {"compras": 0, "stops": 0, "gains": 0, "vendas": 0}

for i in range(window, len(data) - 1):
    X    = np.array([data[i - window:i]])
    pred = float(model.predict(X, verbose=0)[0][0])
    price = float(df["Close"].iloc[i])
    rsi   = float(df["RSI"].iloc[i])

    if pred > args.compra and rsi < 65 and position == 0:
        position    = capital / price
        entry_price = price
        capital     = 0.0
        stats["compras"] += 1
        if args.salvar:
            registrar_trade(args.ativo, args.intervalo, "compra",
                            price, position, 0.0, pred, rsi)
        print(f"  🟢 Compra  @ {price:.4f}  pred={pred:.2f}  RSI={rsi:.1f}")

    elif pred < args.venda and position > 0:
        capital  = position * price
        position = 0.0
        stats["vendas"] += 1
        if args.salvar:
            registrar_trade(args.ativo, args.intervalo, "venda",
                            price, 0.0, capital, pred, rsi)
            registrar_equity(args.ativo, capital)
        print(f"  🔴 Venda   @ {price:.4f}  pred={pred:.2f}")

    if position > 0:
        if price <= entry_price * args.stop:
            capital  = position * price
            position = 0.0
            stats["stops"] += 1
            if args.salvar:
                registrar_trade(args.ativo, args.intervalo, "stop",
                                price, 0.0, capital, pred, rsi)
                registrar_equity(args.ativo, capital)
            print(f"  🛑 Stop    @ {price:.4f}  ({(price/entry_price-1)*100:.1f}%)")

        elif price >= entry_price * args.gain:
            capital  = position * price
            position = 0.0
            stats["gains"] += 1
            if args.salvar:
                registrar_trade(args.ativo, args.intervalo, "gain",
                                price, 0.0, capital, pred, rsi)
                registrar_equity(args.ativo, capital)
            print(f"  💰 Gain    @ {price:.4f}  ({(price/entry_price-1)*100:.1f}%)")


final = capital if position == 0 else position * float(df["Close"].iloc[-1])
lucro = final - args.capital
pct   = (lucro / args.capital) * 100

print(f"""
╔══════════════════════════════════════╗
║          RESULTADO BACKTEST          ║
╠══════════════════════════════════════╣
  Ativo:           {args.ativo.upper()} ({args.intervalo})
  Capital inicial: R$ {args.capital:,.2f}
  Capital final:   R$ {final:,.2f}
  Lucro:           R$ {lucro:+,.2f}  ({pct:+.1f}%)
  ──────────────────────────────────
  Entradas:        {stats['compras']}
  Stops:           {stats['stops']}
  Take profits:    {stats['gains']}
  Vendas p/ sinal: {stats['vendas']}
  ──────────────────────────────────
  Salvo no banco:  {'Sim ✅' if args.salvar else 'Não (use --salvar)'}
╚══════════════════════════════════════╝
""")