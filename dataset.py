import yfinance as yf
import os
import pandas as pd

os.makedirs("data", exist_ok=True)

from config import ATIVOS

def baixar_e_salvar(nome, ticker, period, interval, sufixo):
    """Baixa dados e corrige colunas MultiIndex do yfinance novo."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        print(f" Sem dados para {nome} ({interval})")
        return False


    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()


    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    colunas = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in colunas:
        if col not in df.columns:
            print(f" Coluna '{col}' ausente em {nome} ({interval})")
            return False

    df = df[colunas]
    caminho = f"data/{nome}_{sufixo}.csv"
    df.to_csv(caminho, index=False)
    print(f"  {nome} ({interval}) salvo → {caminho}  [{len(df)} linhas]")
    return True


for nome, ticker in ativos.items():
    print(f"\n Baixando {nome} ({ticker})...")
    baixar_e_salvar(nome, ticker, period="60d",  interval="5m",  sufixo="5m")
    baixar_e_salvar(nome, ticker, period="730d", interval="1h",  sufixo="1h")

print("\n DOWNLOAD FINALIZADO")