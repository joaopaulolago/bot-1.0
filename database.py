"""
database.py — Cria e gerencia o banco SQLite.

Tabelas:
    prices  → candles OHLCV por ativo + intervalo
    trades  → histórico de todas as operações
    equity  → evolução do capital ao longo do tempo
"""

import sqlite3
import os

DB_PATH = "trader.db"


def get_conn():
    """Retorna conexão com o banco. Cria o arquivo se não existir."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row 
    return conn


def criar_tabelas():
    """Cria todas as tabelas se ainda não existirem."""
    conn = get_conn()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ativo     TEXT    NOT NULL,
            intervalo TEXT    NOT NULL,
            datetime  TEXT    NOT NULL,
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL    NOT NULL,
            volume    REAL,
            UNIQUE(ativo, intervalo, datetime)   -- evita duplicatas
        )
    """)


    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ativo       TEXT    NOT NULL,
            intervalo   TEXT    NOT NULL,
            tipo        TEXT    NOT NULL,   -- compra | venda | stop | gain
            preco       REAL    NOT NULL,
            quantidade  REAL    NOT NULL,
            capital     REAL    NOT NULL,   -- capital após a operação
            pred        REAL,               -- probabilidade do modelo
            rsi         REAL,               -- RSI no momento
            datetime    TEXT    DEFAULT (datetime('now','localtime'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS equity (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ativo     TEXT NOT NULL,
            capital   REAL NOT NULL,
            datetime  TEXT DEFAULT (datetime('now','localtime'))
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Banco de dados pronto →", DB_PATH)

def inserir_prices(df, ativo: str, intervalo: str):
    """
    Insere candles no banco.
    df deve ter colunas: Date (ou Datetime), Open, High, Low, Close, Volume.
    Duplicatas são ignoradas (INSERT OR IGNORE).
    """
    conn = get_conn()
    cur  = conn.cursor()

    col_data = "Datetime" if "Datetime" in df.columns else "Date"
    inseridos = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT OR IGNORE INTO prices
                    (ativo, intervalo, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ativo, intervalo,
                str(row[col_data]),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(row.get("Volume", 0))
            ))
            if cur.rowcount:
                inseridos += 1
        except Exception as e:
            pass 

    conn.commit()
    conn.close()
    return inseridos


def buscar_prices(ativo: str, intervalo: str, limit: int = 0):
    """Retorna DataFrame com os candles do banco, ordenado por datetime."""
    import pandas as pd

    conn  = get_conn()
    query = "SELECT * FROM prices WHERE ativo=? AND intervalo=? ORDER BY datetime"
    params = [ativo, intervalo]

    if limit:
        query += " DESC LIMIT ?"
        params.append(limit)
        df = pd.read_sql_query(query, conn, params=params)
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        df = pd.read_sql_query(query, conn, params=params)

    conn.close()

    df = df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low",   "close": "Close", "volume": "Volume"
    })
    return df
def registrar_trade(ativo, intervalo, tipo, preco, quantidade, capital,
                    pred=None, rsi=None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO trades
            (ativo, intervalo, tipo, preco, quantidade, capital, pred, rsi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ativo, intervalo, tipo, preco, quantidade, capital, pred, rsi))
    conn.commit()
    conn.close()


def registrar_equity(ativo, capital):
    conn = get_conn()
    conn.execute("INSERT INTO equity (ativo, capital) VALUES (?, ?)",
                 (ativo, capital))
    conn.commit()
    conn.close()


def resumo_trades(ativo=None):
    """Retorna DataFrame com histórico de operações."""
    import pandas as pd
    conn  = get_conn()
    query = "SELECT * FROM trades"
    if ativo:
        query += f" WHERE ativo='{ativo}'"
    query += " ORDER BY datetime DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


if __name__ == "__main__":
    criar_tabelas()