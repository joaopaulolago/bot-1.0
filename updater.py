"""
updater.py — Atualiza os preços no banco automaticamente.

Uso:
    python updater.py               # roda loop contínuo
    python updater.py --once        # atualiza uma vez e sai (bom para testar)
    python updater.py --intervalo 5m

Como funciona:
    - A cada X minutos baixa apenas as últimas N candles do yfinance
    - Insere no banco (duplicatas são ignoradas automaticamente)
    - Loga tudo em logs/updater.log
"""

import argparse
import logging
import os
import time
from datetime import datetime

import pandas as pd
import schedule
import yfinance as yf

from database import criar_tabelas, inserir_prices

from config import ATIVOS


FETCH_CONFIG = {
    "5m":  {"period": "5d",   "intervalo_min": 5},
    "15m": {"period": "7d",   "intervalo_min": 15},
    "1h":  {"period": "30d",  "intervalo_min": 60},
    "1d":  {"period": "90d",  "intervalo_min": 1440},
}

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/updater.log", encoding="utf-8"),
        logging.StreamHandler() 
    ]
)
log = logging.getLogger(__name__)



def atualizar_ativo(nome: str, ticker: str, intervalo: str):
    cfg = FETCH_CONFIG.get(intervalo)
    if not cfg:
        log.warning(f"Intervalo '{intervalo}' não configurado. Use: {list(FETCH_CONFIG)}")
        return

    try:
        df = yf.download(ticker, period=cfg["period"], interval=intervalo,
                         auto_adjust=True, progress=False)

        if df.empty:
            log.warning(f"{nome}: sem dados retornados")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        col_data = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={col_data: "Date"})

        n = inserir_prices(df, nome, intervalo)
        log.info(f" {nome:<12} +{n:>4} novas candles  (total baixado: {len(df)})")

    except Exception as e:
        log.error(f" Erro em {nome}: {e}")


def ciclo_completo(intervalo: str):
    log.info(f"{'─'*50}")
    log.info(f" Iniciando ciclo de atualização  [{datetime.now():%H:%M:%S}]")
    for nome, ticker in ATIVOS.items():
        atualizar_ativo(nome, ticker, intervalo)
    log.info(" Ciclo concluído")



def calcular_intervalo_minutos(intervalo: str) -> int:
    """Converte string de intervalo para minutos."""
    cfg = FETCH_CONFIG.get(intervalo)
    return cfg["intervalo_min"] if cfg else 60


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervalo", default="1h",
                        help="Intervalo das candles: 5m, 15m, 1h, 1d")
    parser.add_argument("--once", action="store_true",
                        help="Atualiza uma vez e sai")
    args = parser.parse_args()

  
    criar_tabelas()

    if args.once:
        ciclo_completo(args.intervalo)
    else:
        
        intervalo_min = calcular_intervalo_minutos(args.intervalo)

        log.info(f"🚀 Updater iniciado | intervalo={args.intervalo} | "
                 f"atualização a cada {intervalo_min} minutos")

    
        ciclo_completo(args.intervalo)

        schedule.every(intervalo_min).minutes.do(ciclo_completo, args.intervalo)

        log.info("⏳ Aguardando próxima atualização... (Ctrl+C para parar)")
        try:
            while True:
                schedule.run_pending()
                time.sleep(10)
        except KeyboardInterrupt:
            log.info("🛑 Updater encerrado pelo usuário")