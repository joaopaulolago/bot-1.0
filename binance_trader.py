import os
import time
import joblib
import numpy as np
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from indicators import add_indicators
from config import ATIVOS

# Configuração Inicial
load_dotenv()
#chaves de acesso 
client = Client(os.getenv("API_KEY"), os.getenv("SECRET_KEY"))

def obter_dados_binance(ticker, intervalo="1h"):
    """Busca as últimas velas direto da Binance para a IA analisar."""
    # Converte o intervalo para o formato da Binance
    interval_map = {"1h": Client.KLINE_INTERVAL_1HOUR, "5m": Client.KLINE_INTERVAL_5MINUTE}
    
    klines = client.get_klines(symbol=ticker, interval=interval_map[intervalo], limit=50)
    df = pd.DataFrame(klines, columns=[
        'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'CloseTime', 'QuoteAssetVol', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore'
    ])
    
    # Converte colunas para numérico
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col])
        
    return df

def executar_estrategia():
    print(f"\n--- Verificação de Mercado: {time.strftime('%H:%M:%S')} ---")
    
    for nome, ticker_yahoo in ATIVOS.items():
        # Ajusta o nome do ticker para o padrão da Binance (ex: BTCUSDT)
        ticker_binance = ticker_yahoo.replace("-USD", "USDT").replace("=F", "")
        
        try:
            # Carrega Cérebro (Modelo) e o Normalizador (Scaler)
            model = load_model(f"models/lstm_model_{nome}_1h.h5")
            scaler = joblib.load(f"models/scaler_{nome}_1h.pkl")
            
            # Pega dados e calcula indicadores
            df = obter_dados_binance(ticker_binance)
            df = add_indicators(df)
            
            # Prepara entrada para a IA (últimas 20 velas)
            features = ["SMA20", "EMA20", "RSI", "MACD", "ADX", "BB_width", "Vol_Log"]
            dados_recentes = df[features].tail(20).values
            dados_normalizados = scaler.transform(dados_recentes)
            entrada_ia = np.array([dados_normalizados])
            
            # Predição (0 a 1)
            confianca = model.predict(entrada_ia, verbose=0)[0][0]
            
            print(f" {nome.upper()}: Confiança de subida em {confianca:.2%}")

            # LÓGICA DE EXECUÇÃO
            if confianca > 0.75:
                print(f" COMPRANDO {nome.upper()}...")
                
            elif confianca < 0.25:
                print(f"📉 VENDENDO/SHORT {nome.upper()}...")


        except Exception as e:
            print(f"Erro ao processar {nome}: {e}")

if __name__ == "__main__":
    while True:
        executar_estrategia()
        time.sleep(3600)