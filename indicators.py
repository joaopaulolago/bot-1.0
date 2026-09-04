import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Open", "High", "Low", "Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:

            if col in ["High", "Low"]:
                df[col] = df["Close"]


    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9) 
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    std = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["SMA20"] + (std * 2)
    df["BB_lower"] = df["SMA20"] - (std * 2)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["SMA20"]
   
    if "Volume" in df.columns:
        df["Vol_Log"] = np.log1p(df["Volume"])
    else:
        df["Vol_Log"] = 0

    period = 14
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['Close'].shift(1))
    df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    df['plus_dm'] = df['High'].diff()
    df['minus_dm'] = df['Low'].diff().apply(lambda x: -x)
    
    df['plus_dm'] = np.where((df['plus_dm'] > df['minus_dm']) & (df['plus_dm'] > 0), df['plus_dm'], 0)
    df['minus_dm'] = np.where((df['minus_dm'] > df['plus_dm']) & (df['minus_dm'] > 0), df['minus_dm'], 0)

    df['TR_smooth'] = df['TR'].rolling(window=period).sum()
    df['plus_dm_smooth'] = df['plus_dm'].rolling(window=period).sum()
    df['minus_dm_smooth'] = df['minus_dm'].rolling(window=period).sum()

    df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['TR_smooth'].replace(0, 1e-9))
    df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['TR_smooth'].replace(0, 1e-9))

    df['DX'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 1e-9))
    df['ADX'] = df['DX'].rolling(window=period).mean()

    df['ADX'] = df['ADX'].ffill().bfill()
    df['BB_width'] = df['BB_width'].ffill().bfill()

    cols_to_drop = ['tr1', 'tr2', 'tr3', 'TR', 'plus_dm', 'minus_dm', 
                    'TR_smooth', 'plus_dm_smooth', 'minus_dm_smooth', 
                    'plus_di', 'minus_di', 'DX']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df