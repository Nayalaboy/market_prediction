import pandas as pd

def add_features(df: pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    
    df["return_1d"] = df["price"].pct_change()
    df["return_5d"] = df["price"].pct_change(5)
    df["return_10d"] = df["price"].pct_change(10)

    df["ma_5"] = df["price"].rolling(window=5).mean()
    df["ma_10"] = df["price"].rolling(window=10).mean()

    df["ma_gap_5"] = df["price"] / df["ma_5"] - 1
    df["ma_gap_10"] = df["price"] / df["ma_10"] - 1

    df["volatility_10"] = df["return_1d"].rolling(window=10).std()
    df["momentum_5"] = df["price"] - df["price"].shift(5)

    # === VOLUME FEATURES (NEW) ===

    # Volume change
    df["volume_change"] = df["Volume"].pct_change()

    # Volume moving average
    df["volume_ma_10"] = df["Volume"].rolling(10).mean()

    # Volume spike (very useful)
    df["volume_spike"] = df["Volume"] / df["volume_ma_10"]

    # Price * Volume interaction (advanced signal)
    df["price_volume"] = df["price"] * df["Volume"]
    df["target"] = df["return_1d"].shift(-1)  
    df = df.dropna().reset_index(drop=True)

    return df