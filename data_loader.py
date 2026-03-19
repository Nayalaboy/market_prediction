import yfinance as yf
import pandas as pd


def load_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading data for {ticker}...")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} from {start_date} to {end_date}")

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    df = df[["Date", price_col, "Volume"]].copy()
    df = df.rename(columns={price_col: "price"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df