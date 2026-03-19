print("Loading data...")
from data_loader import load_data


def main():
    ticker = "SPY"
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    df = load_data(ticker=ticker, start_date=start_date, end_date=end_date)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nShape:")
    print(df.shape)


if __name__ == "__main__":
    main()