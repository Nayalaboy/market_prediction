print("Loading data...")
from data_loader import load_data
import dataset
from features import add_features
from dataset import TimeSeriesDataset
from dataloader import create_dataloader
from split import time_split
from model import LSTMModel

def main():
    ticker = "SPY"
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    sequence_length = 20
    batch_size = 32

    print("Loading data...")
    df = load_data(ticker=ticker, start_date=start_date, end_date=end_date)

    print("\nAdding features...")
    df = add_features(df)

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nSelected columns:")
    cols_to_check = [
        "Date",
    
        "return_1d",
    
        "target",
    ]
    print(df[cols_to_check].head(7))

    print("\nShape:")
    print(df.shape)
    train_df, val_df, test_df = time_split(df)
    print(f"split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    train_df = TimeSeriesDataset(train_df, seq_len=sequence_length)
    val_df = TimeSeriesDataset(val_df, seq_len=sequence_length)
    test_df = TimeSeriesDataset(test_df, seq_len=sequence_length)

    print("\nDataset lengths:")
    print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_loader, val_loader, test_loader = create_dataloader(train_df, val_df, test_df, batch_size=batch_size)
    x_batch, y_batch = next(iter(train_loader))
    print("\nBatch shapes:")
    print(f"x_batch: {x_batch.shape}, y_batch: {y_batch.shape}")

    model = LSTMModel(input_size=x_batch.shape[2])
    x_batch, y_batch = next(iter(train_loader))
    print("\nBatch shapes:")
    print(f"x_batch: {x_batch.shape}")
    print(f"y_batch: {y_batch.shape}")

    prediction = model(x_batch)
    print("\nPrediction shape:")
    print(prediction.shape)
    print(f"Prediction sample:\n{prediction[:5].detach().numpy()}")

if __name__ == "__main__":
    main()