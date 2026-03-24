def time_split(df, train_size=0.7, val_size=0.15):
    n = len(df)
    train_end = int(n*train_size)
    val_end  = int(n*(train_size + val_size))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()
    return train_df, val_df, test_df