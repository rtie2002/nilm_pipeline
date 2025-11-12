import pandas as pd
import numpy as np
import os
import glob
import random
import torch
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler, MinMaxScaler


device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_seed(seed: int = 1234):
    """set a fix random seed.

    Args:
        seed (int, optional): random seed. Defaults to 9.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Divide window data into x and y
class NormalDataset(data.Dataset):

    def __init__(self, data):
        tensor = torch.from_numpy(data).float()
        self.x = tensor[:, :-576]
        self.y = tensor[:, -576:]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

def _glob_sorted(path: str, pattern: str):
    return sorted(glob.glob(os.path.join(path, pattern)))


def _resolve_single(path: str, pattern: str) -> str:
    matches = _glob_sorted(path, pattern)
    if not matches:
        raise FileNotFoundError(f"No files found for pattern '{pattern}' under '{path}'")
    return matches[0]


def load_data(path, postfix, appliance, choose=None):
    files = _glob_sorted(path, postfix)
    if type(choose) is int:
        if choose < 0 or choose >= len(files):
            raise IndexError(f"Index {choose} is out of range for {len(files)} files")
        print(f"building: {files[choose]}")
        df = pd.read_csv(files[choose], usecols=['Aggregate', appliance])
        return df
    elif type(choose) is str:
        file = _resolve_single(path, choose + postfix)
        df = pd.read_csv(file, usecols=['Aggregate', appliance])
        return df
    elif type(choose) is list:
        dfs = []
        for file_name in choose:
            if isinstance(file_name, int):
                if file_name < 0 or file_name >= len(files):
                    raise IndexError(f"Index {file_name} is out of range for {len(files)} files")
                print(f"building: {files[file_name]}")
                dfs.append(pd.read_csv(files[file_name], usecols=['Aggregate', appliance]))
            elif isinstance(file_name, str):
                resolved = _resolve_single(path, file_name + postfix)
                dfs.append(pd.read_csv(resolved, usecols=['Aggregate', appliance]))
            else:
                raise TypeError("Entries in 'choose' must be int indices or str patterns")
        return dfs
    elif choose is None:
        random_choose = np.random.randint(0, len(files))
        df = pd.read_csv(files[random_choose], usecols=['Aggregate', appliance])
        return df
    else:
        dfs = []
        for file in files:
            dfs.append(pd.read_csv(file, usecols=['Aggregate', appliance]))
        return dfs, files
    
def choose_data(dataset, appliance):
    # dataset name
    refit_folder = dataset.rstrip('/')
    csv_files = [f for f in os.listdir(refit_folder) if f.endswith(".csv")]
    csv_files_with_column= []

    for csv_file in csv_files:
        file_path = os.path.join(refit_folder, csv_file)
        df = pd.read_csv(file_path)

        # appliance name
        if appliance in df.columns:
            csv_files_with_column.append(csv_file.rstrip(".csv"))
    print(csv_files_with_column)
    return csv_files_with_column

def merge_dataframes(dataframes, max_rows_per_df=None, max_total_rows=None, random_sample=True):
    """
    Merge multiple dataframes into a single dataframe with optional sampling.
    
    Args:
        dataframes: List of pandas DataFrames to merge
        max_rows_per_df: Maximum number of rows to use from each dataframe (None = use all)
        max_total_rows: Maximum total rows in merged dataframe (None = no limit)
        random_sample: If True, randomly sample rows; if False, take first N rows
        
    Returns:
        Merged DataFrame
    """
    if not dataframes:
        raise ValueError("No dataframes provided")
    
    sampled_dfs = []
    for i, df in enumerate(dataframes):
        df_sample = df.copy()
        
        # Sample from each dataframe if specified
        if max_rows_per_df is not None and len(df_sample) > max_rows_per_df:
            if random_sample:
                df_sample = df_sample.sample(n=max_rows_per_df, random_state=42).reset_index(drop=True)
            else:
                df_sample = df_sample.head(max_rows_per_df).reset_index(drop=True)
            print(f"DataFrame {i}: sampled {len(df_sample)} rows from {len(df)} total rows")
        
        sampled_dfs.append(df_sample)
    
    merged_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Apply total row limit if specified
    if max_total_rows is not None and len(merged_df) > max_total_rows:
        if random_sample:
            merged_df = merged_df.sample(n=max_total_rows, random_state=42).reset_index(drop=True)
        else:
            merged_df = merged_df.head(max_total_rows).reset_index(drop=True)
        print(f"Total rows limited to {max_total_rows}")
    
    print(f"Merged {len(dataframes)} dataframes into one with shape: {merged_df.shape}")
    return merged_df

def series_to_supervised(data: pd.DataFrame,
                         n_in: int = 1,
                         rate_in: int = 1,
                         sel_in: list = None,
                         sel_out: list = None,
                         dropnan: bool = True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    orig_cols = df.columns
    cols, names = list(), list()

    # input sequence (t-n, ... t-1) n=n_in
    for i in range(n_in, 0, -rate_in):
        if sel_in is None:
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (orig_cols[j], i)) for j in range(n_vars)]
        else:
            for var in sel_in:
                cols.append(df[var].shift(i))
                names += [('%s(t-%d)' % (var, i))]

    # current time (t) sequence
    for i in range(n_in, 0, -rate_in):
        if sel_out is None:
            cols.append(df)
            names += [('%s(t-%d)' % (orig_cols[j])) for j in range(n_vars)]
        else:
            for var in sel_out:
                cols.append(df[var].shift(i))
                names += [('%s(t-%d)' % (var, i))]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    agg.index = range(len(agg))
    return agg

def construct_dataset(df, appliance, batchsize=64, max_rows=None):

    # 0. Optional: Limit total rows to prevent memory issues
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"Limited input dataframe to {max_rows} rows")
    
    # 1. Split dataset first (to avoid data leakage)
    n = len(df)
    train_ratio = 0.8
    test_ratio = 0.2
    split_idx = int(train_ratio * n)
    train_df = df[:split_idx].copy()  # Use copy to avoid warnings
    test_df = df[split_idx:].copy()
    
    print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")
    
    # 2. Standardization: fit only on training set (critical fix: avoid data leakage)
    scalerx = StandardScaler()
    scalery = StandardScaler()
    
    # Important: fit only on training set
    scaler_x = scalerx.fit(train_df[['Aggregate']])
    scaler_y = scalery.fit(train_df[[appliance]])
    
    # Transform separately
    train_df['Aggregate'] = scaler_x.transform(train_df[['Aggregate']])
    train_df[appliance] = scaler_y.transform(train_df[[appliance]])
    
    # Test set uses training set's scaler to transform (must not refit!)
    test_df['Aggregate'] = scaler_x.transform(test_df[['Aggregate']])
    test_df[appliance] = scaler_y.transform(test_df[[appliance]])

    # 3. Create supervised learning dataset
    print("Creating supervised learning dataset (this may take a while for large datasets)...")
    train_ds = series_to_supervised(train_df,
                                    n_in=576,
                                    rate_in=1,
                                    sel_in=['Aggregate'],
                                    sel_out=[appliance])

    test_ds = series_to_supervised(test_df,
                                n_in=576,
                                rate_in=1,
                                sel_in=['Aggregate'],
                                sel_out=[appliance])

    print(f"Supervised learning - Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    print(f"Generated data shape: {train_ds.shape}")  # Should be (num_samples, 1152)
    
    # Clean up intermediate dataframes to free memory
    del train_df, test_df
    import gc
    gc.collect()

    # 4. Create dataset
    train_ds = NormalDataset(train_ds.values)
    test_ds = NormalDataset(test_ds.values)

    print(f"Final - Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    print(f"Input dimension: {train_ds.x[0].shape}, Output dimension: {train_ds.y[0].shape}")

    # 5. Create DataLoader
    input_dim = train_ds.x[0].shape[-1]

    train_dataloader = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    val_batchsize = max(1, min(batchsize, len(test_ds)))
    test_dataloader = data.DataLoader(test_ds, batch_size=val_batchsize, shuffle=False)
    del test_ds
    torch.cuda.empty_cache()
    return train_dataloader, test_dataloader, scaler_y, input_dim, train_ds, scaler_x
 