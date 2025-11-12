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

def construct_dataset(df, appliance, batchsize=64):
    """
    Construct dataset with proper train/test split and standardization.
    CRITICAL: Split first, then fit scalers only on training data to avoid data leakage.
    """
    # Step 1: Split dataset FIRST (before any standardization)
    n = len(df)
    train_ratio = 0.8
    split_idx = int(train_ratio * n)
    train_df = df[:split_idx].copy()  # Use copy to avoid SettingWithCopyWarning
    test_df = df[split_idx:].copy()
    
    print(f"Dataset split - Training set size: {len(train_df)}, Test set size: {len(test_df)}")
    
    # Step 2: Standardization - Fit scalers ONLY on training data
    scalerx = StandardScaler()
    scalery = StandardScaler()
    
    # CRITICAL: Only fit on training data to prevent data leakage
    scaler_x = scalerx.fit(train_df[['Aggregate']])
    scaler_y = scalery.fit(train_df[[appliance]])
    
    # Transform training data
    train_df['Aggregate'] = scaler_x.transform(train_df[['Aggregate']])
    train_df[appliance] = scaler_y.transform(train_df[[appliance]])
    
    # Transform test data using training scalers (DO NOT refit!)
    test_df['Aggregate'] = scaler_x.transform(test_df[['Aggregate']])
    test_df[appliance] = scaler_y.transform(test_df[[appliance]])
    
    # Step 3: Create supervised learning datasets
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
    print(f"Generated data shape: {train_ds.shape}")  # Should be (samples, 1152) = (samples, 576*2)
    
    # Step 4: Create PyTorch datasets
    train_ds = NormalDataset(train_ds.values)
    test_ds = NormalDataset(test_ds.values)
    
    print(f"Final dataset - Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    print(f"Input dimension: {train_ds.x[0].shape}, Output dimension: {train_ds.y[0].shape}")
    
    # Step 5: Create DataLoaders
    input_dim = train_ds.x[0].shape[-1]
    
    train_dataloader = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    val_batchsize = max(1, min(batchsize, len(test_ds)))
    test_dataloader = data.DataLoader(test_ds, batch_size=val_batchsize, shuffle=False)
    
    del test_ds
    torch.cuda.empty_cache()
    
    return train_dataloader, test_dataloader, scaler_y, input_dim, train_ds, scaler_x


def verify_batch_shapes(train_dataloader, test_dataloader):
    """
    Debug function to verify batch shapes are correct.
    Expected: Batch input shape: torch.Size([batch_size, 576])
              Batch output shape: torch.Size([batch_size, 576])
    """
    print("\n=== Verifying Batch Shapes ===")
    
    # Check training batches
    for x, y in train_dataloader:
        print(f"Training batch - Input shape: {x.shape}, Output shape: {y.shape}")
        print(f"Expected: Input torch.Size([{x.shape[0]}, 576]), Output torch.Size([{y.shape[0]}, 576])")
        break
    
    # Check test batches
    for x, y in test_dataloader:
        print(f"Test batch - Input shape: {x.shape}, Output shape: {y.shape}")
        print(f"Expected: Input torch.Size([{x.shape[0]}, 576]), Output torch.Size([{y.shape[0]}, 576])")
        break
    
    print("=== Batch Shape Verification Complete ===\n")
 