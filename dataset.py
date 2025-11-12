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

    # Standardization
    scalerx = StandardScaler()
    scalery = StandardScaler()
    scaler_x = scalerx.fit(df[['Aggregate']])
    scaler_y = scalery.fit(df[[appliance]])

    df['Aggregate'] = scaler_x.transform(df[['Aggregate']])
    df[appliance] = scaler_y.transform(df[[appliance]])

    #Split the dataset
    n = len(df)
    train_ratio = 0.8
    test_ratio = 0.2
    split_idx = int(train_ratio * n)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

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

    train_ds = NormalDataset(train_ds.values)
    test_ds = NormalDataset(test_ds.values)

    input_dim = train_ds.x[0].shape[-1]

    train_dataloader = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    test_dataloader = data.DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    del test_ds
    torch.cuda.empty_cache()
    return train_dataloader, test_dataloader, scaler_y, input_dim, train_ds, scaler_x
 