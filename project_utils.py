import pandas as pd
import numpy as np
from pathlib import Path


def Waterbirds_CelebA128_dataloader(dir_: str) -> pd.DataFrame:
    """Attribute Data Loader
    :param dir_: csv file path
    """
    data = pd.read_csv(dir_, skiprows=None, sep=' ', index_col=0)
    # Type float needed for criterion in training
    # data = data.astype({i: 'float' for i in data.columns})
    data.index = [int(i.split('.')[0]) for i in data.index.tolist()]

    return data


def features_loader_npz(dir_):
    npz = np.load(dir_)

    def to_csv_norm(data_flag):
        # import pdb; pdb.set_trace()
        data_df = pd.DataFrame(npz[f'{data_flag}_x'], index=npz[f'{data_flag}_idx'])

        data_df = data_df.astype(dtype='float64')
        data_df = data_df.replace([np.inf, -np.inf], 0)
        data_df = data_df.div(np.sqrt(np.square(data_df).sum(axis=1)), axis=0)
        data_df.index = [int(i.split('.')[0]) for i in data_df.index.tolist()]

        return data_df

    data_df_train = to_csv_norm('train')
    data_df_val = to_csv_norm('val')
    data_df_test = to_csv_norm('test')

    # Sanity check
    assert not data_df_train.isnull().values.any()
    assert not (data_df_test.values == 0).all()

    assert not data_df_val.isnull().values.any()
    assert not (data_df_val.values == 0).all()

    assert not data_df_test.isnull().values.any()
    assert not (data_df_test.values == 0).all()

    return data_df_train, data_df_val, data_df_test


def features_loader(dir_: Path):
    data_df_train, data_df_val, data_df_test = features_loader_npz(dir_)

    return data_df_train, data_df_val, data_df_test


def filter_df(data, variable, variable_value, return_index=True):
    """"
    Filter the dataset based on a variable value
    """
    filtered_data = data[data[variable] == variable_value]
    filtered_data_idx = filtered_data.index.tolist()
    if return_index:
        return filtered_data, filtered_data_idx
    return filtered_data
