import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class NNSalesDataset(Dataset):
    def __init__(self, sales_df: pd.DataFrame):
        sales_data = sales_df
        # drop Store id column
        sales_data = sales_data.drop('Store', axis=1)
        self.size = len(sales_data)

        # get numpy target column
        self.sales_data_target = sales_data['Sales'].to_numpy()

        # get numpy input columns
        sales_data.drop("Sales", axis=1, inplace=True)

        self.no_cols = len(sales_data.columns)
        self.sales_data_input = sales_data.to_numpy()

    def __getitem__(self, idx):
        target=np.expand_dims(self.sales_data_target[idx],axis=0)
        return torch.tensor(self.sales_data_input[idx], dtype=torch.float), torch.tensor(target,
                                                                                         dtype=torch.float)

    def __len__(self):
        return self.size
