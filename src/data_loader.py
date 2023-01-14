import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset, Dataset
import pandas as pd
import random


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
        target = np.expand_dims(self.sales_data_target[idx], axis=0)
        return torch.tensor(self.sales_data_input[idx], dtype=torch.float), torch.tensor(target,
                                                                                         dtype=torch.float)

    def __len__(self):
        return self.size


class LSTMSalesDataset(IterableDataset):
    lstm_sales_cols = ['Sales', 'month', 'day', 'StateHoliday_christmas', 'StateHoliday_easter', 'StateHoliday_public',
                       'SchoolHoliday', 'Promo', 'CompetitionOpenSince'] + [f"DayOfWeek_{i}" for i in range(1, 8)]

    lstm_store_cols = ['CompetitionDistance'] + [f"StoreType_{i}" for i in ['a', 'b', 'c', 'd']] + [f"Assortment_{i}"
                                                                                                    for i in
                                                                                                    ['a', 'b', 'c']]
    lstm_sales_cols = {col_name: idx for idx, col_name in enumerate(lstm_sales_cols)}
    lstm_store_cols = {col_name: idx for idx, col_name in enumerate(lstm_store_cols)}

    def __init__(self, store_sales, store_data, seq_length=30):
        """

        :param merged_data_df:
        :param lstm_sales_cols: col name to index dictionary for lstm sales data
        :param lstm_store_cols: col name to index dictionary for store data for nn of the lstm
        """
        self.stores_sales = []

        self.seq_length = seq_length

        self.stores_data_dict = self.get_sales_store_data(store_sales, store_data)

        self.stores_indices = list(self.stores_data_dict.keys())

        self.no_lstm_cols = len(LSTMSalesDataset.lstm_sales_cols)
        self.no_store_data_cols = len(LSTMSalesDataset.lstm_store_cols)
        self.size = len(store_sales)-seq_length

    def __iter__(self):
        random.shuffle(self.stores_indices)

        for store_idx in self.stores_indices:
            store_data = self.stores_data_dict[store_idx]['data']
            store_sales = self.stores_data_dict[store_idx]['sales']

            no_sequences = len(store_sales) - self.seq_length

            in_idx, out_idx = 0, self.seq_length
            for i in range(no_sequences):
                lstm_in = torch.tensor(store_sales[i:i + self.seq_length], dtype=torch.float)
                nn_in = torch.tensor(store_data.flatten(), dtype=torch.float)

                in_idx += self.seq_length

                # get the sales column only
                out = store_sales[out_idx:out_idx + self.seq_length, self.lstm_sales_cols['Sales']]
                # out = np.expand_dims(out, axis=0).reshape((self.seq_length, 1))
                out = torch.tensor(out, dtype=torch.float)
                out_idx += self.seq_length

                yield lstm_in, nn_in, out

    def get_sales_store_data(self, sales_data, stores_data):

        # get store id
        stores_idx = stores_data.Store
        sales_cols_list = [None] * len(self.lstm_sales_cols)
        for col_name, col_idx in self.lstm_sales_cols.items():
            sales_cols_list[col_idx] = col_name

        store_cols_list = [None] * len(self.lstm_store_cols)

        for col_name, col_idx in self.lstm_store_cols.items():
            store_cols_list[col_idx] = col_name

        stores_data_dict = {}
        for store_idx in stores_idx:
            stores_data_dict[store_idx] = {}
            store_sales = sales_data[sales_data.Store == store_idx]
            store_data = stores_data[stores_data.Store == store_idx]

            stores_data_dict[store_idx]['sales'] = store_sales[sales_cols_list].to_numpy()
            stores_data_dict[store_idx]['data'] = store_data[store_cols_list].to_numpy()
        return stores_data_dict

    def __len__(self):
        return self.size
