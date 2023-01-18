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


"""
 #   Column                  Non-Null Count   Dtype  
---  ------                  --------------   -----  
 0   Store                   675513 non-null  float64
 1   Sales                   675513 non-null  float64
 2   Promo                   675513 non-null  float64
 3   SchoolHoliday           675513 non-null  float64
 4   month                   675513 non-null  float64
 5   day                     675513 non-null  float64
 6   CompetitionDistance     675513 non-null  float64
 7   Promo2                  675513 non-null  float64
 8   Promo2Since             675513 non-null  float64
 9   CompetitionOpenSince    675513 non-null  float64
 10  isPromoMonth            675513 non-null  float64
 11  DayOfWeek_1             675513 non-null  float64
 12  DayOfWeek_2             675513 non-null  float64
 13  DayOfWeek_3             675513 non-null  float64
 14  DayOfWeek_4             675513 non-null  float64
 15  DayOfWeek_5             675513 non-null  float64
 16  DayOfWeek_6             675513 non-null  float64
 17  DayOfWeek_7             675513 non-null  float64
 18  StateHoliday_christmas  675513 non-null  float64
 19  StateHoliday_easter     675513 non-null  float64
 20  StateHoliday_public     675513 non-null  float64
 21  StoreType_a             675513 non-null  float64
 22  StoreType_b             675513 non-null  float64
 23  StoreType_c             675513 non-null  float64
 24  StoreType_d             675513 non-null  float64
 25  Assortment_a            675513 non-null  float64
 26  Assortment_b            675513 non-null  float64
 27  Assortment_c            675513 non-null  float64
dtypes: float64(28)
memory usage: 144.3 MB
"""


class LSTMSalesDataset(IterableDataset):
    state_holiday_cols = ['StateHoliday_christmas', 'StateHoliday_easter', 'StateHoliday_public']
    weekdays_cols = [f"DayOfWeek_{i}" for i in range(1, 8)]
    store_type_cols = [f"StoreType_{i}" for i in ['a', 'b', 'c', 'd']]
    assortment_cols = [f"Assortment_{i}" for i in ['a', 'b', 'c']]

    lstm_sales_cols = ['Sales', 'Promo', 'SchoolHoliday', 'month', 'day'] + state_holiday_cols + weekdays_cols

    nn_sales_cols = ['Promo', 'SchoolHoliday', 'month', 'day', 'CompetitionDistance', 'Promo2', 'Promo2Since',
                     'CompetitionOpenSince',
                     'isPromoMonth'] + weekdays_cols + state_holiday_cols + store_type_cols + assortment_cols

    lstm_sales_cols = {col_name: idx for idx, col_name in enumerate(lstm_sales_cols)}
    nn_sales_cols = {col_name: idx for idx, col_name in enumerate(nn_sales_cols)}

    def __init__(self, merged_sales_dataset: pd.DataFrame, seq_length=30):
        """
        future
        :param merged_data_df:
        :param lstm_sales_cols: col name to index dictionary for lstm sales data
        :param lstm_store_cols: col name to index dictionary for store data for nn of the lstm
        """
        self.stores_sales = []

        self.seq_length = seq_length

        self.stores_data_dict = self.get_sales_store_data(merged_sales_dataset)

        self.stores_indices = list(self.stores_data_dict.keys())

        self.no_lstm_cols = len(LSTMSalesDataset.lstm_sales_cols)
        self.no_nn_cols = len(LSTMSalesDataset.nn_sales_cols)

        self.size = self.get_size()
        self.current_store = 0

        # sequences

        # the last valid window is the window-size + 1 + the len of future output
        # ex size=8 and seq_len=2 , future output=3 last valid idx for window is 3
        # 0 1 2 3 4 5 6 7
        # _ _ _ _ _ _ _ _
        #       |_| |___|

    def __iter__(self):
        random.shuffle(self.stores_indices)
        lstm_sales_cols = list(self.lstm_sales_cols.values())
        nn_sales_cols = list(self.nn_sales_cols.values())
        for store_idx in self.stores_indices:
            store_sales = self.stores_data_dict[store_idx]
            self.current_store = store_idx

            # out starts from seq_len +1 but we are zero based so seq_len +1 -1 = seq_len
            out_idx = self.seq_length
            no_sequences = (len(store_sales) - self.seq_length)
            for i in range(no_sequences):
                # from window_idx to seq_len (slicing end is exclusive)
                # past rows
                lstm_in = torch.tensor(store_sales[i:i + self.seq_length, lstm_sales_cols], dtype=torch.float)

                # future row
                # get the sales column only
                out = store_sales[out_idx, self.lstm_sales_cols['Sales']]

                out = np.expand_dims(out, axis=0)
                out = torch.tensor(out, dtype=torch.float)

                nn_in = torch.tensor(store_sales[out_idx, nn_sales_cols], dtype=torch.float)
                out_idx += 1

                yield lstm_in, nn_in, out

    def get_sales_store_data(self, merged_sales_dataset):

        # get store id
        stores_idx = merged_sales_dataset.Store.unique()
        stores_data_dict = {}
        for store_idx in stores_idx:
            store_sales = merged_sales_dataset[
                merged_sales_dataset.Store == store_idx]
            store_sales=store_sales.loc[:,merged_sales_dataset.columns != 'Store']

            stores_data_dict[store_idx] = store_sales.to_numpy()

        return stores_data_dict

    def __len__(self):
        return self.size

    def get_size(self):
        sequences_sum = 0
        for store_data in self.stores_data_dict.values():
            sequences_sum += (len(store_data) - self.seq_length)
        return sequences_sum
