import pandas as pd

import datetime
import numpy as np


def store_sales_dtypes(stores_sales_df: pd.DataFrame) -> pd.DataFrame:
    stores_sales_df = stores_sales_df.copy(deep=True)
    stores_sales_df['Date'] = pd.to_datetime(stores_sales_df['Date'])
    sales_category_cols = ["Promo", "Open", "StateHoliday", "SchoolHoliday", "DayOfWeek"]
    for col_name in sales_category_cols:
        stores_sales_df[col_name] = stores_sales_df[col_name].astype("category")

    return stores_sales_df


def store_data_dtypes(stores_data_df: pd.DataFrame) -> pd.DataFrame:
    stores_data_df = stores_data_df.copy(deep=True)
    stores_category_cols = ["StoreType", "Assortment", "Promo2"]
    for col_name in stores_category_cols:
        stores_data_df[col_name] = stores_data_df[col_name].astype("category")

    stores_int_cols = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
    stores_data_df['CompetitionOpenSinceYear'].fillna(2013, inplace=True)
    stores_data_df['CompetitionOpenSinceMonth'].fillna(1, inplace=True)
    for col_name in stores_int_cols:
        stores_data_df[col_name] = stores_data_df[col_name].astype(int)
    return stores_data_df


def store_sales_prep(stores_sales_df: pd.DataFrame) -> pd.DataFrame:
    # make deep copy , change datatypes
    stores_sales_df = store_sales_dtypes(stores_sales_df)

    # ----------------Date column-------------------
    stores_sales_df['year'] = stores_sales_df.Date.dt.year
    stores_sales_df['month'] = stores_sales_df.Date.dt.month
    stores_sales_df['day'] = stores_sales_df.Date.dt.day

    # -----------School_holiday column------------
    # change from symbols to name
    # (a = public holiday, b = Easter holiday, c = Christmas, 0 = None)

    state_holiday_map = {"a": "public", "b": "easter", "c": "christmas"}
    stores_sales_df['StateHoliday'] = stores_sales_df['StateHoliday'].map(state_holiday_map)

    return stores_sales_df


def store_data_prep(store_data_df: pd.DataFrame) -> pd.DataFrame:
    # make deep copy , change datatypes
    store_data_df = store_data_dtypes(store_data_df)
    # ------------------- Competition Distance Col ------------------------
    store_data_df['CompetitionDistance'] = store_data_df['CompetitionDistance'].fillna(
        store_data_df['CompetitionDistance'].max())
    store_data_df['CompetitionDistance'] = store_data_df['CompetitionDistance'] / 1000.0

    # ----------Competition OpenSinceYear/Month to CompetitionOpen date filed-----------

    # create string date filed of yyyy-m
    store_data_df['CompetitionOpenDate'] = store_data_df['CompetitionOpenSinceYear'].astype('str') + '-' + \
                                           store_data_df[
                                               'CompetitionOpenSinceMonth'].astype('str')
    # parse datetime
    store_data_df['CompetitionOpenDate'] = pd.to_datetime(store_data_df['CompetitionOpenDate'])

    # drop OpenSinceYear/Month
    store_data_df.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1, inplace=True)

    # ---------- Promo2 SinceYear/Month to Promo2 Since date filed -----------

    promo2_since_list = []
    for index, row in store_data_df.iterrows():
        store_promo2_since = {"Store": row['Store']}
        if row['Promo2'] == 1:
            date_str = f"{int(row['Promo2SinceYear'])}-{int(row['Promo2SinceWeek'])}-1"
            store_promo2_since["Promo2Since"] = datetime.datetime.strptime(date_str, "%Y-%W-%w")
        else:
            store_promo2_since["Promo2Since"] = None
        promo2_since_list.append(store_promo2_since)

    promo2Since_col = pd.DataFrame(promo2_since_list)
    store_data_df = store_data_df.merge(promo2Since_col, on="Store")
    # store_data_df["Promo2Since"]=store_data_df["Promo2Since"]
    store_data_df.drop(['Promo2SinceWeek', 'Promo2SinceYear'], axis=1, inplace=True)

    return store_data_df


def merge_store_sales(sales_data_df: pd.DataFrame, store_data_df: pd.DataFrame) -> pd.DataFrame:
    merged_data = pd.merge(sales_data_df, store_data_df, on='Store')

    # -------------------- CompetitionOpenSince col --------------------
    merged_data['CompetitionOpenSince'] = (merged_data.CompetitionOpenDate.dt.year - merged_data.Date.dt.year) * 12 + (
            merged_data.CompetitionOpenDate.dt.month - merged_data.Date.dt.month)

    # set negative values which are the rows that the competitor hasn't opened yet to zero
    merged_data['CompetitionOpenSince'] = merged_data['CompetitionOpenSince'].apply(
        lambda months: months if months > 0 else 0)
    # log transform
    merged_data['CompetitionOpenSince'] = merged_data['CompetitionOpenSince'].apply(
        lambda months: np.log(months) if months > 1 else 0)

    # -------------------- isPromoMonth ----------------------
    def is_promo2_month(row):

        months_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
        if row['Date'] >= row['Promo2Since']:
            promo2Months = row['PromoInterval'].split(',')
            if months_name[row['Date'].month - 1] in promo2Months:
                return True
        return False

    merged_data['isPromoMonth'] = merged_data.apply(is_promo2_month, axis=1)

    # ------------------ Promo2Since col ------------------------
    merged_data['Promo2Since'] = (merged_data.Date.dt.year - merged_data.Promo2Since.dt.year) * 12 + (
            merged_data.Date.dt.month - merged_data.Promo2Since.dt.month)
    merged_data['Promo2Since'] = merged_data.Promo2Since * merged_data.Promo2.astype(int)
    merged_data['Promo2Since'] = merged_data['Promo2Since'].apply(lambda months: months if months > 0 else 0)
    # log transform
    merged_data['Promo2Since'] = merged_data['Promo2Since'].apply(
        lambda months: np.log(months) if months > 1 else 0)

    cols = ['year', 'Date', 'PromoInterval', 'CompetitionOpenDate', 'Customers']
    merged_data.drop(cols, axis=1, inplace=True)
    return merged_data


def hot_encoding(merged_data: pd.DataFrame) -> pd.DataFrame:
    merged_data = merged_data.copy(deep=True)
    encoding_cols = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment']
    merged_data = pd.get_dummies(merged_data, columns=encoding_cols)
    for col in merged_data.columns:
        merged_data[col] = merged_data[col].astype(float
                                                   )

    return merged_data


def drop_closed_days(sales_df: pd.DataFrame) -> pd.DataFrame:
    sales_df = sales_df.copy()
    sales_df = sales_df[sales_df.Open == 1]
    sales_df.drop('Open', axis=1, inplace=True)
    return sales_df
