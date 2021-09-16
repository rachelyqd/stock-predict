import pandas as pd
import datetime as dt
import pandas_datareader.data as web
import technical_analysis as ta
import numpy as np
from sklearn.model_selection import TimeSeriesSplit



def get_process_df(tickers, start=dt.datetime(2010, 1, 4), end=dt.datetime(2014, 12, 31)):
    '''
    get data frame and add all indicators and remove unused columns
    return a list of dataframes
    :param tickers: list, list of tickers
    :param start: datetime, start time
    :param end: datetime, end time
    :return: dataframe with closing price and 6 indicators as columns
    '''
    dfs = []
    for ticker in tickers:
        df = web.DataReader(ticker, 'yahoo', start, end)
        df.columns = df.columns.str.lower()
        df['close'] = df['adj close']
        df.drop(['adj close'], axis=1, inplace=True)
        df['sma15'] = ta.sma15(df)
        df['macd26'] = ta.macd26(df)
        df['k14'] = ta.k14(df)
        df['r3'] = ta.r3(df)
        df['rsi14'] = ta.rsi14(df)
        df['wr14'] = ta.wr14(df)
        df['trend'] = ta.tr(df)
        df['tr'] = ta.signals(df)

        df = df[25:]  # strip away entries missing MACD
        df = df[:-2]  # strip away entries missing signal

        df = df.drop(['open', 'high', 'low', 'volume'], axis=1)

        dfs.append(df)

    return dfs


def get_train_test(dfs, original=True):
    '''
    splitting dataframe into train and test set
    :param dfs: dataframes of different stocks/indices
    :param original: boolean, True if using original data (SP500 and BSE), False if not
    :return: 6 lists, containing train, test set of X and y, and train, test dataframes
    '''
    X_trains, y_trains = [], []
    X_tests, y_tests = [], []
    dfs_train, dfs_test = [], []

    for df in dfs:
        t, s = df['trend'].values, df['tr'].values
        t = t.reshape(len(t), 1)  # used as feature
        s = s.reshape(len(s), 1)  # used as target

        if original:
            df_train, df_test = df[:1000], df[1000:]
            t_train, t_test = t[:1000], t[1000:]
            y_train, y_test = s[:1000], s[1000:]
        else:
            total_length = len(df)
            train_indices = int(total_length * 0.8)
            df_train, df_test = df.iloc[:train_indices], df.iloc[train_indices:]
            t_train, t_test = t[:train_indices], t[train_indices:]
            y_train, y_test = s[:train_indices], s[train_indices:]

        train_scaled = ta.normalize(df_train)
        test_scaled = ta.normalize(df_test)
        X_train = np.hstack((train_scaled, t_train))
        X_test = np.hstack((test_scaled, t_test))
        X_trains.append(X_train)
        y_trains.append(y_train.reshape(-1, ))
        X_tests.append(X_test)
        y_tests.append(y_test)
        dfs_train.append(df_train)
        dfs_test.append(df_test)

    return X_trains, X_tests, y_trains, y_tests, dfs_train, dfs_test

def get_train_val(dfs_trains, X_trains, y_trains):
    '''
    further splitting training set into training and validation set used for
    parameter tuning and blending
    :param X_trains: list, list of training features
    :param y_trains: list, list of training targets
    :return: list of training, validation features and targets
    '''
    X_train_subs = []
    X_vals = []
    y_train_subs = []
    y_vals = []
    dfs_train_sub = []
    dfs_val = []
    tss = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    for train_ind, val_ind in tss.split(X_trains[0]):
        tmp_a, tmp_b, tmp_c, tmp_d, tmp_e, tmp_f = [], [], [], [], [], []  # for storing 5 subsets
        for i in range(len(X_trains)):
            X_train_sub, X_val = X_trains[i][train_ind], X_trains[i][val_ind]
            y_train_sub, y_val = y_trains[i][train_ind], y_trains[i][val_ind]
            df_train_sub, df_val = dfs_trains[i].iloc[train_ind], dfs_trains[i].iloc[val_ind]
            tmp_a.append(X_train_sub)
            tmp_b.append(X_val)
            tmp_c.append(y_train_sub)
            tmp_d.append(y_val)
            tmp_e.append(df_train_sub)
            tmp_f.append(df_val)
        X_train_subs.append(tmp_a)
        X_vals.append(tmp_b)
        y_train_subs.append(tmp_c)
        y_vals.append(tmp_d)
        dfs_train_sub.append(tmp_e)
        dfs_val.append(tmp_f)
    return X_train_subs, X_vals, y_train_subs, y_vals, dfs_train_sub, dfs_val

