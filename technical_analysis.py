'''
Computing SMA15, MACD26, Stochastic K14, Stochastic R3, RSI14, WR14
deciding trends and computing trading signals

Ideally each function should be executed sequentially
'''

import numpy as np
import pyti.simple_moving_average as sma
import pyti.moving_average_convergence_divergence as macd
from sklearn.preprocessing import MinMaxScaler


# 15-day simple moving average
# return type: np array
def sma15(df):
    return sma.simple_moving_average(df.close, 15)


# 12-26 moving average convergence divergence
# starting from the 26th entry (index 25)
# return type: np array
def macd26(df):
    return macd.moving_average_convergence_divergence(df.close, 12, 26)


# 14-day stochastic K
# starting from the 14th entry (index 13)
# return type: np array
def k14(df):
    lows = df.low.rolling(window=14).min()
    highs = df.high.rolling(window=14).max()

    return (df.close - lows) / (highs - lows) * 100

# 3 day stochastic R
# starting from the 16th entry (index 15)
# return type: np array
def r3(df):
    if 'k14' not in df.columns:
        k = k14(df)
    else:
        k = df['k14']
    return k.rolling(window=3).mean()

# 14-day relative strength indicator
# starting from the 14th entry (index 13)
# return type: np array
# *********** different from package results ****************
def rsi14(df):
    #df['rsi14'] = rsi.relative_strength_index(df['close'], 14)
    r = np.zeros(len(df))
    diff = df.close.diff()
    for i in range(14, len(df), 1):
        ups = [diff[i - j] if diff[i - j] >= 0 else 0 for j in range(14)]
        downs = [-diff[i - j] if diff[i - j] < 0 else 0 for j in range(14)]
        avg_up = np.mean(ups)
        avg_down = np.mean(downs)
        r[i] = 100 - 100 / (1 + avg_up / avg_down)
    return r

# 14-day williams-R
# starting from the 14th entry (index 13)
# return type: np array
def wr14(df):
    min14 = df.low.rolling(14).min()
    max14 = df.high.rolling(14).max()
    return (max14 - df.close) / (max14 - min14) * 100


# start from the 19th entry (index 18)
# return type: np array
def tr(df):
    def down5(seq):
        for i in range(len(seq) - 1):
            if seq[i + 1] >= seq[i]:
                return False
        return True

    def up5(seq):
        for i in range(len(seq) - 1):
            if seq[i + 1] <= seq[i]:
                return False
        return True

    trend = np.zeros(len(df))
    if 'sma15' not in df.columns:
        s = sma15(df)
    else:
        s = df['sma15']
    for i in range(18, len(df), 1):
        sma5 = s[i - 4:i + 1]
        if down5(sma5):
            if df['close'][i] < s[i]:
                trend[i] = -1
        if up5(sma5):
            if df['close'][i] > s[i]:
                trend[i] = 1

    return trend


# starting from 19th entry
# last 2 values should be stripped away(no min max)
# return type: np array
def signals(df):
    c = df.close
    signal = np.zeros(len(df))
    maxcp2 = [max(max(c[i], c[i+1]), c[i+2]) for i in range(len(df)-2)]
    mincp2 = [min(min(c[i], c[i+1]), c[i+2]) for i in range(len(df)-2)]
    for i in range(18, len(df)-2, 1):
        if df['trend'][i] == 1:
            signal[i] = 0.5 * (c[i] - mincp2[i]) / (maxcp2[i] - mincp2[i]) + 0.5
        if df['trend'][i] == -1:
            signal[i] = 0.5 * (c[i] - mincp2[i]) / (maxcp2[i] - mincp2[i])
    return signal


# should be executed after stripping away the first 25 rows and last 2 rows
# with incomplete information
# return type: np array
def normalize(df):
    data = df[['sma15', 'macd26', 'k14', 'r3', 'rsi14', 'wr14']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# 1 for up, -1 for down
def predicted_trend(df, otr):
    tr = df['tr']
    mean_tr = np.mean(tr)
    t = [1 if otr[i] >= mean_tr else -1 for i in range(len(tr))]

    return t