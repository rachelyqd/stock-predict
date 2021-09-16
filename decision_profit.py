import numpy as np
import technical_analysis as ta

import numpy as np
import technical_analysis as ta
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def trading_decision(trend):
    '''
    determine position (buy/sell/hold)
    rule: next day's trend is up and current position is not buy --> buy (1)
          next day's trend is down and current position is buy & hold --> sell (-1)
          else --> hold
    assume no shorting
    :param trend from dataframe column 'trend' (np.array, shape(n,))
    :return: trading decision (np.array, shape(n,))
    '''
    decisions = np.zeros(len(trend))
    buy = False
    sell = True
    for i in range(1, len(decisions) - 1, 1):
        if trend[i + 1] == 1 and not buy:
            decisions[i] = 1
            buy = True
            sell = False
        elif trend[i + 1] == -1 and not sell:
            decisions[i] = -1
            sell = True
            buy = False

    return decisions


# calculate final profit
def profit_calculation(df, actual=False):
    '''
    calculating profit
    formula: (sell price - buy price) / buy price * 100%
    :param df: dataframe
    :param  actual: boolean, True if using predicted trend, False if using actual trend
    :return: profit (float)
    '''
    if actual:
        positions = df['actual position']
    else:
        positions = df['predicted position']

    transac_idx = np.nonzero([positions])[1]
    profit = 0
    for i in range(len(transac_idx) // 2):
        p_b = df['close'][transac_idx[2 * i]]
        p_s = df['close'][transac_idx[2 * i + 1]]
        profit += (p_s - p_b) / p_b * 100
    return profit


# calculate predicted positions based on output signal
# return both actual and predicted positions
def get_positions(models, dfs, Xs, ys=None, training=True):
    '''
    calculate predicted positions based on output signal
    return both actual and predicted positions
    :param models: list, list of models
    :param dfs: list, list of dataframes
    :param Xs: list
    :param ys: list
    :param training: boolean, True if is training set, False if is test set
    :return: 2 lists, containing actual and predicted positions of each ticker
    '''
    position_actual, position_pred = [], []

    for i in range(len(models)):
        if training:
            models[i].fit(Xs[i], ys[i])
        otr_pred = models[i].predict(Xs[i])
        trend_pred = ta.predicted_trend(dfs[i], otr_pred)
        dfs[i]['trend_pred'] = trend_pred
        pa = trading_decision(dfs[i]['trend'].values)
        pp = trading_decision(dfs[i]['trend_pred'].values)
        position_actual.append(pa)
        position_pred.append(pp)

    return position_actual, position_pred


def get_positions_blender(dfs, otr_preds):
    '''
    return positions predicted by stacking and blending models
    :param dfs: list of dataframes
    :param otr_preds: predicted trend signals
    :return: list of positions
    '''
    position_actual, position_pred = [], []

    for i in range(len(dfs)):
        trend_pred = ta.predicted_trend(dfs[i], otr_preds[i])
        dfs[i]['trend_pred'] = trend_pred
        pa = trading_decision(dfs[i]['trend'].values)
        pp = trading_decision(dfs[i]['trend_pred'].values)
        position_actual.append(pa)
        position_pred.append(pp)

    return position_actual, position_pred


# print results of number of positions and add positions to dataframes
def position_summary(names, position_actual, position_predicted, dfs, printing=True, training=True):
    '''
    add positions to dataframes
    print summary if printing set to True
    :param names: list, name of stocks/indices
    :param position_actual: list, list of actual positions
    :param position_predicted: list, list of predicted positions
    :param dfs: dataframes of the stocks
    :param training: boolean, True if is training set, False if is test set
    :param printing: boolean, True if printing summary
    :return: None
    '''
    num_a = []
    num_p = []

    for pa in position_actual:
        num_a.append(len(pa[~(pa == 0)]))
    for pd in position_predicted:
        num_p.append(len(pd[~(pd == 0)]))

    if printing:
        if training:
            print("Training set")
        else:
            print("Test set")

    for i in range(len(names)):
        dfs[i]['actual position'] = position_actual[i]
        dfs[i]['predicted position'] = position_predicted[i]
        if printing:
            print(names[i])
            print("number of trading positions taken with actual trend:", num_a[i])
            print("number of trading positions taken with predicted trend:", num_p[i])


def profit_summary(names, dfs, training=True, printing=True):
    '''
    print result of summary
    :param names: list, list of stock names
    :param dfs: list, list of dataframes
    :param training: boolean
    :param printing: boolean
    :return: lists, list of actual profits and predicted profits
    '''
    pa = []
    pd = []
    for df in dfs:
        pa.append(profit_calculation(df, actual=True))
        pd.append(profit_calculation(df))

    if printing:
        if training:
            print("Training set")
        else:
            print("Test set")

        for i in range(len(names)):
            print(names[i])
            print("profit with actual trend:", pa[i])
            print("profit with predicted trend:", pd[i])
    return pa, pd


def cv_mse(model, i, X_train_subs, y_train_subs, X_vals, y_vals):
    '''
    find the best parameters using MSE metric (parameters that yield lowest MSE for each model)
    :param model: single model in list
    :param i: ith security
    :param X_train_subs: list, training features
    :param y_train_subs: list, training target
    :param X_vals: list, validation features
    :param y_vals: list, validation targets
    :return:
    '''
    score = 0
    for j in range(len(X_train_subs[i])):
        model.fit(X_train_subs[j][i], y_train_subs[j][i])
        y_pred = model.predict(X_vals[j][i])
        score += np.sqrt(mean_squared_error(y_pred, y_vals[j][i]))
    return score


def cv_profit(model, i, dfs_train_sub, dfs_val, X_train_subs, y_train_subs, X_vals, y_vals):
    '''
    find the best parameters using profit metric (parameters that yield highest profit for each model)
    :param model: single model in list
    :param i: ith security
    :param dfs_train_sub: dataframes of training set
    :param dfs_val: list, dataframes of validation set
    :param X_train_subs: list, training features
    :param y_train_subs: list, training target
    :param X_vals: list, validation features
    :param y_vals: list, validation targets
    :return:
    '''
    profit = 0
    for j in range(len(X_train_subs[i])):
        actual_positions, predicted_positions = get_positions([model], [dfs_train_sub[j][i]], [X_train_subs[j][i]],
                                                              [y_train_subs[j][i]])
        actual_positions, predicted_positions = get_positions([model], [dfs_val[j][i]], [X_vals[j][i]], training=False)
        position_summary(['a'], actual_positions, predicted_positions, [dfs_val[j][i]], printing=False, training=False)
        actual_profit, predicted_profit = profit_summary(['a'], [dfs_val[j][i]], printing=False,
                                                         training=False)  # 'a' -- placeholder
        profit += np.sum(np.array(predicted_profit) - np.array(actual_profit))
    return profit


def train_and_summary(models, names, dfs_train, X_trains, y_trains, printing=True):
    actual_positions, predicted_positions = get_positions(models, dfs_train, X_trains, y_trains)
    position_summary(names, actual_positions, predicted_positions, dfs_train, printing=printing)
    actual_profits, predicted_profits = profit_summary(names, dfs_train, printing=printing)

    errors = np.mean([mean_squared_error(models[i].predict(X_trains[i]), y_trains[i]) for i in range(2)])
    if printing:
        print("average MSE:", errors)


# for test set names
def test_summary(models, names, model_name, dfs_test, X_tests, y_tests, extra_profit, mse=False):
    actual_positions, predicted_positions = get_positions(models, dfs_test, X_tests, training=False)
    position_summary(names, actual_positions, predicted_positions, dfs_test, training=False)
    actual_profit, predicted_profit = profit_summary(names, dfs_test, training=False)
    extra_profit[model_name] = np.sum(np.array(predicted_profit) - np.array(actual_profit))
    if mse:
        errors = []
        for i in range(len(names)):
            errors.append(np.sqrt(mean_squared_error(models[i].predict(X_tests[i]), y_tests[i])))
        print("Average MSE", model_name, np.mean(errors))


# plot actual and predicted buy and sell signals
def plot_signals(df, name):
    '''
    :param df: single dataframe
    :param name: name of the stock
    :return:
    '''
    df_actual = df[df['actual position'] != 0]
    df_predicted = df[df['predicted position'] != 0]
    plt.figure(figsize=(14, 6))

    count = 0

    plt.subplot(1, 2, 1)
    plt.plot(df.index, df.close)
    for i in range(df_actual.shape[0]):
        if df_actual['actual position'][i] == 1:
            if count <= 1:
                plt.scatter([df_actual.index[i]], [df_actual.close[i]], marker='o', color='red', label='actual buy')
            else:
                plt.scatter([df_actual.index[i]], [df_actual.close[i]], marker='o', color='red')
            count += 1
        else:
            if count <= 1:
                plt.scatter([df_actual.index[i]], [df_actual.close[i]], marker='v', color='green', label='actual sell')
            else:
                plt.scatter([df_actual.index[i]], [df_actual.close[i]], marker='v', color='green')
            count += 1
    plt.legend()

    count = 0
    plt.subplot(1, 2, 2)
    plt.plot(df.index, df.close)

    for i in range(df_predicted.shape[0]):
        if df_predicted['predicted position'][i] == 1:
            if count <= 1:
                plt.scatter([df_predicted.index[i]], [df_predicted.close[i]], marker='s', color='orange',
                            label='predicted buy')
            else:
                plt.scatter([df_predicted.index[i]], [df_predicted.close[i]], marker='s', color='orange')
            count += 1
        elif df_predicted['predicted position'][i] == -1:
            if count <= 1:
                plt.scatter([df_predicted.index[i]], [df_predicted.close[i]], marker='d', color='purple',
                            label='predicted sell')
            else:
                plt.scatter([df_predicted.index[i]], [df_predicted.close[i]], marker='d', color='purple')
            count += 1
    plt.legend()
    plt.suptitle(name)
