import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import talib


# Keltner Channel
def KELCH(df, n):
    for n in range(1, n + 1):

        df['KC_M_' + str(n)] = df[['High', 'Low', 'Close']
                                  ].mean(axis=1).rolling(3).mean()
        df['KC_U_' + str(n)] = ((4 * df['High'] - 2 * df['Low'] +
                                 df['Close']) / 3).rolling(3).mean()
        df['KC_D_' + str(n)] = ((-2 * df['High'] + 4 * df['Low'] +
                                 df['Close']) / 3).rolling(3).mean()

    return df


# Donchian Channel
def DONCH(df, n):
    df['Donchian_High_' + str(n)] = df.High.rolling(n).max()
    df['Donchian_Low_' + str(n)] = df.Low.rolling(n).min()
    return df


def ATR(df, n=4):
    df['ATR'] = talib.ATR(df.High.values, df.Low.values,
                          df.Close.values, timeperiod=n)
    return df


def EMA(df):
    df['EMA12'] = talib.EMA(df.Open.values, timeperiod=12)
    df['EMA30'] = talib.EMA(df.Open.values, timeperiod=30)
    return df


def pip_delta(df, col1, col2, pip_threshold):

    return np.where((df[col1] - df[col2]) * 10000.0 > pip_threshold, 1, 0)


def candlestick_prop(df):
    df['cs_size'] = df.High - df.Low
    df['cs_body_size'] = df.Open - df.Close
    df['cs_body_pos'] = (df.Close + df.Open) / 2
    df['cs_body_ratio'] = (df.High + df.Low) / 2 - df['cs_body_pos']
    return df


def plot_candlestick(df, name, ax=None, fmt="%Y-%m-%d"):
    if ax is None:
        fig, ax = plt.subplots()
    idx_name = df.index.name
    dat = df.reset_index()[[idx_name, "Open", "High", "Low", "Close"]]
    dat[df.index.name] = dat[df.index.name].map(mdates.date2num)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.xticks(rotation=45)
    _ = candlestick_ohlc(ax, dat.values, width=.6, colorup='g', alpha=1)
    ax.set_xlabel(idx_name)
    ax.set_ylabel(name)
    return ax

# source: http://seaborn.pydata.org/examples/many_pairwise_correlations.html


def plot_correlation(df):
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots()

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
