import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import talib


# Keltner Channel
def KELCH(df, n):

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
