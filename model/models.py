"""
Given a date, the Factory splits it into several time intervals.
For each time interval and each target, the Factory compute a prediction.and
All the prediction are stored as Prediction object."""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from service.trade_analysis import KELCH, DONCH, ATR, EMA, candlestick_prop, plot_candlestick, pip_delta


class Simulator(object):

    def __init__(self, dt_from=None, dt_to=None, target=None, shift=None,
                 ticks_to_shift=[],
                 fit_model=False,
                 clf=None,
                 datasource_path=None,
                 verbose=False):

        self.dt_from = datetime.strptime(dt_from, '%Y-%m-%d %H:%M')
        self.dt_to = datetime.strptime(dt_to, '%Y-%m-%d %H:%M')
        self.interval = self._create_interval()
        self.target = target
        self.shift = shift
        self.ticks_to_shift = ticks_to_shift
        self.fit_model = fit_model
        self.datasource_path = datasource_path
        self.predictions = None
        self.features_weight = None
        self.r2 = None
        self.verbose = verbose
        self.clf = clf
        self._set_target_params()
        logging.info('new Simulator created')

    def __repr__(self):
        return "Target : {0.target} : clf : {0.clf} : [{0.dt_from}-{0.dt_to}]\
         ticks_to_shift:{0.ticks_to_shift} shift:{0.shift}".format(self)

    def _set_target_params(self):

        self.target_root_name = self.target.split('_')[2]
        self.target_family = '_'.join(self.target.split('_')[:2])

    def _create_interval(self):

        interval = {'from': self.dt_from, 'to': self.dt_to}

        return interval


class TradeModel(object):

    def __init__(self, file_path, name, frequency='D',
                 datetime_col='Time (UTC)', sep=';', n=3, dayfirst=True):
        self.file_path = file_path
        self.frequency = frequency
        self.name = name
        self.n = n
        self.dayfirst = dayfirst
        self.datetime_col = datetime_col
        self.df = self._get_df(file_path)
        self.add_properties()

    def _get_df(self, file_path):
        df = pd.read_csv(
            file_path, parse_dates=[self.datetime_col], dayfirst=self.dayfirst)
        df = df.set_index(self.datetime_col)
        df.index.name = 'date_time'
        return df

    def add_properties(self):
        self.df = KELCH(self.df, self.n)
        self.df = DONCH(self.df, self.n)
        self.df = ATR(self.df)
        self.df = EMA(self.df)
        self.df = candlestick_prop(self.df)
        self.shift_candlestick()
        self.add_trend_parameters()
        self.add_calendar_params()

    def shift_candlestick(self):
        self.df['next_Low'] = self.df['Low'].shift(-1)
        self.df['next_High'] = self.df['High'].shift(-1)
        self.df['next_Close'] = self.df['Close'].shift(-1)

    def add_trend_parameters(self):
        self.df['DOWN_next_Low_under_Close'] = pip_delta(
            self.df, 'Close', 'next_Low', 20)

        self.df['DOWN_next_High_over_Close'] = pip_delta(
            self.df, 'Close', 'next_High', 18)

        self.df[
            'UP_next_High_over_Close'] = pip_delta(self.df, 'next_High', 'Close', 20)

        self.df[
            'UP_next_Low_under_Close'] = pip_delta(self.df, 'Close', 'next_Low', 18)

        self.df['DOWN_next_Low_and_High'] = self.df[
            'DOWN_next_Low_under_Close'] * self.df['DOWN_next_High_over_Close']

        self.df['UP_next_Low_and_High'] = self.df[
            'UP_next_Low_under_Close'] * self.df['UP_next_High_over_Close']

    def add_calendar_params(self):
        self.df['hour'] = self.df.index.hour
        self.df['day'] = self.df.index.weekday
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year

    def get_predictors_columns(self):
        return [col for col in self.df.columns.tolist() if 'next' not in col]

    def plot(self):
        df = self.df
        ax = plot_candlestick(df, self.name)
        df[['EMA12', 'EMA30', 'KC_M_' + str(self.n)]].plot(
            legend=True, style=['-', '-', '-', '--'], linewidth=1, ax=ax)

        plt.fill_between(df.index, df['KC_D_' + str(self.n)], df['KC_U_' + str(self.n)],
                         facecolor='red', alpha=0.1, interpolate=True)
        df[['Donchian_High_' + str(self.n), 'Donchian_Low_' + str(self.n)]].plot(
            style=':', linewidth=1, ax=ax, color='r')

        df.ATR.plot(secondary_y=True, style='y', linewidth=1)
        ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(.5, 1.0))
