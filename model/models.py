"""
Given a date, the Factory splits it into several time intervals.
For each time interval and each target, the Factory compute a prediction.and
All the prediction are stored as Prediction object."""

from datetime import datetime, timedelta
import pandas as pd


class Prediction(object):

    def __init__(self, df, serie_name, interval, shift):
        self.df = df
        self.serie_name = serie_name
        self.shift = shift
        self.interval = interval

    def __repr__(self):
        self.interval_str = self.interval['from'].strftime('%Y-%m-%d %H:%M') + " to " +\
            self.interval['to'].strftime('%Y-%m-%d %H:%M')

        return "{0.serie_name} : {0.shift} : {0.interval_str} :\
         {0.df.shape[0]}".format(self)


class Simulator(object):

    def __init__(self, dt_from=None, dt_to=None, targets=None, shift=None, steps=10, fit_model=False):

        self.dt_from = datetime.strptime(dt_from, '%Y-%m-%d %H:%M')
        self.dt_to = datetime.strptime(dt_to, '%Y-%m-%d %H:%M')
        self.steps = steps
        self.intervals = self._create_intervals()
        self.targets = targets
        self.shift = shift
        self.fit_model = fit_model
        self.predictions = []

    def __repr__(self):
        return "Targets : {0.targets} : [{0.dt_from}-{0.dt_to}] steps:{0.steps} shift:{0.shift}".format(self)

    def _create_intervals(self):
        intervals = []
        dt_ = self.dt_from

        while dt_ < self.dt_to:
            intervals.append(
                {'from': dt_, 'to': dt_ + timedelta(minutes=self.steps)})
            dt_ += timedelta(minutes=self.steps)
        return intervals
