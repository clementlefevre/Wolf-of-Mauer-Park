"""
Given a date, the Factory splits it into several time intervals.
For each time interval and each target, the Factory compute a prediction.and
All the prediction are stored as Prediction object."""

from datetime import datetime
import logging


class Simulator(object):

    def __init__(self, dt_from=None, dt_to=None, target=None, shift=None,
                 ticks_to_shift=[],
                 fit_model=False, datasource_path=None):

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
        logging.info('new Simulator created')

    def __repr__(self):
        return "Targets : {0.target} : [{0.dt_from}-{0.dt_to}]\
         ticks_to_shift:{0.ticks_to_shift} shift:{0.shift}".format(self)

    def _create_interval(self):

        interval = {'from': self.dt_from, 'to': self.dt_to}

        return interval
