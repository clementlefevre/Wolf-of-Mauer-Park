import os
import pandas as pd
import logging
from model.models import Simulator
import service.regressor as regressor
import pickle

import service.files_service as fs
import csv
import datetime


datasource_path = '/drive_ext4/merged_no_spread_light.csv'

cols = ['Open_Ask_EURRUB', 'Open_Ask_USDRUB', 'Open_Ask_USDCAD', 'Open_Ask_LIGHTCMDUSD',
        'Open_Ask_USDMXN', 'Open_Ask_EURNOK', 'Open_Ask_USDNOK', 'Open_Ask_BRENTCMDUSD']


def compute(targets):
    for target in targets:
        for shift in [3]:
            simulator = Simulator(dt_from='2016-10-24 12:00', dt_to='2016-10-26 12:30', target=target, shift=shift, fit_model=True, datasource_path=datasource_path, ticks_to_shift=[0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50],verbose=True)
            regressor.dataset(simulator)
            regressor.fit(simulator)
            regressor.predict(simulator)
            print simulator.features_weight[:10]


if __name__ == '__main__':
    try:
        compute(['Open_Ask_EURRUB', 'Open_Ask_USDRUB', 'Open_Ask_USDCAD', 'Open_Ask_LIGHTCMDUSD', 'Open_Ask_USDMXN',
                 'Open_Ask_EURNOK', 'Open_Ask_USDNOK', 'Open_Ask_BRENTCMDUSD'])
    except Exception as e:
        logging.error('Error :')
        raise
