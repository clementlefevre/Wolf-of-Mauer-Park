from datetime import datetime, timedelta
from model.models import Simulator
from service.predictions import compute_and_pickle
import logging


def test_simulator():
    start = datetime.now()
    dt_from = datetime(2016, 11, 1, 8)
    # , 'Ask_GBPUSD', 'Ask_CADCHF', 'Ask_XAGUSD', 'Ask_BRENTCMDUSD', 'Ask_EURCHF', 'Ask_GBPCHF', 'Ask_CHFSGD']
    targets = ['Ask_EURGBP']  # , 'Ask_EURUSD']
    dt_to = dt_from + timedelta(minutes=60)
    simulator = Simulator(dt_from=dt_from, dt_to=dt_to, targets=targets,
                          shift=3, steps=10, fit_model=True)

    compute_and_pickle(simulator)
    total_time = datetime.now() - start
    logging.info("test took : {} seconds".format(total_time.total_seconds()))
    assert len(simulator.predictions) > 0
