from datetime import datetime, timedelta
from model.models import Simulator
from service.predictions import compute_and_pickle


def test_simulator():
    dt_from = datetime(2016, 11, 1, 8)
    dt_to = dt_from + timedelta(minutes=60 * 8)
    simulator = Simulator(dt_from=dt_from, dt_to=dt_to, targets=[
        'Ask_GBPUSD'], shift=3, steps=60 * 8)

    compute_and_pickle(simulator)
    assert len(simulator.predictions) > 0
