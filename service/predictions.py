import pickle
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import logging

from service.train_test import create_dataset
from model.models import Prediction


def fit(dataset, target, shift):
    model = xgboost.XGBRegressor(max_depth=30,
                                 learning_rate=0.1,
                                 n_estimators=100,
                                 silent=True,
                                 objective='reg:linear',
                                 nthread=-1,
                                 gamma=0,
                                 min_child_weight=1,
                                 max_delta_step=0,
                                 subsample=1,
                                 colsample_bytree=1,
                                 colsample_bylevel=1,
                                 reg_alpha=0.2,
                                 reg_lambda=1,
                                 scale_pos_weight=1,
                                 base_score=0.5,
                                 seed=0,
                                 missing=None)
    model.fit(dataset['training_X'], dataset['training_y'])

    pickle.dump(model, open(
        "data/predictions/model_{}.p".format(target), "wb"))


def predict(dataset, target, shift):
    model = pickle.load(
        open("data/predictions/model_{}.p".format(target), "rb"))

    predictions = model.predict(dataset['forecast_X'])

    prediction_df = pd.DataFrame({'predicted': predictions,
                                  'observed': dataset[
                                      'observed_y']},
                                 index=dataset['label_forecast'])

    #prediction_df.predicted = prediction_df.predicted.shift(periods=shift)

    return prediction_df


def make_prediction(df, target, interval, shift, fit_model=False):
    dataset = create_dataset(df, target, interval, shift)
    if fit_model:
        fit(dataset, target, shift)

    df_prediction = predict(dataset, target, shift)
    return Prediction(df_prediction, target, interval, shift)


def plot_predictions(simulator):
    fig, axes = plt.subplots(len(simulator.predictions), figsize=(12, 12))
    fig.suptitle(
        "XGBoost regressor with {0}steps in advance".format(simulator.shift))
    fig.subplots_adjust(hspace=0.6, wspace=0.3)

    for j, prediction in enumerate(simulator.predictions):
        title = str(prediction)
        df = prediction.df[simulator.shift:]
        if not df.empty:

            df.plot(ax=axes[j], title=title, lw=1)


def compute_and_pickle(simulator, dataset_path=None):

    df_merged = pd.read_csv(dataset_path,
                            sep=';', parse_dates=['cal_time'])
    for target in simulator.targets:
        for interval in simulator.intervals:
            logging.info(
                "compute prediction for {0} : {1}".format(target, interval))
            prediction = make_prediction(
                df_merged, target, interval,
                simulator.shift, simulator.fit_model)
            simulator.predictions.append(prediction)

    pickle.dump(simulator, open("data/predictions/simulator.p", "wb"))
