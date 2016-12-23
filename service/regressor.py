import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import xgboost
import logging

from service.train_forecast import create_dataset
from model.models import Prediction


def get_classifier_params():
    clf = xgboost.XGBRegressor()
    params = dict(max_depth=[30, 60, 90, 300],
                  learning_rate=[0.01, 0.1],
                  n_estimators=[50, 100, 200],
                  silent=[True],
                  objective=['reg:linear'],
                  nthread=[-1],
                  gamma=[0],
                  min_child_weight=[1],
                  max_delta_step=[0],
                  subsample=[1],
                  colsample_bytree=[1],
                  colsample_bylevel=[1],
                  reg_alpha=[0.2],
                  reg_lambda=[1],
                  scale_pos_weight=[1],
                  base_score=[0.5],
                  seed=[0],
                  missing=[None])
    return clf, params


def cv_optimize(target, clf, parameters, Xtrain, ytrain, n_folds=5):
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    gs.fit(Xtrain, ytrain)
    print "BEST PARAMS", gs.best_params_
    best = gs.best_estimator_
    pickle.dump(best, open(
        "data/predictions/model_{}.p".format(target), "wb"))


def fit_model(dataset, target):
    clf, params = get_classifier_params()
    if fit_model:
        logging.info('Start GridSearchCV...')
        cv_optimize(target,
                    clf, params, dataset['training_X'], dataset['training_y'])
        logging.info('finished  fitting via GridSearchCV')


def make_prediction(dataset, target, interval, shift):
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


def create_dataset_and_fit(simulator):
    logging.info('Start fitting the model {}'.format(simulator))
    # for each target, fit a model
    df_source = pd.read_csv(simulator.datasource_path,
                            sep=';', parse_dates=['cal_time'])

    for target in simulator.targets:
        dataset = create_dataset(df_source, target, simulator.intervals,
                                 simulator.shift)
        pickle.dump(
            dataset, open('data/predictions/dataset_{}'.format(target), 'wb'))
        fit_model(dataset, target)


def predict(simulator, target):
    model = pickle.load(
        open("data/predictions/model_{}.p".format(target), "rb"))
    dataset = pickle.load(
        open('data/predictions/dataset_{}'.format(target), 'rb'))

    target_predictions = []

    for i, interval in enumerate(simulator.intervals):

        predictions = model.predict(dataset['forecast_X_' + str(i)])
        df_prediction = pd.DataFrame({'predicted': predictions, 'observed': dataset['observed_y_' + str(i)]},
                                     index=dataset['label_forecast'])

        df_prediction.predicted = df_prediction.predicted.shift(
            periods=simulator.shift)
        target_predictions.append(Prediction(
            df_prediction, target, interval, simulator.shift))
    simulator.predictions[target] = target_predictions
    pickle.dump(simulator, open(
        'data/predictions/simulator_{}'.format(target), 'wb'))
