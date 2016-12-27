import pickle
import config
import pandas as pd
from sklearn.grid_search import GridSearchCV
import xgboost
import logging

from service.train_forecast import create_dataset
from model.models import Prediction


def get_classifier_params():
    clf = xgboost.XGBRegressor()
    params = dict(max_depth=[30],
                  learning_rate=[0.1],
                  n_estimators=[200],
                  silent=[False],
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
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=-1)
    gs.fit(Xtrain, ytrain)
    print "BEST PARAMS", gs.best_params_
    best = gs.best_estimator_
    pickle.dump(best, open(config.data_store_path +'predictions/model_{}.p'.format(target), "wb"))


def features_weight(simulator):
    model = pickle.load(
        open(config.data_store_path +'predictions/model_{}.p'.format(simulator.target), "rb"))
    dataset = pickle.load(
        open(config.data_store_path +'predictions/dataset_{}.p'.format(simulator.target), "rb"))

    df_weights = pd.DataFrame(
        model.booster().get_score().items(), columns=['index', 'weight'])

    df_weights['feature_name'] = dataset['features_names']
    df_weights = df_weights.sort_values(by='weight', ascending=False)

    return df_weights


def fit_model(dataset, simulator):
    clf, params = get_classifier_params()
    if simulator.fit_model:
        logging.info('Start GridSearchCV...')
        cv_optimize(simulator.target,
                    clf, params, dataset['training_X'], dataset['training_y'])
        logging.info('finished  fitting via GridSearchCV')




def dataset(simulator):
    df_source = pd.read_csv(simulator.datasource_path,
                            sep=';', parse_dates=['cal_time'])

    dataset = create_dataset(df_source, simulator)
    pickle.dump(
        dataset, open(config.data_store_path + 'predictions/dataset_{}.p'.format(simulator.target), 'wb'))


def fit(simulator):
    logging.info('Start fitting the model {}'.format(simulator))
    # for each target, fit a model

    dataset = pickle.load(
        open(config.data_store_path + 'predictions/dataset_{}.p'.format(simulator.target), 'rb'))
    fit_model(dataset, simulator)
    simulator.features_weight = features_weight(simulator)


def predict(simulator):
    model = pickle.load(
        open(config.data_store_path + 'predictions/model_{}.p'.format(simulator.target), "rb"))
    dataset = pickle.load(
        open(config.data_store_path + 'predictions/dataset_{}.p'.format(simulator.target), 'rb'))

    predictions = model.predict(dataset['forecast_X'])
    df_prediction = pd.DataFrame({'predicted': predictions, 'observed': dataset['observed_y']},
                                 index=dataset['label_forecast'])

    r2 = model.score(dataset['forecast_X'],
                     dataset['observed_y'])

    simulator.r2 = r2
    logging.info('R2 = {}'.format(r2))

    df_prediction.predicted = df_prediction.predicted.shift(
        periods=simulator.shift)

    simulator.predictions = Prediction(
        df_prediction, simulator.target, simulator.interval, simulator.shift, r2)
    pickle.dump(simulator, open(
        config.data_store_path + 'predictions/simulator_{}.p'.format(simulator.target), 'wb'))
