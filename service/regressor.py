import pandas as pd
from sklearn.grid_search import GridSearchCV
import xgboost
import logging
import warnings
from service.persistence import pickle_dump, pickle_load
from service.train_forecast import create_dataset
from IPython.core.debugger import Tracer

warnings.filterwarnings("ignore")


def get_classifier_params():
    clf = xgboost.XGBRegressor()
    params = dict(max_depth=[12],
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


def cv_optimize(simulator, clf, parameters, Xtrain, ytrain, n_folds=5):
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=-1)
    gs.fit(Xtrain, ytrain)
    print "BEST PARAMS", gs.best_params_
    best = gs.best_estimator_
    pickle_dump(best, simulator)


def features_weight(simulator):
    model = pickle_load('model', simulator)
    dataset = pickle_load('dataset', simulator)

    df_weights = pd.DataFrame(
        model.booster().get_score().items(), columns=['index', 'weight'])

    # the length of features weight from the model is not always the length of total features used
    # as predictor in the fitting !!!
    # adjust the features list according to the shape of the model's features
    # weight list.dataset
    df_weights['feature_name'] = dataset[
        'features_names'][:df_weights.shape[0]]
    df_weights = df_weights.sort_values(by='weight', ascending=False)

    return df_weights


def fit_model(dataset, simulator):
    clf, params = get_classifier_params()
    if simulator.fit_model:
        logging.info('Start GridSearchCV...')
        cv_optimize(simulator,
                    clf, params, dataset['training_X'], dataset['training_y'])
        logging.info('finished  fitting via GridSearchCV')


def dataset(simulator):
    df_source = pd.read_csv(simulator.datasource_path,
                            sep=';', parse_dates=['cal_time'])

    dataset = create_dataset(df_source, simulator)
    pickle_dump(dataset, simulator)


def fit(simulator):
    logging.info('Start fitting the model {}'.format(simulator))
    # for each target, fit a model

    dataset = pickle_load('dataset', simulator)
    fit_model(dataset, simulator)
    simulator.features_weight = features_weight(simulator)


def predict(simulator):
    model = pickle_load('model', simulator)
    dataset = pickle_load('dataset', simulator)

    predictions = model.predict(dataset['forecast_X'])
    df_prediction = pd.DataFrame({'predicted': predictions,
                                  'observed': dataset['observed_y']},
                                 index=dataset['label_forecast'])

    r2 = model.score(dataset['forecast_X'],
                     dataset['observed_y_retroshifted'])

    simulator.r2 = r2
    logging.info('Simulator {0} : R2= {1}'.format(simulator, r2))

    df_prediction.predicted = df_prediction.predicted.shift(
        periods=simulator.shift)

    simulator.predictions = df_prediction
    pickle_dump(simulator, simulator)
