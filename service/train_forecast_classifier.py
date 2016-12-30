'''This module split the data to compute a prediction
There is one specific features in the process:

The aim is to predict a value for a time serie

Given a dataset with time index:
 __________________________
|time_i    | X_i    | y_i  |
|----------|--------|------|
|time_i+1  | X_i+1  | y_i+1|
|----------|--------|------|


The trick is to considere the (X_i,y_i+n) as a pair when fitting the model
where n is the number of periods shifted.

Given X_i at time_i, we can then predict the y_i+n value at time_i+n.


'''

from IPython.core.debugger import Tracer
import pandas as pd
import logging


def train_forecast_split(df, interval):
    training_df = df[df.cal_time < interval['from']]

    forecast_df = df[(df.cal_time >= interval['from'])
                     & (df.cal_time < interval['to'])]
    return training_df, forecast_df


def add_previous_ticks(df, simulator):
    logging.info('Add previous ticks...')
    cols_target = [col for col in df.columns.tolist() if (
        simulator.target_root_name in col and '_reg' in col)]
    cols_others = [col for col in df.columns.tolist() if ('_reg' in col and
                                                          simulator.target_family
                                                          in col)]

    cols = cols_target + cols_others

    if simulator.verbose:
        logging.info('Columns with shifted ticks : ')
        logging.info(cols)

    for col in cols:
        for tick in simulator.ticks_to_shift:

            df[col + '_shifted_' + str(tick)] = df[col].shift(periods=tick)

    df = df.fillna(0)

    return df


def add_pip_categories(df, simulator):

    def compute_category(row):
        if row.pip_O >= 10:
            return 1
        if row.pip_O <= -10:
            return 2
        else:
            return 0

    df['target_shifted'] = df[simulator.target].shift(periods=3)

    df['pip_O'] = (df[simulator.target] - df['target_shifted']) * 10000

    df['pip_category'] = df.apply(compute_category, axis=1)

    return df


def set_features(df, simulator):
    logging.info('Set features...')


def clean_features(df, simulator):
    logging.info('Clean features...')
    cols = df.columns.tolist()

    # Keep only calendar columns and regularized features and shifted values
    features_calendar = [col for col in cols if 'cal_' in col]

    features_regularized = [col for col in cols if ("_reg" in col and
                                                    simulator.target_root_name not in
                                                    col and simulator.target_family in
                                                    col)]

    features_shifted = [col for col in cols if '_shifted_' in col]

    features_col = set(features_calendar + features_regularized +
                       features_shifted)

    # Time index is not a feature
    features_col.remove('cal_time')

    if simulator.verbose:
        logging.info('Features to be used as predictors :')
        logging.info(features_col)

    return list(features_col)


def create_dataset(df, simulator):
    logging.info('Start creating the dataset for {}'.format(simulator.target))

    df = add_previous_ticks(df, simulator)
    df = add_pip_categories(df, simulator)

    X_y_dict = {}

    training_df, forecast_df = train_forecast_split(df, simulator.interval)

    #Tracer()()  # this one triggers the debugger

    training_df[simulator.target] = training_df[
        simulator.target].shift(periods=-simulator.shift)

    training_df = training_df[:-simulator.shift]

    features_col = clean_features(df, simulator)

    X_y_dict['training_X'] = training_df[
        features_col].values

    X_y_dict['training_y'] = training_df['pip_category'].values

    X_y_dict['forecast_X'] = forecast_df[features_col].values

    X_y_dict['observed_y'] = forecast_df['pip_category'].values
    X_y_dict['observed_y_retroshifted'] = forecast_df[
        'pip_category'].shift(periods=-simulator.shift).values

    X_y_dict['label_training'] = training_df.cal_time.values
    X_y_dict['label_forecast'] = forecast_df.cal_time.values

    X_y_dict['features_names'] = features_col

    logging.info(
        'Finished creating the dataset for {}'.format(simulator.target))
    return X_y_dict
