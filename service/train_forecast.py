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


def add_pip(df, target):

    df.loc[target + '_pip'] = df[target] - df[target].iloc[0]
    return df


def add_previous_ticks(df, simulator):
    cols = [col for col in df.columns.tolist() if
            simulator.target.split('_')[2] in col]
    print 'cols of target : {}'.format(cols)
    logging.info('ticks_to_shift : {}'.format(simulator.ticks_to_shift))
    for col in cols:
        for tick in simulator.ticks_to_shift:

            df[col + '_shifted_' +str(tick)] = df[col].shift(periods=tick)
            logging.info('ticks : {} added for column :{}'.format(tick, col))

    df = df.fillna(0)

    return df


def clean_features(df, target):
    target_root_name = target.split('_')[2]
    # Keep only regularized features
    features_col = [col for col in df.columns if (
        "reg" in col or "cal_" in col)]
    logging.info('features col before target exlusion :{}'.format(features_col))
    # Exclude the target from features
    features_col = [col for col in features_col if (
        target_root_name not in col or '_shifted' in col)]
    # Time index is not a feature
    features_col.remove('cal_time')
    logging.info('Features selected: {}'.format(features_col))
    return features_col


def create_dataset(df, simulator):
    logging.info('Start creating the dataset for {}'.format(simulator.target))

    df = add_previous_ticks(df, simulator)
    logging.info('df columns before train split : {}'.format(df.columns))
    X_y_dict = {}
    training_df, forecast_df = train_forecast_split(df, simulator.interval)

    # Tracer()()  # this one triggers the debugger

    training_df.loc[simulator.target] = training_df[
        simulator.target].shift(periods=-simulator.shift)

    training_df = training_df[:-simulator.shift]

    features_col = clean_features(df, simulator.target)

    X_y_dict['training_X'] = training_df[
        features_col].values

    X_y_dict['training_y'] = training_df[simulator.target].values

    X_y_dict['forecast_X'] = forecast_df[features_col].values

    X_y_dict['observed_y'] = forecast_df[simulator.target].values

    X_y_dict['label_training'] = training_df.cal_time.values
    X_y_dict['label_forecast'] = forecast_df.cal_time.values

    X_y_dict['features_names'] = features_col

    logging.info(
        'Finished creating the dataset for {}'.format(simulator.target))
    return X_y_dict
