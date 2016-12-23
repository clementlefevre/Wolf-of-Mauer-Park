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
import logging


def train_forecast_split(df, intervals):
    training_df = df[df.cal_time < intervals[0]['from']]

    forecast_df_list = []
    for interval in intervals:
        forecast_df = df[(df.cal_time >= interval['from'])
                         & (df.cal_time < interval['to'])]
        forecast_df_list.append(forecast_df)

    return training_df, forecast_df_list


def clean_features(df, target):
    # Keep only regularized features
    features_col = [col for col in df.columns if (
        "reg" in col or "cal_" in col)]

    # Exclude the target from features
    features_col = [col for col in features_col if target not in col]
    # Time index is not a feature
    features_col.remove('cal_time')

    return features_col


def create_dataset(df, target, intervals, shift=None):
    logging.info('Start creating the dataset for {}'.format(target))
    X_y_dict = {}
    training_df, forecast_df_list = train_forecast_split(df, intervals)

    features_col = clean_features(df, target)

    X_y_dict['training_X'] = training_df[
        features_col].shift(periods=shift).values
    X_y_dict['training_y'] = training_df[target].values

    for i, forecast_df in enumerate(forecast_df_list):

        X_y_dict['forecast_X_' + str(i)] = forecast_df[
            features_col].shift(periods=shift).values

        X_y_dict['observed_y_' + str(i)] = forecast_df[target].values

    X_y_dict['label_training'] = training_df.cal_time.values
    X_y_dict['label_forecast'] = forecast_df.cal_time.values

    X_y_dict['features_names'] = features_col

    logging.info('Finished creating the dataset for {}'.format(target))
    return X_y_dict
