import logging


def train_test_split(df, interval):
    training_df = df[df.cal_time < interval['from']]
    forecast_df = df[(df.cal_time >= interval['from'])
                     & (df.cal_time < interval['to'])]
    return training_df, forecast_df


def clean_features(df, target):
    # Keep only regularized features
    features_col = [col for col in df.columns if (
        "reg" in col or "cal_" in col)]

    # Exclude the target from features
    features_col = [col for col in features_col if target.split('_')[
        1] not in col]

    # Time index is not a feature
    features_col.remove('cal_time')

    return features_col


def create_dataset(df, target, interval, shift=None):
    X_y_dict = {}
    training_df, forecast_df = train_test_split(df, interval)

    features_col = clean_features(df, target)

    X_y_dict['training_X'] = training_df[
        features_col].shift(periods=shift).values

    X_y_dict['training_y'] = training_df[target].values

    X_y_dict['forecast_X'] = forecast_df[
        features_col].shift(periods=shift).values

    X_y_dict['observed_y'] = forecast_df[target].values

    X_y_dict['label_training'] = training_df.cal_time.values
    X_y_dict['label_forecast'] = forecast_df.cal_time.values

    X_y_dict['features_names'] = features_col

    return X_y_dict
