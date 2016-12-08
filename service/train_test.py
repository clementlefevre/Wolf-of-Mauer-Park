def split_data_on_date(df, date):
    #date_until = datetime.strptime(date_str, '%Y-%m-%d')

    training_df = df[df.cal_time < date]
    forecast_df = df[df.cal_time >= date]
    return training_df, forecast_df


def regularize(df):
    reg_col = [col for col in df.columns.tolist() if not "cal_" in col]
    for col in reg_col:
        df[col + "_reg"] = (df[col] - df[col].mean()) / df[col].std()


def create_dataset(df, target, until_date, shift):
    X_y_dict = {}
    training_df, forecast_df = split_data_on_date(df, until_date)

    features_col = [col for col in df.columns if (
        "reg" in col or "cal_" in col)]
    features_col = [col for col in features_col if target.split('_')[
        1] not in col]

    features_col.remove('cal_time')

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
