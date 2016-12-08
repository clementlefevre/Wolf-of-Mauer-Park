import os

import pandas as pd


def get_csv_files_dict():
    files_dict = {}
    files = [filo for filo in os.listdir("data") if ".csv" in filo]
    for file in files:

        file_name = file.split('_UTC')[0]
        files_dict[file_name] = file
    return files_dict


def read_chunk(file_path, chunksize=1000000, resample=None):
    print file_path
    chunks = pd.read_csv("data/" + file_path,
                         parse_dates=['Time (UTC)'], chunksize=chunksize)
    df = pd.DataFrame()
    for _ in chunks:
        _.rename(columns={'Time (UTC)': 'Time'}, inplace=True)
        _.rename(columns={'BidVolume ': 'BidVolume'}, inplace=True)
        _.set_index('Time', inplace=True)
        _ = _.resample(resample).mean()
        _['spread'] = _['Ask'] - _['Bid']
        _['spread_volume'] = _['AskVolume'] - _['BidVolume']
        df = pd.concat([df, _], axis=0)

    df.to_csv('data/resampled/' + file_path)


def create_files(resample=None):
    files_dict = get_csv_files_dict()
    for file_name, file_path in files_dict.iteritems():
        read_chunk(file_path=file_path, resample=resample)


def combine_files(resample=None):
    onlyfiles = [f for f in os.listdir(
        'data/resampled/') if os.path.isfile(os.path.join('data/resampled/', f))]
    onlyfiles = [filo for filo in onlyfiles if ".csv" in filo]
    print onlyfiles

    df_all = pd.DataFrame()
    for filo in onlyfiles:
        serie_name = filo.split('_UTC')[0]
        df = pd.read_csv(os.path.join('data/resampled/', filo),
                         parse_dates=['Time'], index_col=0)
        df.columns = [col + "_" + serie_name for col in df.columns.tolist()]
        df.reset_index()

        try:
            df_all = pd.merge(df.reset_index(), df_all, on='Time', how='outer')
        except KeyError:
            df_all = df.reset_index()
    return df_all


def add_calendar_data(df):
    df['cal_hour'] = df.Time.dt.hour
    df['cal_minute'] = df.Time.dt.minute
    df['cal_dayofweek'] = df.Time.dt.dayofweek
    df['cal_dayofyear'] = df.Time.dt.dayofyear
    df.rename(columns={'Time': 'cal_time'}, inplace=True)


def regularize(df):
    reg_col = [col for col in df.columns.tolist() if not "cal_" in col]
    for col in reg_col:
        df[col + "_reg"] = (df[col] - df[col].mean()) / df[col].std()


def create_raw_data(sample_period='1Min'):

    # create_files(resample=sample_period)
    df = combine_files(resample=sample_period)
    df.to_csv('data/merged/raw.csv', sep=';')
    add_calendar_data(df)
    regularize(df)
    df = df[pd.notnull(df.Ask_EURGBP)]
    df = df[pd.notnull(df.Ask_GBPUSD)]
    df = df[pd.notnull(df.Ask_EURUSD)]

    df.to_csv('data/merged/regularized.csv', sep=';')
