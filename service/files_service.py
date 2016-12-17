import os
import pandas as pd
import zipfile
import logging

# this one triggers the debugger

LAG = 10


def _get_files(folder=None, extension=None, as_dict=False):

    files = [filo for filo in os.listdir(
        "data/" + folder) if extension in filo]

    if as_dict:
        files_dict = {}
        for file in files:

            file_name = file.split('_UTC')[0]
            files_dict[file_name] = file
        return files_dict
    else:
        return files


def merge_ask_bid(source_folder=None):
    files_dict = {}
    files = _get_files(folder=source_folder, extension='.csv')
    files = sorted(files)

    if not os.path.exists('data/' + source_folder + '/ask_bid'):
        os.makedirs('data/' + source_folder + '/ask_bid')
        logging.info("folder ask_bid created in {}".format(source_folder))

    for i, f in enumerate(files):
        key = f.split('_UTC_')[0]

        if key not in files_dict.keys():
            files_dict[key] = []
        files_dict[key].append(f)
    for k in files_dict.keys():
        index_0 = ''.join(['_', files_dict[k][0].split('_')[3]])
        index_1 = ''.join(['_', files_dict[k][1].split('_')[3]])
        print index_0

        df1 = pd.read_csv('data/' + source_folder + '/' + files_dict[k][0])
        df2 = pd.read_csv('data/' + source_folder + '/' + files_dict[k][1])

        df1 = df1.rename(columns={'Volume_': 'Volume'})
        df2 = df2.rename(columns={'Volume_': 'Volume'})
        df = pd.merge(df1, df2, on='Time (UTC)', suffixes=[index_0, index_1])
        df.to_csv('data/' + source_folder + '/ask_bid/' + k + '.csv')


def _chunk_and_resample(file_path, chunksize=1000000, resample=None):
    logging.info("Start resampling :{}".format(file_path))

    if not os.path.exists('data/' + source_folder + '/resampled'):
        os.makedirs('data/' + source_folder + '/resampled')
        logging.info("folder resampled created in {}".format(source_folder))

    chunks = pd.read_csv("data/" + file_path,
                         parse_dates=['Time (UTC)'], chunksize=chunksize)
    df = pd.DataFrame()

    for i, _ in enumerate(chunks):

        _ = _transform(_)

        df = pd.concat([df, _], axis=0)

    logging.info("Finished resampling :{}".format(file_path))
    df.to_csv('data/' + source_folder + '/resampled' + file_path)


def _transform(df, resample=None):
    df.columns = [col.replace(" ", "") for col in df.columns]
    df = df.drop('Unnamed:0', 1)

    df = df.rename(columns={'Time(UTC)': 'datetime'})
    df2 = df.set_index('datetime')
    df2 = df2.resample(resample).mean()
    #_add_spread(df2)

    _add_rolling_mean(df2)
    _add_ewma(df2)

    return df2


def _transform_folder(source_folder=None):
    files = _get_files(folder=source_folder, extension='.csv')
    path = 'data/' + source_folder + '/'
    if not os.path.exists(path + 'transformed'):
        os.makedirs(path + 'transformed')
        logging.info("folder resampled created in {}".format(source_folder))
    for file in files:

        df = pd.read_csv(path + file, parse_dates=['Time (UTC)'])
        df = _transform(df, resample='1Min')
        df.to_csv(path + 'transformed/' + file)


def _add_spread(df):
    index = ['Open', 'Low', 'High', 'Close', 'Volume']
    for i in index:
        df[i + '_spread'] = df[i + '_Bid'] - df[i + '_Ask']


def _add_rolling_mean(df):
    cols = df.columns
    for col in cols:
        df[col +
            '_rolling_mean'] = df[col].rolling(window=10, min_periods=1).mean()


def _add_ewma(df):
    cols = df.columns
    cols = [col for col in cols if '_lag_' not in col]
    cols = [col for col in cols if '_rolling_mean' not in col]
    print cols
    for col in cols:
        df[col + "_ewma"] = pd.ewma(df[col], span=60)


def _chunk_and_resample_folder(resample=None, source_folder=None):
    files_dict = _get_files(folder=source_folder,
                            extension='csv', as_dict=False)
    for file_name, file_path in files_dict.iteritems():
        _chunk_and_resample(file_path=file_path, resample=resample)


def _merge_files(source_folder=None, resample=None, filter_on=None):
    files_to_merge = _get_files(folder=source_folder, extension='.csv')

    df_all = pd.DataFrame()

    if not os.path.exists('data/' + source_folder + '/merged'):
        os.makedirs('data/' + source_folder + '/merged')
        logging.info("folder merged created in {}".format(source_folder))
    for file in files_to_merge:

        df = pd.read_csv(os.path.join('data/' + source_folder, file),
                         parse_dates=['datetime'], index_col=0)

        df = _filter_on_columns(df, filter_on)

        df.columns = [col + "_" +
                      file.replace('.csv', '') for col in df.columns.tolist()]
        df.reset_index()

        try:
            df_all = pd.merge(df.reset_index(), df_all,
                              on='datetime', how='outer')
        except KeyError:
            df_all = df.reset_index()
    return df_all


def _filter_on_columns(df, filter_on=None):
    if filter_on is not None:
        df = df[[col for col in df.columns if filter_on in col]]
    return df


def _add_calendar_data(df):
    df['cal_hour'] = df.datetime.dt.hour
    df['cal_minute'] = df.datetime.dt.minute
    df['cal_dayofweek'] = df.datetime.dt.dayofweek
    df['cal_dayofyear'] = df.datetime.dt.dayofyear
    df.rename(columns={'datetime': 'cal_time'}, inplace=True)


def _regularize(df):
    reg_col = [col for col in df.columns.tolist() if not "cal_" in col]
    for col in reg_col:
        df[col + "_reg"] = (df[col] - df[col].mean()) / df[col].std()


def _remove_null(df, source_folder=None):
    index_names = _get_files(
        folder=source_folder, extension='csv')
    print index_names
    for index in index_names:
        df = df[pd.notnull(df['Close_Ask_' + index.replace('.csv', '')])]

    return df


def reformat_csv(file_name):
    """Fix wrongly formatted csv file
     when separator is missing and date format is reversed.
    """
    df_all = pd.DataFrame()
    chunks = pd.read_csv('data/NOK/' + file_name,
                         skiprows=[0], names=['Time (UTC)', 'Ask', 'Bid', 'AskVolume', 'BidVolume '], chunksize=1000000, delim_whitespace=True)
    print sum(1 for row in open('data/NOK/' + file_name, 'r'))
    for i, df in enumerate(chunks):
        print i
        df = df.reset_index()
        df['index'] = df['index'].apply(
            lambda x: '.'.join(x.split('.')[:: -1]))
        df['Time (UTC)'] = df['index'] + " " + df['Time (UTC)']
        df = df.drop('index', 1)

        df_all = pd.concat([df_all, df], axis=0)

    df_all.to_csv('data/' + 'clean_' + file_name, index=False)

    logging.info("Finished reformatting {}".format(file_name))


def unzip_folder(folder_origin=None, folder_target=None):
    if not os.path.exists(folder_target):
        os.makedirs(folder_target)
        logging.info("Created new folder :{}".format(folder_target))

    files = [filo for filo in os.listdir(
        "data/" + folder_origin) if '.zip' in filo]
    for file in files:
        logging.info("unzipping {}".format(file))
        zip_ref = zipfile.ZipFile('data/' + folder_origin + file, 'r')
        zip_ref.extractall('data/' + folder_target)
        zip_ref.close()


def create_dataset(sample_period='1Min', create=False,
                   source_folder=None, file_name=None, filter_on=None):
    if create:
        _chunk_and_resample_folder(
            resample=sample_period, source_folder=source_folder)
    logging.info('Finished resampling folder : {}'.format(source_folder))
    df = _merge_files(source_folder=source_folder,
                      resample=sample_period, filter_on=filter_on)

    _add_calendar_data(df)
    df.to_csv('data/' + source_folder + '/merged/raw.csv')
    df = _remove_null(df, source_folder=source_folder)
    _regularize(df)

    df.to_csv('data/' + source_folder + '/merged/merged.csv', sep=';')
