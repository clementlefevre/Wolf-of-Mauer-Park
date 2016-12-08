from datetime import timedelta
import pandas as pd
import xgboost
from service.train_test import create_dataset
from model.models import Prediction


def xgbooster(dataset):
    model = xgboost.XGBRegressor()
    model.fit(dataset['training_X'], dataset['training_y'])
    print model
    predictions = model.predict(dataset['forecast_X'])
    print type(predictions)
    print dataset['observed_y'].shape
    # print accuracy_score(dataset['observed_y'], y_pred)
    prediction_df = pd.DataFrame({'predicted': predictions, 'observed': dataset[
                                 'observed_y']}, index=dataset['label_forecast'])

    return prediction_df


def make_prediction(df, target, date, shift):
    dataset = create_dataset(df, target, date, shift)
    df_prediction = xgbooster(dataset)
    return Prediction(df_prediction, target, date, shift)


def plot_prediction(df, target, date, j, fig, axes):
    title = date.strftime("%b-%d") + ":" + target
    df = df[1:]
    df = df[(df.index < (date + timedelta(days=20)))]

    if not df.empty:
        df.plot(ax=axes[j], title=title, lw=1)
