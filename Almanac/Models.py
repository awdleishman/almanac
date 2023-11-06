import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def hw_weekly_frost_date_forecast(train, forecast_period=52, max_offset=7):
    """
    A function that predicts one year of weekly weather data using
        a modified Holt Winters model.


    Parameters:

    train: pandas.DataFrame
            A DataFrame object containing weather data
                        on which to train the model.

    forecast_period: int (optional)
            An integer representing the number of weeks that should be
            forecast. Defaults to 52 weeks (1 year).

    max_offset: int (optional)
            An integer representing the maximum allowed offset
            of the prediction.


    Returns:

    predicted: pandas.Series
            A Series object containing the
                        predicted weekly minimum temperature values.

    offset: float
            The offset that was used when tuning the model.
    """
    # If the data frequency is not weekly then resample it
    # before training the model.
    if type(train["tmin"].index.freq) is not pd._libs.tslibs.offsets.Week:
        train = train.resample("W").min()
        fitted_model = ExponentialSmoothing(
            train["tmin"][0:-9],
            trend="add",
            seasonal="add",
            seasonal_periods=52,
        ).fit()
        predicted = fitted_model.forecast(12)

    else:
        fitted_model = ExponentialSmoothing(
            train["tmin"][0:-9],
            trend="add",
            seasonal="add",
            seasonal_periods=52,
        ).fit()
        predicted = fitted_model.forecast(12)

    offset = np.arange(0, max_offset, 0.5)

    # Grab the last 9 weeks of data to use
    # for determining the offset.
    df_future = train["tmin"][-9:]

    # Calculate the RMSE between the new 9 week period
    # and the prediction - offset
    os_test = [
        mean_squared_error(
            df_future,
            predicted[df_future.index] - x,
            squared=False,
        )
        for x in offset
    ]
    if pd.Series(os_test).diff()[pd.Series(os_test).diff() > 0].empty:
        os_ind = len(offset) - 1
    else:
        os_ind = (
            pd.Series(os_test).diff()[pd.Series(os_test).diff() > 0].index[0]
        )
    """ The "correct" offset is choosen as the first offset that
        causes the RMSE to stop decreasing and start increasing
        to a max offset of 6.5.
        This should yield predictions with smaller RMSE
        than no offset and be less likely to be early in the prediction of
        the last frost of spring."""

    # Retrain the model on all of the training data
    # and forecast
    fitted_model = ExponentialSmoothing(
        train["tmin"],
        trend="add",
        seasonal="add",
        seasonal_periods=52,
    ).fit()
    predicted = fitted_model.forecast(forecast_period)

    predicted = predicted - offset[os_ind]
    # return the prediction series and the offset that was used.
    return predicted, offset[os_ind]
