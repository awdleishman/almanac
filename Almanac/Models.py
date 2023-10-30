from Almanac.Data import get_weather_data
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def hw_weekly_frost_date_forecast(train, location=None):
    """
    A function that predicts one year of weekly weather data using
        a modified Holt Winters model.


    Parameters:

    train: pandas.DataFrame
            A DataFrame object containing weather data
                        on which to train the model.

    location: str (optional)
            A string containing the location of the weather data
                        to allow tuning of the model
            using out of sample data.


    Returns:

    predicted: pandas.Series
            A Series object containing the
                        predicted weekly minimum temperature values.

    offset: float
            The offset that was used when tuning the model.
    """

    fitted_model = ExponentialSmoothing(
        train["tmin"].resample("W").min(),
        trend="add",
        seasonal="add",
        seasonal_periods=52,
    ).fit()
    predicted = fitted_model.forecast(52)

    if location is not None:
        # Create an array of offset values to test
        offset = np.arange(0, 4, 0.5)
        # Get weather data for 9 weeks after the training period
        df_future = get_weather_data(
            location,
            start=train.index[-1] + timedelta(weeks=1),
            end=train.index[-1] + timedelta(weeks=9),
        )
        # Calculate the RMSE between the new 9 week period
        # and the prediction - offset
        os_test = [
            mean_squared_error(
                df_future["tmin"].resample("W").min(),
                predicted[df_future["tmin"].resample("W").min().index] - x,
                squared=False,
            )
            for x in offset
        ]
        if pd.Series(os_test).diff()[pd.Series(os_test).diff() > 0].empty:
            os_ind = len(offset) - 1
        else:
            os_ind = (
                pd.Series(os_test)
                .diff()[pd.Series(os_test).diff() > 0]
                .index[0]
            )
        """ The "correct" offset is choosen as the first offset that
         causes the RMSE to stop decreasing and start increasing
         to a max offset of 3.5.
         This should yield predictions with smaller RMSE
         than no offset and be less likely to be early in the prediction of
         the last frost of spring."""
        predicted = predicted - offset[os_ind]
        # return the prediction series and the offset that was used.
        return predicted, offset[os_ind]
    else:
        return predicted
