# from sklearn.metrics import mean_absolute_error
from Almanac.Data import get_weather_data
from Almanac.Models import sarima_forecast
import pandas as pd
import streamlit as st
import plotly.express as px

DEFAULT_DICT = {
    "place": "Chicago, IL",
    "start": "2019-01-01",
    "end": "2024-01-01",
    "train_frac": 0.8,
}

"""
# almanac
"""

params = DEFAULT_DICT

# Input parameters
st.text_input("Location", key="place", value=DEFAULT_DICT["place"])
params["place"] = st.session_state.place

st.text_input("Start Date", key="start", value=DEFAULT_DICT["start"])
params["start"] = st.session_state.start

st.text_input("End Date", key="end", value=DEFAULT_DICT["end"])
params["end"] = st.session_state.end


def get_data(hyperparameters: dict) -> pd.DataFrame:
    """Gets the local weather station data"""
    print("fetching data")
    data = get_weather_data(
        hyperparameters["place"],
        hyperparameters["start"],
        hyperparameters["end"],
    )

    data = data.resample("W").min()
    return data


data = get_data(hyperparameters=params)

"""
Temperature vs Time
"""
st.line_chart(data.iloc[:, 0:3])

total_rows = len(data)
train_rows = int(total_rows * params["train_frac"])
train_data = data.iloc[:train_rows, :].copy()

test_rows = total_rows - train_rows
if test_rows > 0:
    test_data = data.iloc[train_rows:, :].copy()
else:
    test_data = data.iloc[:train_rows, :].copy()

"""
Min Temp vs Time (Training)
"""

st.line_chart(train_data.loc[:, "tmin"])


sarima_config = ((3, 0, 0), (0, 1, 1, 52), ("c"))

fitted_model = sarima_forecast(
    data=train_data["tmin"],
    config=sarima_config,
)

"""
Min Temp vs Time (Testing)
"""

n_pred = len(test_data)
pred = fitted_model.forecast(n_pred)
test_data["prediction"] = pred

st.line_chart(test_data.loc[:, ["tmin", "prediction"]])

fig = px.line(test_data, y="tmin")

st.plotly_chart(fig)
