from dash import Dash, html, dcc
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate

# import pandas as pd
import plotly.express as px

from Almanac.Data import get_weather_data

# from Almanac.Data import get_frost_dates
# from Almanac.Data import binarize
from Almanac.Models import hw_weekly_frost_date_forecast
from Almanac.Models import sarima_forecast
from datetime import timedelta
import time


def run_app():
    app = Dash(__name__)

    app.layout = html.Div(
        [
            "Location:",
            html.Br(),
            dcc.Input(
                id="location",
                type="text",
                placeholder="Enter Location",
                debounce=True,
            ),
            html.Div(id="text-output"),
            html.Br(),
            # html.Div(id="hidden location",style={"display":"none"}),
            "Start Date:",
            html.Br(),
            dcc.Input(
                id="start-date",
                type="text",
                placeholder="yyyy-mm-dd",
                debounce=True,
            ),
            html.Div(id="start-date-output"),
            html.Br(),
            "End Date:",
            html.Br(),
            dcc.Input(
                id="end-date",
                type="text",
                placeholder="yyyy-mm-dd",
                debounce=True,
            ),
            html.Div(id="end-date-output"),
            html.Br(),
            "Model:",
            html.Br(),
            dcc.Dropdown(
                ["Holt Winters", "SARIMA"],
                "Holt Winters",
                id="model dropdown",
            ),
            html.Br(),
            dcc.Graph(id="fig1"),
        ]
    )

    app.run()

    @app.callback(
        Output("text-output", "children"), Input("location", "value")
    )
    def update_location_div(location):
        if not location:
            raise PreventUpdate
        return f"Location Selected: {location}"

    @app.callback(
        Output("start-date-output", "children"), Input("start-date", "value")
    )
    def update_start_div(start):
        if not start:
            raise PreventUpdate
        return f"Start date Selected: {start}"

    @app.callback(
        Output("end-date-output", "children"), Input("end-date", "value")
    )
    def update_end_div(end):
        if not end:
            raise PreventUpdate
        return f"End date Selected: {end}"

    @app.callback(
        Output("fig1", "figure"),
        Input("location", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
        Input("model dropdown", "value"),
        # Input("hidden location","data")
    )
    def generate_figure(location, start, end, model):
        #     print(f"{location}, {start}, {end}")
        if not end:
            raise PreventUpdate

        for i in range(3):
            try:
                data = get_weather_data(location, start, end)
            except Exception:
                print(f"Fetching Weather Data attempt {i+1}")
                time.sleep(3)
                continue

        if model == "Holt Winters":
            prediction, os = hw_weekly_frost_date_forecast(data)
            figure = px.line(x=prediction.index, y=prediction)
            figure.update_layout(
                title={
                    "text": (
                        f"Holt Winters Prediction Weekly Min Temperature"
                        f" {figure.data[0].x[0].strftime('%Y-%m-%d')}"
                        f" to {figure.data[0].x[-1].strftime('%Y-%m-%d')}"
                        f" for {location}"
                    ),
                    "x": 0.5,
                    "y": 0.95,
                    "xanchor": "center",
                },
                title_font={"color": "deepskyblue", "size": 17},
            )
            figure.update_xaxes(title="Date")
            figure.update_yaxes(title="Temperature")
            return figure

        if model == "SARIMA":
            df = data["tmin"].resample("W").min()
            prediction = sarima_forecast(
                df, start=df.index[-1], end=df.index[-1] + timedelta(days=365)
            )

            figure = px.line(x=prediction.index, y=prediction)
            figure.update_layout(
                title={
                    "text": (
                        f"SARIMA Prediction Weekly Min Temperature"
                        f" {figure.data[0].x[0].strftime('%Y-%m-%d')}"
                        f" to {figure.data[0].x[-1].strftime('%Y-%m-%d')}"
                        f"for {location}"
                    ),
                    "x": 0.5,
                    "y": 0.95,
                    "xanchor": "center",
                },
                title_font={"color": "deepskyblue", "size": 17},
            )
            figure.update_xaxes(title="Date")
            figure.update_yaxes(title="Temperature")
            return figure


if __name__ == "__main__":
    run_app()
