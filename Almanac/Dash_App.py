from dash import Dash, html, dcc
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px

from Almanac.Data import get_weather_data
from Almanac.Data import get_frost_dates

# from Almanac.Data import binarize
from Almanac.Models import hw_weekly_frost_date_forecast
from Almanac.Models import sarima_forecast
from datetime import timedelta
import time


def run_app():
    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.Div(
                "Location:",
                style={"display": "inline-block", "padding-right": 125},
            ),
            html.Div(
                "Start Date:",
                style={"display": "inline-block", "padding-right": 120},
            ),
            html.Div(
                "End Date:",
                style={"display": "inline-block", "padding-right": 100},
            ),
            html.Br(),
            # Location Input
            dcc.Input(
                id="location",
                type="text",
                placeholder="Enter Location",
                debounce=True,
                style={
                    "display": "inline-block",
                },
            ),
            html.Div(
                " ", style={"display": "inline-block", "padding-right": 10}
            ),
            # Start Date Input
            dcc.Input(
                id="start-date",
                type="text",
                placeholder="yyyy-mm-dd",
                debounce=True,
            ),
            html.Div(
                " ", style={"display": "inline-block", "padding-right": 10}
            ),
            # End Date Input
            dcc.Input(
                id="end-date",
                type="text",
                placeholder="yyyy-mm-dd",
                debounce=True,
            ),
            html.Br(),
            html.Div(
                id="text-output",
                style={
                    "display": "inline-block",
                    "width": 150,
                    "padding-right": 40,
                },
            ),
            html.Div(
                id="start-date-output",
                style={
                    "display": "inline-block",
                    "width": 150,
                    "padding-right": 40,
                },
            ),
            html.Div(
                id="end-date-output",
                style={"display": "inline-block", "width": 150},
            ),
            html.Br(),
            html.Br(),
            # Dropdown Menu
            "Model:",
            html.Br(),
            dcc.Dropdown(
                ["Holt Winters", "SARIMA"],
                "Holt Winters",
                id="model dropdown",
            ),
            html.Br(),
            dcc.Graph(id="fig1"),
            # Data Store
            dcc.Store(id="weather data"),
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
        Output("weather data", "data"),
        Input("location", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
    )
    def get_data(location, start, end):
        if (not location) or (not start) or (not end):
            raise PreventUpdate

        for i in range(3):
            try:
                data = get_weather_data(location, start, end)
            except Exception:
                print(f"Fetching Weather Data attempt {i+1}")
                time.sleep(3)
                continue

        return data.to_json(orient="split", date_format="iso")

    @app.callback(
        Output("fig1", "figure"),
        Input("weather data", "data"),
        Input("model dropdown", "value"),
        Input("location", "value"),
    )
    def generate_figure(data_json, model, location):
        if not data_json:
            raise PreventUpdate

        data = pd.read_json(data_json, orient="split")

        if model == "Holt Winters":
            prediction, os = hw_weekly_frost_date_forecast(data)
            first, last = get_frost_dates(pd.DataFrame({"tmin": prediction}))
            figure = px.line(x=prediction.index, y=prediction)
            figure.update_layout(
                title={
                    "text": (
                        f"Holt Winters Prediction Weekly Min Temperature "
                        f"{figure.data[0].x[0].strftime('%Y-%m-%d')}"
                        f" to {figure.data[0].x[-1].strftime('%Y-%m-%d')} "
                        f"for {location}"
                    ),
                    "x": 0.5,
                    "y": 0.95,
                    "xanchor": "center",
                },
                title_font={"color": "deepskyblue", "size": 17},
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            figure.update_xaxes(title="Date", showgrid=False)
            figure.update_yaxes(title="Temperature")
            figure.add_scatter(
                x=[first[0], last[0]],
                y=[prediction[first[0]], prediction[last[0]]],
                mode="markers",
                marker_symbol="star",
                marker_size=10,
                showlegend=False,
            )

            figure.add_annotation(
                x=first[0],
                y=prediction[first[0]],
                text="Fall Frost",
                showarrow=True,
                arrowhead=2,
                arrowcolor="black",
                axref="pixel",
                ax=20,
                xshift=5,
                yshift=5,
            )

            return figure

        if model == "SARIMA":
            df = data["tmin"].resample("W").min()
            prediction = sarima_forecast(
                df, start=df.index[-1], end=df.index[-1] + timedelta(days=365)
            )
            first, last = get_frost_dates(pd.DataFrame({"tmin": prediction}))
            figure = px.line(x=prediction.index, y=prediction)
            figure.update_layout(
                title={
                    "text": (
                        f"SARIMA Prediction Weekly Min Temperature "
                        f"{figure.data[0].x[0].strftime('%Y-%m-%d')}"
                        f" to {figure.data[0].x[-1].strftime('%Y-%m-%d')}"
                        f"for {location}"
                    ),
                    "x": 0.5,
                    "y": 0.95,
                    "xanchor": "center",
                },
                title_font={"color": "deepskyblue", "size": 17},
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            figure.update_xaxes(title="Date", showgrid=False)
            figure.update_yaxes(title="Temperature")

            figure.add_scatter(
                x=[first[0], last[0]],
                y=[prediction[first[0]], prediction[last[0]]],
                mode="markers",
                marker_symbol="star",
                marker_size=10,
                showlegend=False,
            )

            figure.add_annotation(
                x=first[0],
                y=prediction[first[0]],
                text="Fall Frost",
                showarrow=True,
                arrowhead=2,
                arrowcolor="black",
                axref="pixel",
                ax=20,
                xshift=5,
                yshift=5,
            )

            return figure


if __name__ == "__main__":
    run_app()
