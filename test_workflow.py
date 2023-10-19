import pandas as pd
from sklearn.metrics import mean_absolute_error
from flytekit import task, workflow
from Almanac.Data import get_weather_data
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@task
def get_data(parameters: dict) -> pd.DataFrame:
    """Get the wine dataset."""
    print("fetching data")
    data = get_weather_data(
        parameters["place"], parameters["start"], parameters["end"]
    )
    return data


@task
def train_split(data: pd.DataFrame, hyperparameters: dict) -> pd.DataFrame:
    print("splitting data")
    total_rows = len(data)
    train_rows = int(total_rows * hyperparameters["train_frac"])
    train_data = data.iloc[:train_rows, :].copy()
    return train_data


@task
def test_split(data: pd.DataFrame, hyperparameters: dict) -> pd.DataFrame:
    print("splitting data")
    total_rows = len(data)
    train_rows = int(total_rows * hyperparameters["train_frac"])
    test_rows = total_rows - train_rows
    if test_rows > 0:
        test_data = data.iloc[train_rows:, :].copy()
    else:
        test_data = data.iloc[:train_rows, :].copy()
    return test_data


@task
def train_model(
    train_data: pd.DataFrame, hyperparameters: dict
) -> ExponentialSmoothing:
    """Train a model on the wine dataset."""
    print("training model")
    fitted_model = ExponentialSmoothing(
        train_data["tmin"],
        trend=hyperparameters["ES__trend"],
        seasonal=hyperparameters["ES__seasonal"],
        seasonal_periods=hyperparameters["ES__seasonal_periods"],
    ).fit()
    # features = train_data.iloc[2:5]
    # target = train_data[["tmin"]]
    # fitted_model = LinearRegression(max_iter=3000).fit(features, target)
    return fitted_model


@task
def evaluate_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    fitted_model: ExponentialSmoothing,
    hyperparameters: dict,
) -> None:
    """Evaluate model performance on train & test sets"""
    print("evaluating model")
    n_pred = len(test_data)
    pred = fitted_model.forecast(n_pred)

    mae = mean_absolute_error(test_data["tmin"], pred)

    with open("results.txt", "w") as file1:
        # Writing data to a file
        file1.write(f"Mean Absolute Error: {mae} Celsius")

    return


# @task(disable_deck=False)
# def viz_model(
#     train_data: pd.DataFrame,
#     test_data: pd.DataFrame,
#     fitted_model: ExponentialSmoothing,
#     hyperparameters: dict
# ) -> None:
#     n_pred = len(test_data)
#     pred = fitted_model.forecast(n_pred)

#     fig = px.line(train_data, y='tmin')
#     # test_data['tmin'].plot(label='TEST', legend=True)
#     # pred.plot(label='PREDICTION', legend=True)
#     Deck("train data", plotly.io.to_html(fig))
#     main_deck = Deck("pca", MarkdownRenderer().to_html("### Principal
#  Component Analysis"))
#     main_deck.append(plotly.io.to_html(fig))

#     return main_deck


@workflow
def training_workflow(parameters: dict) -> None:
    """Put all of the steps together into a single workflow."""
    data = get_data(parameters=parameters)
    train_data = train_split(data=data, hyperparameters=parameters)
    test_data = test_split(data=data, hyperparameters=parameters)
    trained_model = train_model(
        train_data=train_data,
        hyperparameters=parameters,
    )

    evaluate_model(
        train_data=train_data,
        test_data=test_data,
        fitted_model=trained_model,
        hyperparameters=parameters,
    )

    # main_deck = viz_model(train_data=train_data,
    #                         test_data=test_data,
    #                fitted_model=trained_model,
    #                hyperparameters=parameters)

    return
