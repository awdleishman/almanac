import pandas as pd
from sklearn.linear_model import LogisticRegression

from flytekit import task, workflow

from almanext.get_weather_data import get_weather_data


@task
def get_data(parameters: dict) -> pd.DataFrame:
    """Get the wine dataset."""
    data = get_weather_data(
        parameters["place"], parameters["start"], parameters["end"]
    )
    return data


@task
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Simplify the task from a 3-class to a binary classification problem."""
    return data.assign(target=lambda x: x["target"].where(x["target"] == 0, 1))


@task
def train_model(
    data: pd.DataFrame, hyperparameters: dict
) -> LogisticRegression:
    """Train a model on the wine dataset."""
    features = data.drop("target", axis="columns")
    target = data["target"]
    return LogisticRegression(max_iter=3000, **hyperparameters).fit(
        features, target
    )


@workflow
def training_workflow(parameters: dict) -> LogisticRegression:
    """Put all of the steps together into a single workflow."""
    data = get_data(parameters=parameters)
    processed_data = process_data(data=data)
    return train_model(
        data=processed_data,
        hyperparameters=parameters,
    )
