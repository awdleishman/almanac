from sklearn.metrics import mean_absolute_error
from flytekit import task, workflow
from Almanac.Data import get_weather_data
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import papermill as pm
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
from traitlets.config import Config
import pickle
import uuid
import os


@task
def get_data(hyperparameters: dict) -> str:
    """Get the wine dataset."""
    print("fetching data")
    data = get_weather_data(
        hyperparameters["place"],
        hyperparameters["start"],
        hyperparameters["end"],
    )

    UUID = uuid.uuid4().hex
    dir = "artifacts/" + UUID + "/data/"
    file_name = dir + "full_data.pkl"
    if not os.path.exists(dir):
        os.makedirs(dir)

    pickle.dump(data, open(file_name, "wb"))

    return UUID


@task
def split_data(UUID: str, hyperparameters: dict) -> None:
    print("splitting data")
    file_name = "artifacts/" + UUID + "/data/full_data.pkl"
    data = pickle.load(open(file_name, "rb"))

    total_rows = len(data)
    train_rows = int(total_rows * hyperparameters["train_frac"])
    train_data = data.iloc[:train_rows, :].copy()

    test_rows = total_rows - train_rows
    if test_rows > 0:
        test_data = data.iloc[train_rows:, :].copy()
    else:
        test_data = data.iloc[:train_rows, :].copy()

    file_name = "artifacts/" + UUID + "/data/train.pkl"
    pickle.dump(train_data, open(file_name, "wb"))

    file_name = "artifacts/" + UUID + "/data/test.pkl"
    pickle.dump(test_data, open(file_name, "wb"))

    return


@task
def train_model(data_UUID: str, hyperparameters: dict) -> str:
    """Train a model on the wine dataset."""
    print("training model")

    file_name = "artifacts/" + data_UUID + "/data/train.pkl"
    train_data = pickle.load(open(file_name, "rb"))

    fitted_model = ExponentialSmoothing(
        train_data["tmin"],
        trend=hyperparameters["ES__trend"],
        seasonal=hyperparameters["ES__seasonal"],
        seasonal_periods=hyperparameters["ES__seasonal_periods"],
    ).fit()
    # features = train_data.iloc[2:5]
    # target = train_data[["tmin"]]
    # fitted_model = LinearRegression(max_iter=3000).fit(features, target)

    # save the model to disk

    model_UUID = uuid.uuid4().hex
    dir = "artifacts/" + data_UUID + "/models/"
    filename = dir + model_UUID + ".pkl"
    if not os.path.exists(dir):
        os.makedirs(dir)

    pickle.dump(fitted_model, open(filename, "wb"))

    return model_UUID


@task
def evaluate_model(
    data_UUID: str,
    model_UUID: str,
    hyperparameters: dict,
) -> None:
    """Evaluate model performance on train & test sets"""
    print("evaluating model")

    # file_name = 'artifacts/' + data_UUID + "/data/train.pkl"
    # train_data = pickle.load(open(file_name, 'rb'))

    file_name = "artifacts/" + data_UUID + "/data/test.pkl"
    test_data = pickle.load(open(file_name, "rb"))

    file_name = "artifacts/" + data_UUID + "/models/" + model_UUID + ".pkl"
    fitted_model = pickle.load(open(file_name, "rb"))

    n_pred = len(test_data)
    pred = fitted_model.forecast(n_pred)
    mae_test = mean_absolute_error(test_data["tmin"], pred)

    with open("results.txt", "w") as file1:
        # Writing data to a file
        file1.write(f"Mean Absolute Error: {mae_test} Celsius")

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


@task(disable_deck=False)
def viz_model(data_UUID: str, model_UUID: str, hyperparameters: dict) -> None:
    print("Running analysis notebook")

    hyperparameters["model_UUID"] = model_UUID
    hyperparameters["data_UUID"] = data_UUID

    pm.execute_notebook(
        "model_visualization.ipynb",
        "model_visualization_output.ipynb",
        parameters=hyperparameters,
        report_mode=True,
    )

    # Setup config
    c = Config()

    # Configure tag removal - be sure to tag your cells to remove  using the
    # words remove_cell to remove cells. You can also modify the code to use
    # a different tag word
    c.TagRemovePreprocessor.remove_input_tags = ("hide", "injected-parameters")
    c.TagRemovePreprocessor.enabled = True

    # Configure and run out exporter
    c.HTMLExporter.preprocessors = [
        "nbconvert.preprocessors.TagRemovePreprocessor"
    ]

    exporter = HTMLExporter(config=c)
    exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)

    output = HTMLExporter(config=c).from_filename(
        "model_visualization_output.ipynb"
    )

    # Write to output html file
    with open("output.html", "w") as f:
        f.write(output[0])


@workflow
def training_workflow(parameters: dict) -> None:
    """Put all of the steps together into a single workflow."""
    data_UUID = get_data(hyperparameters=parameters)
    split_data(UUID=data_UUID, hyperparameters=parameters)
    model_UUID = train_model(
        data_UUID=data_UUID,
        hyperparameters=parameters,
    )

    evaluate_model(
        data_UUID=data_UUID,
        model_UUID=model_UUID,
        hyperparameters=parameters,
    )

    viz_model(
        data_UUID=data_UUID,
        model_UUID=model_UUID,
        hyperparameters=parameters,
    )
    return
