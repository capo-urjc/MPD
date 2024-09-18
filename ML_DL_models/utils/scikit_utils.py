from models.NNs import MLPRegression, MLPRegressionNN
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor


def prepare_data_2_numpy(dataset, shuffle: bool) -> tuple:
    data: list = []
    labels: list = []

    print("\n Loading data")
    for i in tqdm(range(len(dataset))):
        # print(i)
        batch = dataset[i]
        inputs_, targets_ = batch['x'], batch['y']
        # inputs = inputs_.reshape(inputs_.size(0), -1)
        inputs = np.reshape(inputs_, -1)  # aplanar los datos
        # targets = targets_.view(targets_.size(0), -1)
        targets = np.reshape(targets_, -1)  # aplanar las etiquetas
        # data.append(inputs.numpy())
        data.append(inputs)
        # labels.append(targets.numpy())
        labels.append(targets)

    # Shuffle data
    combined_data = list(zip(data, labels))

    if shuffle:
        random.shuffle(combined_data)

    new_data, new_labels = zip(*combined_data)

    return new_data, new_labels, i


def get_ml_model(model_type: str):
    """

    :param model_type:
    :return:

    May 2024
    """

    model_type = model_type.lower()
    assert model_type in ["svm", "linear", "tweedie", "xgb", "rf"], "model_type is not valid"

    if model_type == "svm":
        # model = MultiOutputRegressor(SVR(kernel="poly", degree=4, C=1, tol=1e-4, epsilon=0.01))
        model = MultiOutputRegressor(SVR(kernel="rbf"))
    elif model_type == "linear":
        model = LinearRegression()
    elif model_type == "tweedie":
        model = MultiOutputRegressor(TweedieRegressor(alpha=0.01, max_iter=500))
    elif model_type == "xgb":
        model = XGBRegressor(n_estimators=90, max_depth=2, learning_rate=0.05, objective="reg:absoluteerror", gamma=0)
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, max_depth=16, verbose=2)

    return model


def get_nn_model(model_type: str, input_dim: int, hidden_dim: int, output_dim: int, device: str):
    if model_type == "model1":
        model = MLPRegression(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).float().to(device)
    if model_type == "model_nn":
        model = MLPRegressionNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).float().to(device)
    return model


if __name__ == "__main__":
    import numpy as np

    # Datos de ejemplo
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    Y = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

    model = get_ml_model("svm")

    model.fit(X, Y)

    predictions = model.predict(X)
    print(predictions)


