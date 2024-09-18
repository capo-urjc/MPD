import numpy as np
import torch
import torchmetrics


def compute_metrics_per_magnitude(predictions: np.ndarray, labels: np.ndarray, i: int, batch_size: int, n_locs: int,
                                  n_times_predict: int, magnitudes_to_predict: list, correspondences: dict) -> tuple:

    n_magnitudes: int = len(magnitudes_to_predict)

    predictions_tensor_form: np.ndarray = predictions.reshape(i, n_locs, n_times_predict, n_magnitudes+2)
    # labels_tensor_form: np.ndarray = labels.reshape(i, n_locs, n_times_predict, n_magnitudes)

    labels_np = np.array(labels)
    labels_tensor_form = labels_np.reshape(i, n_locs, n_times_predict, n_magnitudes+2)

    mae_list: list = []
    mse_list: list = []
    rmse_list: list = []

    preds = predictions_tensor_form
    labls = labels_tensor_form

    for j, magnitude in enumerate(magnitudes_to_predict):
        idx_cod = correspondences["CODIFICACION"].index(magnitude)

        min_pos_value: int = correspondences["MINIMO"][idx_cod]
        max_pos_value: int = correspondences["MAXIMO"][idx_cod]

        preds[:, :, :, j] = predictions_tensor_form[:, :, :, j] * (max_pos_value - min_pos_value) + min_pos_value
        labls[:, :, :, j] = labels_tensor_form[:, :, :, j] * (max_pos_value - min_pos_value) + min_pos_value

        mae = np.abs(preds[..., j] - labls[..., j]).mean()
        mse = ((preds[..., j] - labls[..., j]) ** 2).mean()

        rmse = np.sqrt(mse)
        print(f"MAE for magnitude {magnitude} is {mae}")
        print(f"MSE for magnitude {magnitude} is {mse}")
        print(f"RMSE for magnitude {magnitude} is {rmse}")


        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)

    return mae_list, mse_list, rmse_list, preds, labls


def compute_metrics_per_magnitude_nn(predictions: np.ndarray, labels: np.ndarray, n_locs: int, n_times_predict: int,
                                     magnitudes_to_predict: list, correspondences: dict) -> tuple:

    n_magnitudes: int = len(magnitudes_to_predict)

    predictions_tensor_form: np.ndarray = predictions.reshape(8736, n_locs, n_times_predict, n_magnitudes+2)
    # labels_tensor_form: np.ndarray = labels.reshape(i, n_locs, n_times_predict, n_magnitudes)

    # labels_np = np.array(labels)
    labels_tensor_form = labels.reshape(8736, n_locs, n_times_predict, n_magnitudes+2)

    mae_list: list = []
    mse_list: list = []
    rmse_list: list = []

    for j, magnitude in enumerate(magnitudes_to_predict):
        idx_cod = correspondences["CODIFICACION"].index(magnitude)

        min_pos_value: int = correspondences["MINIMO"][idx_cod]
        max_pos_value: int = correspondences["MAXIMO"][idx_cod]

        preds = predictions_tensor_form * (max_pos_value - min_pos_value) + min_pos_value
        labls = labels_tensor_form * (max_pos_value - min_pos_value) + min_pos_value

        # mae = np.abs(preds[..., j] - labls[..., j]).sum(axis=(1, 2)).mean()
        # mse = ((preds[..., j] - labls[..., j])**2).sum(axis=(1, 2)).mean()
        # rmse = np.sqrt(mse)

        mae = np.abs(preds[..., j] - labls[..., j]).mean()
        mse = ((preds[..., j] - labls[..., j]) ** 2).mean()

        rmse = np.sqrt(mse)

        print(f"MAE for magnitude {magnitude} is {mae}")
        print(f"MSE for magnitude {magnitude} is {mse}")
        print(f"RMSE for magnitude {magnitude} is {rmse}")

        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)

    return mae_list, mse_list, rmse_list
