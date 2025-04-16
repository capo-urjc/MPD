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
    r_list: list = []
    relative_error_list: list = []

    preds = predictions_tensor_form
    labls = labels_tensor_form

    for j, magnitude in enumerate(magnitudes_to_predict):
        print(j)
        idx_cod = correspondences["CODIFICACION"].index(magnitude)

        min_pos_value: int = correspondences["MINIMO"][idx_cod]
        max_pos_value: int = correspondences["MAXIMO"][idx_cod]

        # [0, 1]
        # preds[:, :, :, j] = predictions_tensor_form[:, :, :, j] * (max_pos_value - min_pos_value) + min_pos_value
        # labls[:, :, :, j] = labels_tensor_form[:, :, :, j] * (max_pos_value - min_pos_value) + min_pos_value

        # [-1, 1]
        preds[:, :, :, j] = ((predictions_tensor_form[:, :, :, j] + 1) * (max_pos_value - min_pos_value)) / 2 + min_pos_value
        labls[:, :, :, j] = ((labels_tensor_form[:, :, :, j] + 1) * (max_pos_value - min_pos_value)) / 2 + min_pos_value

        mae = np.abs(preds[..., j] - labls[..., j]).mean()
        mse = ((preds[..., j] - labls[..., j]) ** 2).mean()

        # R coeff
        numerator = np.sum((labls[..., j] - labls[..., j].mean()) * (preds[..., j] - preds[..., j].mean()))
        denominator = np.sqrt(np.sum((labls[..., j] - labls[..., j].mean()) ** 2) * np.sum((preds[..., j] - preds[..., j].mean()) ** 2))
        r = numerator / denominator if denominator != 0 else 0

        relative_error = np.where(labls[..., j] != 0, np.abs((preds[..., j] - labls[..., j]) / labls[..., j]), 0)
        mean_relative_error = relative_error.mean()

        rmse = np.sqrt(mse)
        print(f"MAE for magnitude {magnitude} is {mae}")
        print(f"MSE for magnitude {magnitude} is {mse}")
        print(f"RMSE for magnitude {magnitude} is {rmse}")
        print(f"R for magnitude {magnitude} is {r}")
        print(f"RMEAN for magnitude {magnitude} is {mean_relative_error}")

        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r_list.append(r)
        relative_error_list.append(mean_relative_error)

    return mae_list, mse_list, rmse_list, r_list, mean_relative_error, preds, labls


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
    r_list: list = []
    relative_error_list: list = []

    preds = predictions_tensor_form
    labls = labels_tensor_form

    for j, magnitude in enumerate(magnitudes_to_predict):
        idx_cod = correspondences["CODIFICACION"].index(magnitude)

        min_pos_value: int = correspondences["MINIMO"][idx_cod]
        max_pos_value: int = correspondences["MAXIMO"][idx_cod]

        # preds = predictions_tensor_form * (max_pos_value - min_pos_value) + min_pos_value
        # labls = labels_tensor_form * (max_pos_value - min_pos_value) + min_pos_value

        # [0, 1]
        # preds[:, :, :, j] = predictions_tensor_form[:, :, :, j] * (max_pos_value - min_pos_value) + min_pos_value
        # labls[:, :, :, j] = labels_tensor_form[:, :, :, j] * (max_pos_value - min_pos_value) + min_pos_value

        # [-1, 1]
        preds[:, :, :, j] = ((predictions_tensor_form[:, :, :, j] + 1) * (max_pos_value - min_pos_value)) / 2 + min_pos_value
        labls[:, :, :, j] = ((labels_tensor_form[:, :, :, j] + 1) * (max_pos_value - min_pos_value)) / 2 + min_pos_value

        mae = np.abs(preds[..., j] - labls[..., j]).mean()
        mse = ((preds[..., j] - labls[..., j]) ** 2).mean()

        # R coeff
        numerator = np.sum((labls[..., j] - labls[..., j].mean()) * (preds[..., j] - preds[..., j].mean()))
        denominator = np.sqrt(np.sum((labls[..., j] - labls[..., j].mean()) ** 2) * np.sum((preds[..., j] - preds[..., j].mean()) ** 2))
        r = numerator / denominator if denominator != 0 else 0

        relative_error = np.where(labls[..., j] != 0, np.abs((preds[..., j] - labls[..., j]) / labls[..., j]), 0)
        mean_relative_error = relative_error.mean()

        rmse = np.sqrt(mse)

        print(f"MAE for magnitude {magnitude} is {mae}")
        print(f"MSE for magnitude {magnitude} is {mse}")
        print(f"RMSE for magnitude {magnitude} is {rmse}")
        print(f"R for magnitude {magnitude} is {r}")
        print(f"RMEAN for magnitude {magnitude} is {mean_relative_error}")

        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r_list.append(r)
        relative_error_list.append(mean_relative_error)

    return mae_list, mse_list, rmse_list, r_list, relative_error_list
