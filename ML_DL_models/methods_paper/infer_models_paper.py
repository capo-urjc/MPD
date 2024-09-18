import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from datasets.PaperDataset import PaperDataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchmetrics
from utils.generic_utils import load_correspondences
from utils.scikit_utils import get_nn_model
from utils_os.results_saver import results_saver


def compute_metrics_per_magnitude_nn_paper(predictions: np.ndarray, labels: np.ndarray, n_locs: int,
                                           n_times_predict: int, correspondences: dict, columns: list) -> tuple:

    n_magnitudes: int = 1

    predictions_tensor_form: np.ndarray = predictions.reshape(180 * 24 - 1, n_locs, n_times_predict, n_magnitudes) # TODO: arreglar los numeritos

    labels_tensor_form = labels.reshape(180 * 24 - 1, n_locs, n_times_predict, n_magnitudes)

    mae_list: list = []
    mse_list: list = []
    rmse_list: list = []

    for j, column in enumerate(columns):
        if column in correspondences["MAGNITUD"]:
            idx = correspondences["MAGNITUD"].index(column)

            max_value: float = correspondences["MAXIMO"][idx]
            min_value: float = correspondences["MINIMO"][idx]

            preds = predictions_tensor_form * (max_value - min_value) + min_value
            labls = labels_tensor_form * (max_value - min_value) + min_value

            MAE: torchmetrics = torchmetrics.MeanAbsoluteError()
            MSE: torchmetrics = torchmetrics.MeanSquaredError()

            mae = np.abs(preds[..., j] - labls[..., j]).mean()
            mse = ((preds[..., j] - labls[..., j])**2).mean()

            rmse = np.sqrt(mse)
            print(f"MAE for magnitude {column} is {mae}")
            print(f"MSE for magnitude {column} is {mse}")
            print(f"RMSE for magnitude {column} is {rmse}")

            mae = MAE(torch.from_numpy(preds[..., j]), torch.from_numpy(labls[..., j]))
            mse = MSE(torch.from_numpy(preds[..., j]), torch.from_numpy(labls[..., j]))
            rmse = np.sqrt(mse)

            print(100*'-')

            print(f"MAE for magnitude {column} is {mae}")
            print(f"MSE for magnitude {column} is {mse}")
            print(f"RMSE for magnitude {column} is {rmse}")

            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)

        break

    return mae_list, mse_list, rmse_list


def test(path_csv_test: str, path_correspondences: str, path_model: str, sq_len_to_train: int,
         sq_len_to_predict: int, interpolate: str, model_type: str, device: str, categorical: bool):

    dict_correspondences: dict = load_correspondences(path_correspondences)

    test_tpd = PaperDataset(path=path_csv_test,
                            correspondences=dict_correspondences,
                            sq_len_to_train=sq_len_to_train,
                            sq_len_to_predict=sq_len_to_predict,
                            interpolate=interpolate,
                            transform=None,
                            normalization=True,
                            categorical=categorical
                            )

    test_dataloader = DataLoader(test_tpd, batch_size=1, shuffle=False)

    if args.categorical:
        input_dim = 18 * args.sq_len_to_train * 24
    else:
        input_dim = 11 * args.sq_len_to_train * 24

    output_dim = 1 * args.sq_len_to_predict * 24

    model = get_nn_model(model_type=model_type, input_dim=input_dim, hidden_dim=512, output_dim=output_dim,
                         device=device)

    model_st = torch.load(path_model, map_location=torch.device(device))

    model.load_state_dict(model_st["model_st_dict"])

    inference(path_csv_test=path_csv_test, model=model, test_dataloader=test_dataloader, n_locs=24,
              dict_correspondences=dict_correspondences, device=device)


def inference(path_csv_test: str, model, test_dataloader, n_locs, dict_correspondences, device):
    model.eval()

    csv_test = pd.read_csv(path_csv_test)
    columns = csv_test.columns

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            x_batch = sample['x'].float().to(device)
            y_batch = sample['y'].float().to(device)
            output = model(x_batch)

            output_np = output.cpu().detach().numpy()

            all_predictions.append(output_np)
            all_labels.append(y_batch.view(output_np.shape).cpu().detach().numpy())


    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)

    print(all_predictions.shape)
    print(all_labels.shape)

    mae_list, mse_list, rmse_list = compute_metrics_per_magnitude_nn_paper(predictions=all_predictions,
                                                                           columns=columns,
                                                                           labels=all_labels,
                                                                           n_locs=n_locs,
                                                                           n_times_predict=args.sq_len_to_predict,
                                                                           correspondences=dict_correspondences
                                                                           )

    args_results: dict = {"mae_list": mae_list, "mse_list": mse_list, "rmse_list": rmse_list}

    results_saver(folder="../NN_paper_results/", name_csv="info_results_paper.csv",
                  extra_columns=["mae_list", "mse_list", "rmse_list"], args_dict=args_dict,
                  args_results=args_results, nn=False)


def main(path_csv_test: str, path_correspondences: str, path_model: str, sq_len_to_train: int,
         sq_len_to_predict: int, interpolate: str, model_type: str, device: str, categorical: bool):

    test(path_csv_test=path_csv_test, path_correspondences=path_correspondences, path_model=path_model,
         sq_len_to_train=sq_len_to_train, sq_len_to_predict=sq_len_to_predict, interpolate=interpolate,
         model_type=model_type, device=device, categorical=categorical)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of my script')
    # parser.add_argument('--batch_size', default=8, help='Batch size')
    parser.add_argument('--sq_len_to_train', default=12, help='Sequence length to train')
    parser.add_argument('--sq_len_to_predict', default=12, help='Sequence length to predict')
    parser.add_argument('--model_type', default="model_nn", help='Model type')
    parser.add_argument('--interpolate', default="linear", help='Way to interpolate')
    parser.add_argument('--device', default="cuda", help='Device')
    parser.add_argument('--categorical', default=False, help='Categorical variable for windDir')

    args = parser.parse_args()
    print(args)

    args_dict: dict = vars(args)

    path_csv_test: str = "../Mad_Station/Mad_Station_2022.csv"
    path_correspondences: str = "../correspondences/correspondencesPaper.csv"

    path_model: str = "../logs/PaperDatasigm_250_32_1e-06_l1_adam_12_12_model_nn_linear_False_cuda/version_1/checkpoints/epoch-8.pth"

    main(path_csv_test=path_csv_test,
         path_correspondences=path_correspondences,
         path_model=path_model,
         sq_len_to_train=args.sq_len_to_train,
         sq_len_to_predict=args.sq_len_to_predict,
         interpolate=args.interpolate,
         model_type=args.model_type,
         device=args.device,
         categorical=args.categorical
         )
