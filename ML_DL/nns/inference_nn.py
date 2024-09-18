import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
from datasets.FastPollutionDataset import FastTemporalPollutionDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.generic_utils import load_correspondences
from utils.metrics_utils import compute_metrics_per_magnitude_nn
from utils.scikit_utils import get_nn_model
from utils_os.results_saver import results_saver


def test(path_csv_test: str, path_correspondences: str, path_model: str, batch_size: int, sq_len_to_train: int,
         sq_len_to_predict: int, magnitudes_to_train: list, magnitudes_to_predict: list, locations_to_train: list,
         locations_to_predict: list, interpolate: str, model_type: str, device: str):

    dict_correspondences: dict = load_correspondences(path_correspondences)

    test_tpd = FastTemporalPollutionDataset(path=path_csv_test,
                                            correspondences=dict_correspondences,
                                            sq_len_to_train=sq_len_to_train,
                                            sq_len_to_predict=sq_len_to_predict,
                                            magnitudes_to_train=magnitudes_to_train,
                                            magnitudes_to_predict=magnitudes_to_predict,
                                            locations_to_train=locations_to_train,
                                            locations_to_predict=locations_to_predict,
                                            interpolate=interpolate,
                                            transform=None,
                                            )

    test_dataloader = DataLoader(test_tpd, batch_size=batch_size, shuffle=False)

    input_dim = (len(args.magnitudes_to_train) + 2) * args.sq_len_to_train * len(args.locations_to_train)
    output_dim = (len(args.magnitudes_to_predict) + 2) * args.sq_len_to_predict * len(args.locations_to_predict)

    model = get_nn_model(model_type=args.model_type, input_dim=input_dim, hidden_dim=512, output_dim=output_dim,
                         device=args.device)

    model_st = torch.load(path_model, map_location=torch.device(device))

    model.load_state_dict(model_st["model_st_dict"])

    inference(model=model, test_dataloader=test_dataloader, n_locs=len(locations_to_predict),
              sq_len_to_predict=sq_len_to_predict, dict_correspondences=dict_correspondences, device=device)


def inference(model, test_dataloader, n_locs, sq_len_to_predict, dict_correspondences, device):
    model = model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            # print(f"Inferring batch number {i}")
            x_batch = sample['x'].float().to(device)
            y_batch = sample['y'].float().to(device)
            output = model(x_batch)

            output_np = output.cpu().detach().numpy().reshape(1, n_locs, sq_len_to_predict, -1)

            all_predictions.append(output_np)
            all_labels.append(y_batch.view(output_np.shape).cpu().detach().numpy())

    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)

    print(all_predictions.shape)
    print(all_labels.shape)

    mae_list, mse_list, rmse_list = compute_metrics_per_magnitude_nn(predictions=all_predictions,
                                                                     labels=all_labels,
                                                                     n_locs=n_locs,
                                                                     n_times_predict=args.sq_len_to_predict,
                                                                     magnitudes_to_predict=args.magnitudes_to_predict,
                                                                     correspondences=dict_correspondences
                                                                     )

    args_results: dict = {"mae_list": mae_list, "mse_list": mse_list, "rmse_list": rmse_list}

    results_saver(folder="../NN_results/", name_csv="info_results.csv",
                  extra_columns=["mae_list", "mse_list", "rmse_list"], args_dict=args_dict,
                  args_results=args_results, nn=False)


def main(path_csv_test: str, path_correspondences: str, path_model: str, batch_size: int, sq_len_to_train: int,
         sq_len_to_predict: int, magnitudes_to_train: list, magnitudes_to_predict: list, locations_to_train: list,
         locations_to_predict: list, interpolate: str, model_type: str, device: str):

    test(path_csv_test=path_csv_test, path_correspondences=path_correspondences, path_model=path_model,
         batch_size=batch_size, sq_len_to_train=sq_len_to_train, sq_len_to_predict=sq_len_to_predict,
         magnitudes_to_train=magnitudes_to_train, magnitudes_to_predict=magnitudes_to_predict,
         locations_to_train=locations_to_train, locations_to_predict=locations_to_predict, interpolate=interpolate,
         model_type=model_type, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of my script')
    parser.add_argument('--batch_size', default=1, help='Batch size')
    parser.add_argument('--magnitudes_to_train', default=[14, 81, 82, 83, 86, 87, 88], help='Magnitudes to train')
    parser.add_argument('--magnitudes_to_predict', default=[14], help='Magnitudes to predict')
    parser.add_argument('--locations_to_train', default=[(120, 1)], help='Locations to train')
    parser.add_argument('--locations_to_predict', default=[(120, 1)], help='Locations to predict')
    parser.add_argument('--sq_len_to_train', default=12, help='Sequence length to train')
    parser.add_argument('--sq_len_to_predict', default=12, help='Sequence length to predict')
    parser.add_argument('--model_type', default="model1", help='Model type')
    parser.add_argument('--interpolate', default="linear", help='Way to interpolate')
    parser.add_argument('--device', default="cuda", help='Device')

    args = parser.parse_args()
    print(args)

    args_dict: dict = vars(args)

    path_csv_test: str = "../DATA/2023_24.csv"
    path_correspondences: str = "../correspondences/correspondences.csv"

    path_model: str = "../logs/MyData_350_0.0001_32_l1_adam_[14, 81, 82, 83, 86, 87, 88]_[14]_12_12_model1_linear_cuda/version_1/checkpoints/epoch-5.pth"

    main(path_csv_test=path_csv_test,
         path_correspondences=path_correspondences,
         path_model=path_model,
         batch_size=args.batch_size,
         sq_len_to_train=args.sq_len_to_train,
         sq_len_to_predict=args.sq_len_to_predict,
         magnitudes_to_train=args.magnitudes_to_train,
         magnitudes_to_predict=args.magnitudes_to_predict,
         locations_to_train=args.locations_to_train,
         locations_to_predict=args.locations_to_predict,
         interpolate=args.interpolate,
         model_type=args.model_type,
         device=args.device
         )
