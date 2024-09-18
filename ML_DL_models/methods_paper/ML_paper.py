import argparse
from datasets.PaperDataset import PaperDataset
from datetime import datetime
import numpy as np
import pandas as pd
import torchmetrics
from utils.generic_utils import load_correspondences
from utils.scikit_utils import get_ml_model, prepare_data_2_numpy


def compute_metrics_per_magnitude_nn_paper(predictions: np.ndarray, labels: np.ndarray, n_locs: int,
                                           n_times_predict: int, correspondences: dict, columns: list) -> tuple:

    n_magnitudes: int = 1

    predictions_tensor_form: np.ndarray = predictions.reshape(180 * 24 - 1, n_locs, n_times_predict, n_magnitudes)  # TODO: arreglar los numeritos

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


            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)

    return mae_list, mse_list, rmse_list


def main():

    parser = argparse.ArgumentParser(description='Description of my script')
    parser.add_argument('--sq_len_to_train', default=12, help='Sequence length to train')
    parser.add_argument('--sq_len_to_predict', default=12, help='Sequence length to predict')
    parser.add_argument('--model_type', default="tweedie", help='Model type')
    parser.add_argument('--interpolate', default="linear", help='Way to interpolate')
    parser.add_argument('--categorical', default=False, help='Categorical variable for windDir')

    args = parser.parse_args()
    print(args)

    args_dict: dict = vars(args)

    path_csv: str = "../Mad_Station/Mad_Station_2019.csv"
    path_csv_test: str = "../Mad_Station/Mad_Station_2022.csv"

    path_correspondences: str = "../correspondences/correspondencesPaper.csv"
    dict_correspondences: dict = load_correspondences(path_correspondences)

    n_locs: int = 24

    batch_size: int = 1

    train_tpd = PaperDataset(path=path_csv,
                             correspondences=dict_correspondences,
                             sq_len_to_train=args.sq_len_to_train,
                             sq_len_to_predict=args.sq_len_to_predict,
                             interpolate=args.interpolate,
                             transform=None,
                             normalization=True,
                             categorical=args.categorical
                             )

    data_, labels_, i = prepare_data_2_numpy(train_tpd, shuffle=True)

    print("Training...")
    init_time: str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    model = get_ml_model(model_type=args.model_type)
    model.fit(data_, labels_)

    print("Trained")

    model_name_2_save: str = args.model_type + '-' + init_time


    ## TEST
    tpd_test = PaperDataset(path=path_csv_test,
                            correspondences=dict_correspondences,
                            sq_len_to_train=args.sq_len_to_train,
                            sq_len_to_predict=args.sq_len_to_predict,
                            interpolate=args.interpolate,
                            transform=None,
                            normalization=True,
                            categorical=args.categorical
                            )

    data_test, labels_test, i = prepare_data_2_numpy(tpd_test, shuffle=False)

    predictions = model.predict(data_test)

    final_time: str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    print("Prediction:", predictions[0], "\n- Label:", labels_test[0])
    print()

    csv_test = pd.read_csv(path_csv_test)
    columns = csv_test.columns

    # np.save("preds1.npy", np.array(predictions))
    # np.save("labls1.npy", np.array(labels_test))

    mae_list, mse_list, rmse_list, _, _ = compute_metrics_per_magnitude_nn_paper(predictions=np.array(predictions),
                                                                                 labels=np.array(labels_test),
                                                                                 n_locs=n_locs,
                                                                                 n_times_predict=args.sq_len_to_predict,
                                                                                 correspondences=dict_correspondences,
                                                                                 columns=columns
                                                                                 )

    args_results: dict = {"mae_list": mae_list, "mse_list": mse_list, "rmse_list": rmse_list, "init_time": init_time,
                          "final_time": final_time}

    # results_saver(folder="../ML_results/", name_csv="info_results.csv",
    #               extra_columns=["mae_list", "mse_list", "rmse_list", "init_time", "final_time"], args_dict=args_dict,
    #               args_results=args_results, nn=False)


if __name__ == "__main__":
    main()
