import argparse
from datasets.FastPollutionDataset import FastTemporalPollutionDataset
from datetime import datetime
from utils_os.results_saver import results_saver
from utils.generic_utils import load_correspondences
from utils.metrics_utils import compute_metrics_per_magnitude
from utils.scikit_utils import get_ml_model, prepare_data_2_numpy


def main():
    parser = argparse.ArgumentParser(description='Description of my script')
    parser.add_argument('--magnitudes_to_train', default='[44, 81, 82, 83, 86, 87, 88]', help='Magnitudes to train')
    parser.add_argument('--magnitudes_to_predict', default='[44]', help='Magnitudes to predict')
    parser.add_argument('--locations_to_train', default='[(16, 1)]', help='Locations to train')
    parser.add_argument('--locations_to_predict', default='[(16, 1)]', help='Locations to predict')
    parser.add_argument('--sq_len_to_train', default=12, help='Sequence length to train')
    parser.add_argument('--sq_len_to_predict', default=12, help='Sequence length to predict')
    parser.add_argument('--model_type', default="tweedie", help='Model type')
    parser.add_argument('--interpolate', default="linear", help='Way to interpolate')

    args = parser.parse_args()
    print(args)

    args.sq_len_to_train = int(args.sq_len_to_train)
    args.sq_len_to_predict = int(args.sq_len_to_predict)
    args.magnitudes_to_train = eval(args.magnitudes_to_train)
    args.magnitudes_to_predict = eval(args.magnitudes_to_predict)
    args.locations_to_train = eval(args.locations_to_train)
    args.locations_to_predict = eval(args.locations_to_predict)

    args_dict: dict = vars(args)

    path_csv: str = "../DATA/2022_24.csv"
    path_csv_test: str = "../DATA/2023_24.csv"
    path_correspondences: str = "../correspondences/correspondences.csv"

    dict_correspondences: dict = load_correspondences(path_correspondences)

    n_locs: int = len(args.locations_to_predict)

    batch_size: int = 1

    # composed, _, _ = get_transforms(nn=False, precision=16)

    tpd = FastTemporalPollutionDataset(path=path_csv,
                                       correspondences=dict_correspondences,
                                       sq_len_to_train=args.sq_len_to_train,
                                       sq_len_to_predict=args.sq_len_to_predict,
                                       magnitudes_to_train=args.magnitudes_to_train,
                                       magnitudes_to_predict=args.magnitudes_to_predict,
                                       locations_to_train=args.locations_to_train,
                                       locations_to_predict=args.locations_to_predict,
                                       interpolate=args.interpolate,
                                       transform=None,
                                       )

    tpd_test = FastTemporalPollutionDataset(path=path_csv_test,
                                            correspondences=dict_correspondences,
                                            sq_len_to_train=args.sq_len_to_train,
                                            sq_len_to_predict=args.sq_len_to_predict,
                                            magnitudes_to_train=args.magnitudes_to_train,
                                            magnitudes_to_predict=args.magnitudes_to_predict,
                                            locations_to_train=args.locations_to_train,
                                            locations_to_predict=args.locations_to_predict,
                                            interpolate=args.interpolate,
                                            transform=None
                                            )

    data_, labels_, i = prepare_data_2_numpy(tpd, shuffle=True)

    print("Training...")
    init_time: str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    model = get_ml_model(model_type=args.model_type)
    model.fit(data_, labels_)

    print("Trained")


    ## TEST
    data_test, labels_test, i = prepare_data_2_numpy(tpd_test, shuffle=False)

    predictions = model.predict(data_test)

    final_time: str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    mae_list, mse_list, rmse_list, _, _ = compute_metrics_per_magnitude(predictions=predictions,
                                                                        labels=labels_test,
                                                                        i=i + 1,
                                                                        batch_size=batch_size,
                                                                        n_locs=n_locs,
                                                                        n_times_predict=args.sq_len_to_predict,
                                                                        magnitudes_to_predict=args.magnitudes_to_predict,
                                                                        correspondences=dict_correspondences
                                                                        )

    args_results: dict = {"mae_list": mae_list, "mse_list": mse_list, "rmse_list": rmse_list, "init_time": init_time,
                          "final_time": final_time}

    results_saver(folder="../ML_results/", name_csv="info_results.csv",
                  extra_columns=["mae_list", "mse_list", "rmse_list", "init_time", "final_time"], args_dict=args_dict,
                  args_results=args_results, nn=False)


if __name__ == "__main__":
    main()
