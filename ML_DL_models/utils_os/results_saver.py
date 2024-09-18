import os
import pandas as pd


def get_name_2_save(args_dict: dict) -> str:
    name_2_save: str = (str(args_dict["method"]) + '-' + str(args_dict["n_its"]) + '_' + str(args_dict["reg_noise_std"]) + '_' +
                        str(args_dict["lr"]) + '_' + str(args_dict["input_depth"]) + ".pt")

    return name_2_save


def get_version(configs: pd.DataFrame, args_dict: dict) -> int:
    merged_df = pd.merge(configs, pd.DataFrame([args_dict]), how='inner')
    if len(merged_df) > 0:
        version = len(merged_df)
    else:
        version = 0

    return version


def columns_getter(args_dict: dict, extra_columns: list):
    columns: list = []

    for key, value in args_dict.items():
        columns.append(key)

    for c in extra_columns:
        columns.append(c)

    return columns


def results_saver(folder: str, name_csv: str, extra_columns: list, args_dict: dict, args_results: dict, nn: bool) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)

    args_keys: list = list(args_dict.keys())

    if not os.path.exists(folder + name_csv):
        columns: list = columns_getter(args_dict, extra_columns)
        commands_file: pd.DataFrame = pd.DataFrame(columns=columns)

    else:
        commands_file: pd.DataFrame = pd.read_csv(folder + name_csv)

    # Filter if an experiment already exists with the same configuration
    configs: pd.DataFrame = commands_file[list(commands_file[args_keys].columns)]

    if nn:
        # Get the version
        version: int = get_version(configs, args_dict)

        third_dict: dict = {"version": version}

        # Update the dataframe
        commands_file: pd.DataFrame = pd.concat(
            [commands_file, pd.DataFrame([{**args_dict, **third_dict, **args_results}])], ignore_index=False)

    else:
        commands_file: pd.DataFrame = pd.concat(
            [commands_file, pd.DataFrame([{**args_dict, **args_results}])], ignore_index=False)

    # Save to csv
    commands_file.to_csv(folder + name_csv, index=False)
