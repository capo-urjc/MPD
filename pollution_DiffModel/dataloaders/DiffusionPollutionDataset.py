from functools import reduce
from functools import reduce
import numpy as np
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader

from dataloaders.PaperDataset import PaperDataset



def unique(list1: list):
    """

    :param list1: list with possible duplicates

    :return:the function returns the list passed as an argument without duplicates
    """
    ans = reduce(lambda re, x: re + [x] if x not in re else re, list1, [])
    return ans


def convert_to_float(value) -> float:
    """

    :param value: The value to convert. It should be a string representing
                        a number or any numeric data type.
    :return: The value converted to a floating-point number.

    May 2024
    """

    if isinstance(value, str):
        value = value.replace(',', '.')  # Replace comma with dot to have a valid format for float
    return float(value)


class DiffusionPollutionDataset(PaperDataset):
    def __init__(self, path: str, correspondences: dict, sq_len_to_train: int, sq_len_to_predict: int,
                 magnitudes_to_train: list, magnitudes_to_predict: list, locations_to_train: list,
                 locations_to_predict: list, interpolate: str, transform=None, normalization: bool=True, da_noise=0.0):

        if not (1 <= sq_len_to_train <= 24 * 365 - sq_len_to_predict):
            raise ValueError("sq_len_to_train value out of range")
        if not (1 <= sq_len_to_predict <= 24 * 365 - sq_len_to_train):
            raise ValueError("sq_len_to_predict value out of range")

        self.da_noise = da_noise

        self.path: str = path

        self.sq_len_to_train = sq_len_to_train
        self.sq_len_to_predict = sq_len_to_predict

        self.magnitudes_to_train = sorted(magnitudes_to_train)
        self.locations_to_train = sorted(locations_to_train)

        self.magnitudes_to_predict = sorted(magnitudes_to_predict)
        self.locations_to_predict = sorted(locations_to_predict)

        self.correspondences = correspondences

        self.transform = transform

        self.normalization = normalization

        self.interpolate = interpolate  # necesario?

        csv_file = pd.read_csv(self.path)

        all_magnitudes = unique(sorted(self.magnitudes_to_train + self.magnitudes_to_predict))
        all_locations = unique(sorted(self.locations_to_train + self.locations_to_predict))

        csv_file["VALOR"] = csv_file["VALOR"].apply(convert_to_float)

        # Interpolate NaN values
        csv_file["VALOR"] = csv_file["VALOR"].interpolate(method=self.interpolate)

        csv_file["LOCALIZACION"] = list(zip(csv_file["MUNICIPIO"], csv_file["ESTACION"]))

        csv_file = csv_file.loc[(csv_file["MAGNITUD"].isin(all_magnitudes))
                                & (csv_file["LOCALIZACION"].isin(all_locations))]

        csv_file.reset_index(drop=True, inplace=True)

        self.magnitudes: list = unique(csv_file["MAGNITUD"])
        self.locations: list = unique(csv_file["LOCALIZACION"])

        if self.normalization:
            for m in self.magnitudes:
                idx_cod = self.correspondences["CODIFICACION"].index(m)

                min_pos_value: int = self.correspondences["MINIMO"][idx_cod]
                max_pos_value: int = self.correspondences["MAXIMO"][idx_cod]

                # self.csv_file.loc[self.csv_file["MAGNITUD"] == m, "VALOR"] = (self.csv_file[self.csv_file["MAGNITUD"] == m]["VALOR"] - min_pos_value) / (max_pos_value - min_pos_value)
                csv_file.loc[csv_file["MAGNITUD"] == m, "VALOR"] = (csv_file[csv_file["MAGNITUD"] == m]["VALOR"] - min_pos_value) / (max_pos_value - min_pos_value)
                # self.csv_file = csv_file  # descomentar esto

        # Here we have that csv_file is a subset of the data filtered by the magnitudes, stations and municipalities
        # specified in the constructor

        csv_file_x: pd.DataFrame = csv_file[(csv_file["MAGNITUD"].isin(all_magnitudes)) &
                                            (csv_file["LOCALIZACION"].isin(self.locations_to_train))]

        csv_file_y: pd.DataFrame = csv_file[(csv_file["MAGNITUD"].isin(all_magnitudes)) &
                                            (csv_file["LOCALIZACION"].isin(self.locations_to_predict))]

        csv_file_x_values = csv_file_x.sort_values(by=["MAGNITUD", "FECHA", "HORA"])["VALOR"].values
        csv_file_y_values = csv_file_y.sort_values(by=["MAGNITUD", "FECHA", "HORA"])["VALOR"].values

        values_x: np.ndarray = np.reshape(csv_file_x_values, (len(self.locations_to_train), 365*24,
                                                              len(all_magnitudes)), order='F')

        values_y: np.ndarray = np.reshape(csv_file_y_values, (len(self.locations_to_predict), 365*24,
                                                              len(all_magnitudes)), order='F')

        # Create new matrix for hours (/24 to normalize)
        hours_row_x: np.ndarray = csv_file_x[csv_file["MAGNITUD"] == all_magnitudes[0]]["NHORA"].values / 24
        hours_row_y: np.ndarray = csv_file_y[csv_file["MAGNITUD"] == all_magnitudes[0]]["NHORA"].values / 24

        hours_x = hours_row_x.reshape(len(self.locations_to_train), 365*24)
        hours_y = hours_row_y.reshape(len(self.locations_to_predict), 365*24)

        values_x = np.concatenate((values_x, hours_x[:, :, np.newaxis]), axis=2)
        values_y = np.concatenate((values_y, hours_y[:, :, np.newaxis]), axis=2)

        # Create new matrix for week day (/7 to normalize)
        weekday_row_x: np.ndarray = csv_file_x[csv_file["MAGNITUD"] == all_magnitudes[0]]["WEEKDAY"].values / 7
        weekday_row_y: np.ndarray = csv_file_y[csv_file["MAGNITUD"] == all_magnitudes[0]]["WEEKDAY"].values / 7

        weekday_x = weekday_row_x.reshape(len(self.locations_to_train), 365 * 24)
        weekday_y = weekday_row_y.reshape(len(self.locations_to_predict), 365 * 24)

        values_x = np.concatenate((values_x, weekday_x[:, :, np.newaxis]), axis=2)
        values_y = np.concatenate((values_y, weekday_y[:, :, np.newaxis]), axis=2)

        self.values_x = values_x
        self.values_y = values_y


    def __len__(self):
        # return len(self.csv_file) - (self.sq_len_to_train + self.sq_len_to_predict)
        return 24 * 365 - (self.sq_len_to_train + self.sq_len_to_predict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start_time_ = idx
        final_time_cond = idx + self.sq_len_to_train
        final_time = idx + self.sq_len_to_train + self.sq_len_to_predict

        x = np.float32(self.values_x[:, start_time_: final_time, :])
        cond = np.float32(self.values_y[:, start_time_: final_time_cond, :])

        x = x.transpose(1, 0, 2)
        cond = cond.transpose(1, 0, 2)

        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        cond = cond.reshape((cond.shape[0], cond.shape[1] * cond.shape[2]))

        sample: dict = {'x': x, 'cond': cond + np.random.rand(cond.shape[0], cond.shape[1]) * self.da_noise}

        if self.transform:
            sample: dict = {'x': self.transform(sample['x']), 'cond': self.transform(sample['cond'])}

        return sample


if __name__ == "__main__":
    path_csv: str = "../DATA/2022_24.csv"

    df: pd.DataFrame = pd.read_csv(path_csv)

    path_correspondences: str = "../correspondences/correspondences.csv"

    dict_correspondences: dict = pd.read_csv(path_correspondences).to_dict(orient='list')

    tpd = DiffusionPollutionDataset(path=path_csv,
                                    correspondences=dict_correspondences,
                                    sq_len_to_train=12,
                                    sq_len_to_predict=12,
                                    magnitudes_to_train=[20, 81, 82, 83, 86, 87, 88],
                                    magnitudes_to_predict=[20],
                                    # locations_to_train=[(123, 2), (133, 2), (14, 2), (148, 4), (16, 1), (171, 1), (180, 1), (45, 2), (47, 2), (49, 3), (5, 2), (58, 4), (6, 4), (65, 14), (67, 1), (7, 4), (74, 7), (80, 3), (9, 1), (92, 5)],
                                    locations_to_train=[(16, 1)],
                                    locations_to_predict=[(16, 1)],
                                    interpolate="linear",
                                    transform=None,
                                    )

    dataloader = DataLoader(tpd, batch_size=1, shuffle=True)

    for i, sample in enumerate(dataloader):
        inputs_, targets_ = sample['x'], sample['cond']

