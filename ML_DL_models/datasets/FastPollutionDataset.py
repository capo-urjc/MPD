import numpy as np
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader
from utils.generic_utils import unique


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


class FastTemporalPollutionDataset(Dataset):
    def __init__(self, path: str, correspondences: dict, sq_len_to_train: int, sq_len_to_predict: int,
                 magnitudes_to_train: list, magnitudes_to_predict: list, locations_to_train: list,
                 locations_to_predict: list, interpolate: str, transform=None, normalization: bool=True):

        if not (1 <= sq_len_to_train <= 24 * 365 - sq_len_to_predict):
            raise ValueError("sq_len_to_train value out of range")
        if not (1 <= sq_len_to_predict <= 24 * 365 - sq_len_to_train):
            raise ValueError("sq_len_to_predict value out of range")

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

        all_magnitudes = sorted(self.magnitudes_to_train + self.magnitudes_to_predict)
        all_locations = sorted(self.locations_to_train + self.locations_to_predict)

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

        csv_file_x: pd.DataFrame = csv_file[(csv_file["MAGNITUD"].isin(self.magnitudes_to_train)) &
                                            (csv_file["LOCALIZACION"].isin(self.locations_to_train))]

        csv_file_y: pd.DataFrame = csv_file[(csv_file["MAGNITUD"].isin(self.magnitudes_to_predict)) &
                                            (csv_file["LOCALIZACION"].isin(self.locations_to_predict))]

        csv_file_x_values = csv_file_x.sort_values(by=["MAGNITUD", "FECHA", "HORA"])["VALOR"].values
        csv_file_y_values = csv_file_y.sort_values(by=["MAGNITUD", "FECHA", "HORA"])["VALOR"].values

        values_x: np.ndarray = np.reshape(csv_file_x_values, (len(self.locations_to_train), 365*24,
                                                              len(self.magnitudes_to_train)), order='F')

        values_y: np.ndarray = np.reshape(csv_file_y_values, (len(self.locations_to_predict), 365*24,
                                                              len(self.magnitudes_to_predict)), order='F')

        # Create new matrix for hours (/24 to normalize)
        hours_row_x: np.ndarray = csv_file_x[csv_file["MAGNITUD"] == self.magnitudes_to_train[0]]["NHORA"].values / 24
        hours_row_y: np.ndarray = csv_file_y[csv_file["MAGNITUD"] == self.magnitudes_to_predict[0]]["NHORA"].values / 24

        hours_x = hours_row_x.reshape(len(self.locations_to_train), 365*24)
        hours_y = hours_row_y.reshape(len(self.locations_to_predict), 365*24)

        values_x = np.concatenate((values_x, hours_x[:, :, np.newaxis]), axis=2)
        values_y = np.concatenate((values_y, hours_y[:, :, np.newaxis]), axis=2)

        # Create new matrix for week day (/7 to normalize)
        weekday_row_x: np.ndarray = csv_file_x[csv_file["MAGNITUD"] == self.magnitudes_to_train[0]]["WEEKDAY"].values / 7
        weekday_row_y: np.ndarray = csv_file_y[csv_file["MAGNITUD"] == self.magnitudes_to_predict[0]]["WEEKDAY"].values / 7

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

        x = self.values_x[:, idx: idx + self.sq_len_to_train, :]
        y = self.values_y[:, idx + self.sq_len_to_train: idx + self.sq_len_to_train + self.sq_len_to_predict, :]

        sample: dict = {'x': x, 'y': y}
        # sample: dict = {'x': sample_values_x, 'y': sample_values_y}

        if self.transform:
            sample: dict = {'x': self.transform(sample['x']), 'y': self.transform(sample['y'])}

        return sample


if __name__ == "__main__":
    path_csv: str = "../preprocessing/df_for_training_24_1/2021_24.csv"
    path_correspondences: str = "../correspondences/correspondences.csv"
    dict_correspondences: dict = pd.read_csv(path_correspondences).to_dict(orient='list')
    magnitudes_to_train: list = [83, 86, 87, 1]
    magnitudes_to_predict: list = [1]
    locations_to_train: list = [(5, 2), (171, 1)]
    locations_to_predict: list = [(171, 1)]  # [(5, 2), (171, 1)]

    tic = time.time()

    tpd = FastTemporalPollutionDataset(path=path_csv,
                                       correspondences=dict_correspondences,
                                       sq_len_to_train=12,
                                       sq_len_to_predict=12,
                                       magnitudes_to_train=magnitudes_to_train,
                                       magnitudes_to_predict=magnitudes_to_predict,
                                       locations_to_train=locations_to_train,
                                       locations_to_predict=locations_to_predict,
                                       interpolate="linear",
                                       transform=None,
                                       normalization=True
                                       )

    toc = time.time()

    t = toc - tic

    print(f"El dataset tardó {t:.6f} segundos en cargar.")

    dataloader = DataLoader(tpd, batch_size=1, shuffle=True)

    elem = tpd[0]
    # elem

    start_time = time.time()

    for i, sample in enumerate(dataloader):
        inputs_, targets_ = sample['x'], sample['y']

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"El bucle for tardó {elapsed_time:.6f} segundos.")

