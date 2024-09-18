import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloaders.PaperDataset import PaperDataset

class DiffusionDataloader(PaperDataset):
    def __init__(self, path: str, correspondences: dict, sq_len_to_train: int, sq_len_to_predict: int, interpolate: str,
                 transform=None, normalization: bool=True, categorical: bool=True):

        if not (1 <= sq_len_to_train <= 24 * 365 - sq_len_to_predict):
            raise ValueError("sq_len_to_train value out of range")
        if not (1 <= sq_len_to_predict <= 24 * 365 - sq_len_to_train):
            raise ValueError("sq_len_to_predict value out of range")

        self.path: str = path

        self.sq_len_to_train = sq_len_to_train
        self.sq_len_to_predict = sq_len_to_predict

        self.transform = transform

        self.correspondences = correspondences

        self.normalization = normalization
        self.categorical = categorical

        self.interpolate = interpolate

        csv_file = pd.read_csv(self.path)

        # if "windDir" in csv_file.columns:
        #     csv_file = csv_file.drop(["windDir"], axis=1)

        if self.normalization:
            for column in csv_file.columns:
                if column in correspondences["MAGNITUD"]:
                    idx: int = correspondences["MAGNITUD"].index(column)

                    max_value: float = correspondences["MAXIMO"][idx]
                    min_value: float = correspondences["MINIMO"][idx]

                    csv_file[column] = (csv_file[column] - min_value) / (max_value - min_value)

        if not self.categorical:
            csv_file = csv_file.drop(["windDir_Categ_east"], axis=1)
            csv_file = csv_file.drop(["windDir_Categ_north"], axis=1)
            csv_file = csv_file.drop(["windDir_Categ_northeast"], axis=1)
            csv_file = csv_file.drop(["windDir_Categ_northwest"], axis=1)
            csv_file = csv_file.drop(["windDir_Categ_south"], axis=1)
            csv_file = csv_file.drop(["windDir_Categ_southeast"], axis=1)
            csv_file = csv_file.drop(["windDir_Categ_southwest"], axis=1)
            csv_file = csv_file.drop(["windDir_Categ_west"], axis=1)

        else:
            if "windDir" in csv_file.columns:
                csv_file = csv_file.drop(["windDir"], axis=1)

        # Calculate the number of columns based on whether we have categorized or not
        n_cols = len(csv_file.columns)

        final_data: np.ndarray = csv_file.to_numpy().reshape(-1, 24, n_cols)
        self.final_data = final_data.transpose((1, 0, 2))
        # This returns a numpy tensor of dimensions 4343 x 24 x 18 (18 or n_cols)

        # Data in .csv is organized as follows:

        # index  - Magnitude1 - Magnitude2 - ... - Magnitude18
        # H1S1   - .......... - .......... - ... - ...........
        # H1S2   - .......... - .......... - ... - ...........
        # ....   - .......... - .......... - ... - ...........
        # H1S24  - .......... - .......... - ... - ...........
        # H2S1   - .......... - .......... - ... - ...........
        # H2S2   - .......... - .......... - ... - ...........
        # ....   - .......... - .......... - ... - ...........
        # H2S24  - .......... - .......... - ... - ...........
        # H3S1   - .......... - .......... - ... - ...........
        # ....   - .......... - .......... - ... - ...........
        # ....   - .......... - .......... - ... - ...........
        # ....   - .......... - .......... - ... - ...........
        # H24S24 - .......... - .......... - ... - ...........

    def __len__(self):
        return 4343 - (self.sq_len_to_train + self.sq_len_to_predict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start_time_ = idx
        final_time_cond = idx + self.sq_len_to_train
        final_time = idx + self.sq_len_to_train + self.sq_len_to_predict

        x: np.ndarray = np.float32(self.final_data[:, start_time_: final_time, :])
        x = x.transpose(1, 0, 2)
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))

        cond: np.ndarray = np.float32(self.final_data[:, start_time_: final_time_cond, :])
        cond = cond.transpose(1, 0, 2)
        cond = cond.reshape((cond.shape[0], cond.shape[1] * cond.shape[2]))

        return {'x': x, 'cond': cond}


if __name__ == "__main__":
    path_csv: str = "../Mad_Station/Mad_Station_2022.csv"

    df: pd.DataFrame = pd.read_csv(path_csv)

    path_correspondences: str = "../correspondences/correspondencesPaper.csv"

    dict_correspondences: dict = pd.read_csv(path_correspondences).to_dict(orient='list')

    tpd = DiffusionDataloader(path=path_csv,
                              correspondences=dict_correspondences,
                              sq_len_to_train=12,
                              sq_len_to_predict=12,
                              interpolate="linear",
                              transform=None,
                              normalization=False,
                              categorical=False
                              )

    dataloader = DataLoader(tpd, batch_size=1, shuffle=True)

    for i, sample in enumerate(dataloader):
        inputs_, targets_ = sample['x'], sample['cond']


