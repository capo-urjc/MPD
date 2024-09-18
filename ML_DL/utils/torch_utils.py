from datasets.FastPollutionDataset import FastTemporalPollutionDataset
# from models.NNs import UNet, SimpleCNN
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.normalize import NormalizeToOne
from utils.cast_to_precision import CastToPrecision


def get_opt(model: object, opt: str, lr: float, **kwargs: object) -> torch.optim:
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5, **kwargs)
    elif opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif opt == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, **kwargs)
    elif opt == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=lr, **kwargs)
    elif opt == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, **kwargs)
    elif opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif opt == "adamax":
        return torch.optim.Adamax(model.parameters(), lr=lr, **kwargs)
    elif opt == "asgd":
        return torch.optim.ASGD(model.parameters(), lr=lr, **kwargs)
    elif opt == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), lr=lr, **kwargs)
    elif opt == "rprop":
        return torch.optim.Rprop(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Optimizer {opt} not supported.")


def get_loss(loss: str, **kwargs: object) -> nn.Module:
    if loss == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss == "mse":
        return nn.MSELoss(**kwargs)
    elif loss == "l1":
        return nn.L1Loss(**kwargs)
    elif loss == "poisson":
        return nn.PoissonNLLLoss(**kwargs)
    elif loss == "bce":
        return nn.BCELoss(**kwargs)
    elif loss == "bce_with_logits":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss == "margin_ranking":
        return nn.MarginRankingLoss(**kwargs)
    elif loss == "hinge_embedding":
        return nn.HingeEmbeddingLoss(**kwargs)
    elif loss == "multi_margin":
        return nn.MultiMarginLoss(**kwargs)
    elif loss == "smooth_l1":
        return nn.SmoothL1Loss(**kwargs)
    else:
        raise ValueError(f"Loss function {loss} not supported.")


def get_transforms(nn: bool, precision: int) -> tuple:

    if nn:
        train_composed = transforms.Compose([CastToPrecision(precision=precision),
                                             transforms.ToTensor(),
                                             ])

        valid_composed = transforms.Compose([CastToPrecision(precision=precision),
                                             transforms.ToTensor(),
                                             ])

        test_composed = transforms.Compose([CastToPrecision(precision=precision),
                                            transforms.ToTensor(),
                                            ])
    else:
        train_composed = transforms.Compose([NormalizeToOne(),
                                             CastToPrecision(precision=precision),
                                             ])

        valid_composed = transforms.Compose([NormalizeToOne(),
                                             CastToPrecision(precision=precision),
                                             ])

        test_composed = transforms.Compose([NormalizeToOne(),
                                            CastToPrecision(precision=precision),
                                            ])

    return train_composed, valid_composed, test_composed


def get_datasets(dataset: str, patch_size: tuple, transforms: tuple, test_stride: tuple, coeffs: np.ndarray,
                 rgb_2_hsi: bool):
    assert dataset in ["navarra", "cave", "harvard", "harvard_i"], "The dataset is not valid"

    train_composed, valid_composed, test_composed = transforms

    root_dir: str = "../preprocessing/df_for_training_24"

    train_ds = FastTemporalPollutionDataset()

    valid_ds = FastTemporalPollutionDataset()

    test_ds = FastTemporalPollutionDataset()

    return train_ds, valid_ds, test_ds


def get_dataloaders(datasets: tuple, batch_size: int) -> tuple:
    train_dl = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_dl = DataLoader(datasets[2], batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    return train_dl, valid_dl, test_dl


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(128, 1)
    )

    opt = get_opt(model, "adam", 0.01, weight_decay=0.01)
    print(1)

