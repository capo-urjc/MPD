import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
from datasets.FastPollutionDataset import FastTemporalPollutionDataset
from train_functions import train
import torch
from torch.utils.data import DataLoader
from utils.generic_utils import load_correspondences
from utils.os_utils import get_identifier, logs_folder_structure
from utils.scikit_utils import get_nn_model
from utils.torch_utils import get_loss, get_opt


dtype = torch.cuda.FloatTensor


def main():

    parser = argparse.ArgumentParser(description='Description of my script')
    parser.add_argument('--dataset', default='MyData', help='Data')
    parser.add_argument('--num_epochs', default=350, help='Number of epochs')
    parser.add_argument('--lr', default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', default=32, help='Batch size')
    parser.add_argument('--loss', default='l1', help='Loss function')
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--magnitudes_to_train', default=[14, 81, 82, 83, 86, 87, 88], help='Magnitudes to train')
    parser.add_argument('--magnitudes_to_predict', default=[14], help='Magnitudes to predict')
    parser.add_argument('--locations_to_train', default=[(120, 1)], help='Locations to train')
    parser.add_argument('--locations_to_predict', default=[(120, 1)], help='Locations to predict')
    parser.add_argument('--sq_len_to_train', default=12, help='Sequence length to train')
    parser.add_argument('--sq_len_to_predict', default=12, help='Sequence length to predict')
    parser.add_argument('--model_type', default='model1', help='Model type')
    parser.add_argument('--interpolate', default='linear', help='Way to interpolate')
    parser.add_argument('--device', default='cuda', help='Device')

    args = parser.parse_args()
    print(args)

    args_dict: dict = vars(args)

    path_csv: str = "../DATA/2022_24.csv"
    path_csv_valid: str = "../DATA/2022_24.csv"
    path_correspondences: str = "../correspondences/correspondences.csv"

    # Get identifier of the experiment
    identifier: str = get_identifier(args=args_dict)

    log_dir = logs_folder_structure(identifier=identifier)

    dict_correspondences: dict = load_correspondences(path_correspondences)

    n_locs: int = len(args.locations_to_predict)

    # composed, _, _ = get_transforms(nn=False, precision=16)

    train_tpd = FastTemporalPollutionDataset(path=path_csv,
                                             correspondences=dict_correspondences,
                                             sq_len_to_train=args.sq_len_to_train,
                                             sq_len_to_predict=args.sq_len_to_predict,
                                             magnitudes_to_train=args.magnitudes_to_train,
                                             magnitudes_to_predict=args.magnitudes_to_predict,
                                             locations_to_train=args.locations_to_train,
                                             locations_to_predict=args.locations_to_predict,
                                             interpolate=args.interpolate,
                                             transform=None,
                                             normalization=True
                                             )

    valid_tpd = FastTemporalPollutionDataset(path=path_csv_valid,
                                             correspondences=dict_correspondences,
                                             sq_len_to_train=args.sq_len_to_train,
                                             sq_len_to_predict=args.sq_len_to_predict,
                                             magnitudes_to_train=args.magnitudes_to_train,
                                             magnitudes_to_predict=args.magnitudes_to_predict,
                                             locations_to_train=args.locations_to_train,
                                             locations_to_predict=args.locations_to_predict,
                                             interpolate=args.interpolate,
                                             transform=None,
                                             normalization=True
                                             )

    train_dataloader = DataLoader(train_tpd, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_tpd, batch_size=args.batch_size, shuffle=False)

    print("Training...")

    input_dim = (len(args.magnitudes_to_train) + 2) * args.sq_len_to_train * len(args.locations_to_train)
    output_dim = (len(args.magnitudes_to_predict) + 2) * args.sq_len_to_predict * len(args.locations_to_predict)

    model = get_nn_model(model_type=args.model_type, input_dim=input_dim, hidden_dim=512, output_dim=output_dim,
                         device=args.device)

    criterion = get_loss(args.loss)
    optimizer = get_opt(model=model, opt=args.optimizer, lr=args.lr)  # torch.optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.num_epochs

    train_loss, valid_loss, init_time, end_time = train(epochs=num_epochs,
                                                        model=model,
                                                        # batch_size=args.batch_size,
                                                        train_dl=train_dataloader,
                                                        valid_dl=valid_dataloader,
                                                        loss_fn=criterion,
                                                        device=args.device,
                                                        optimizer=optimizer,
                                                        log_dir=log_dir
                                                        )

    print("Trained")

    print(train_loss, valid_loss, init_time, end_time)


if __name__ == "__main__":
    main()
