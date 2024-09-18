import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from datasets.PaperDataset import PaperDataset
from nns.train_functions import train
from torch.utils.data import DataLoader
from utils.generic_utils import load_correspondences
from utils.os_utils import get_identifier, logs_folder_structure
from utils.scikit_utils import get_nn_model
from utils.torch_utils import get_loss, get_opt, get_transforms


def main():

    parser = argparse.ArgumentParser(description='Description of my script')
    parser.add_argument('--dataset', default='PaperDatasigm', help='Data')
    parser.add_argument('--num_epochs', default=250, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, help='Batch size')
    parser.add_argument('--lr', default=0.000001, help='Learning rate')
    parser.add_argument('--loss', default='l1', help='Loss function')
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--sq_len_to_train', default=12, help='Sequence length to train')
    parser.add_argument('--sq_len_to_predict', default=12, help='Sequence length to predict')
    parser.add_argument('--model_type', default='model_nn', help='Model type')
    parser.add_argument('--interpolate', default='linear', help='Way to interpolate')
    parser.add_argument('--categorical', default=False, help='Categorical variable for windDir')
    parser.add_argument('--device', default='cuda', help='Device')

    args = parser.parse_args()
    print(args)

    args_dict: dict = vars(args)

    path_csv: str = "../Mad_Station/Mad_Station_2019.csv"
    path_csv_test: str = "../Mad_Station/Mad_Station_2022.csv"

    path_correspondences: str = "../correspondences/correspondencesPaper.csv"
    dict_correspondences: dict = load_correspondences(path_correspondences)

    identifier: str = get_identifier(args=args_dict)

    log_dir = logs_folder_structure(identifier=identifier)

    # composed, _, _ = get_transforms(nn=False, precision=16)

    train_tpd = PaperDataset(path=path_csv,
                             correspondences=dict_correspondences,
                             sq_len_to_train=args.sq_len_to_train,
                             sq_len_to_predict=args.sq_len_to_predict,
                             interpolate=args.interpolate,
                             transform=None,
                             normalization=True,
                             categorical=args.categorical
                             )

    valid_tpd = PaperDataset(path=path_csv_test,
                             correspondences=dict_correspondences,
                             sq_len_to_train=args.sq_len_to_train,
                             sq_len_to_predict=args.sq_len_to_predict,
                             interpolate=args.interpolate,
                             transform=None,
                             normalization=True,
                             categorical=args.categorical
                             )

    train_dataloader = DataLoader(train_tpd, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_tpd, batch_size=args.batch_size, shuffle=False)

    print("Training...")

    if args.categorical:
        input_dim = 18 * args.sq_len_to_train * 24
    else:
        input_dim = 11 * args.sq_len_to_train * 24

    output_dim = 1 * args.sq_len_to_predict * 24

    model = get_nn_model(model_type=args.model_type, input_dim=input_dim, hidden_dim=512, output_dim=output_dim,
                         device=args.device)

    criterion = get_loss(args.loss)
    optimizer = get_opt(model=model, opt=args.optimizer, lr=args.lr)

    num_epochs = args.num_epochs

    train_loss, valid_loss, init_time, end_time = train(epochs=num_epochs,
                                                        model=model,
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
