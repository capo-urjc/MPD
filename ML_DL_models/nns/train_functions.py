from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from utils.os_utils import remove_files_in_directory


def train_one_epoch(model, train_dl, optimizer, loss_fn, epoch_number, device, writer) -> tuple:

    model.train(True)

    running_loss = 0.
    running_mae = 0.
    running_mse = 0.

    MAE: torchmetrics = torchmetrics.MeanAbsoluteError().to(device)
    MSE: torchmetrics = torchmetrics.MeanSquaredError().to(device)

    n_steps: int = int(len(train_dl))

    for i, sample in enumerate(train_dl):
        x_batch = sample['x'].float().to(device)
        y_batch = sample['y'].float().to(device)

        optimizer.zero_grad()

        outputs = model(x_batch)

        loss = loss_fn(outputs, y_batch.view(outputs.size()))

        mae = MAE(outputs, y_batch.view(outputs.size()))
        mse = MSE(outputs, y_batch.view(outputs.size()))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mae += mae
        running_mse += mse

        writer.add_scalar("loss_train", loss.item(), global_step=i+n_steps*epoch_number)
        writer.add_scalar("mae_train", mae, global_step=i+n_steps*epoch_number)
        writer.add_scalar("mse_train", mse, global_step=i+n_steps*epoch_number)

    print(f'Epoch [{epoch_number}], Loss: {loss.item():.6f}')

    return running_loss / (i + 1), running_mae / (i + 1), running_mse / (i + 1)


def valid_one_epoch(model, valid_dl, loss_fn, epoch_number, device, writer) -> tuple:

    model.eval()

    running_loss = 0.
    running_mae = 0.
    running_mse = 0.

    MAE: torchmetrics = torchmetrics.MeanAbsoluteError().to(device)
    MSE: torchmetrics = torchmetrics.MeanSquaredError().to(device)

    n_steps: int = int(len(valid_dl))

    with (torch.no_grad()):

        for i, sample in enumerate(valid_dl):
            x_batch = sample['x'].float().to(device)
            y_batch = sample['y'].float().to(device)

            outputs = model(x_batch)

            loss = loss_fn(outputs, y_batch.view(outputs.size()))

            mae = MAE(outputs, y_batch.view(outputs.size()))
            mse = MSE(outputs, y_batch.view(outputs.size()))

            running_loss += loss.item()
            running_mae += mae
            running_mse += mse

        writer.add_scalar("loss_valid", running_loss/(i+1), global_step=i+n_steps*epoch_number)
        writer.add_scalar("mae_valid", running_mae/(i+1), global_step=i+n_steps*epoch_number)
        writer.add_scalar("mse_valid", running_mse/(i+1), global_step=i+n_steps*epoch_number)

    print(f'Epoch [{epoch_number}], Valid Loss: {loss.item():.6f}')

    return running_loss / (i + 1), running_mae / (i + 1), running_mse / (i + 1)


# def valid_one_epoch(model, valid_dl: DataLoader, loss_fn: torch.nn, device: str) -> tuple:
#     print("Validating...")
#     running_vloss: float = 0.0
#
#     model.eval()
#
#     MAE: torchmetrics = torchmetrics.MeanAbsoluteError().to(device)
#     MSE: torchmetrics = torchmetrics.MeanSquaredError().to(device)
#
#     with (torch.no_grad()):
#         for i, valid_data in enumerate(valid_dl):
#             valid_inputs = valid_data['x'].float().to(device)
#             valid_labels = valid_data['y'].float().to(device)
#
#             valid_outputs = model(valid_inputs)
#
#             valid_loss = loss_fn(valid_outputs, valid_labels.view(valid_outputs.size()))
#             running_vloss += valid_loss
#
#             mae = MAE(valid_outputs, valid_labels.view(valid_outputs.size()))
#             mse = MSE(valid_outputs, valid_labels.view(valid_outputs.size()))
#             running_mae += mae
#             running_mse += mse
#     avg_vloss = running_vloss.item() / (i + 1)
#
#     mae = mae / (i + 1)
#     mse = mse / (i + 1)
#
#     print("Valid loss: ", avg_vloss)
#
#     return avg_vloss, mae, mse


def train(epochs: int, model, train_dl, valid_dl, loss_fn, device, optimizer, log_dir):

    writer: torch.utils.tensorboard.SummaryWriter = SummaryWriter(log_dir)

    init_time: str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    for e in range(epochs):
        train_loss, train_mae, train_mse = train_one_epoch(model=model,
                                                           train_dl=train_dl,
                                                           optimizer=optimizer,
                                                           loss_fn=loss_fn,
                                                           epoch_number=e,
                                                           device=device,
                                                           writer=writer)

        valid_loss, valid_mae, valid_mse = valid_one_epoch(model=model,
                                                           valid_dl=valid_dl,
                                                           loss_fn=loss_fn,
                                                           device=device,
                                                           epoch_number=e,
                                                           writer=writer)

        writer.add_scalar("epochs", e, global_step=e)
        writer.add_scalar("loss_train_per_epoch", train_loss, global_step=e)
        writer.add_scalar("loss_valid", valid_loss, global_step=e)
        # writer.add_scalar("mae_train", train_mae, global_step=e)
        # writer.add_scalar("mse_train", train_mse, global_step=e)
        writer.add_scalar("mae_valid", valid_mae, global_step=e)
        writer.add_scalar("mse_valid", valid_mse, global_step=e)
        #

        checkpoint = {
            "epoch": e,
            "model_st_dict": model.state_dict(),
        }

        remove_files_in_directory(log_dir + "/checkpoints/")

        torch.save(checkpoint, log_dir + "/checkpoints/" + "epoch-" + str(e) + ".pth")

    end_time: str = datetime.now().strftime('%Y%m%d_%H%M%S')

    return train_loss, valid_loss, init_time, end_time


# def train_by_iterations(iterations: int, model, batch_size: int, train_dl, valid_dl, loss_fn, device, optimizer, log_dir):
#     writer: torch.utils.tensorboard.SummaryWriter = SummaryWriter(log_dir)
#
#     init_time: str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
#     for it in range(iterations):
