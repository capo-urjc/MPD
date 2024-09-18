# import pdb
# from numpy.distutils.from_template import list_re
# from torch.profiler import tensorboard_trace_handler

# import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
import einops
import numpy as np
import torch
# from diffuser.utils.training import cycle
# import os
# import matplotlib.pyplot as plt
# import torchmetrics

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.ParserNO2):
    config: str = 'config.NO2'
    # diffusion_loadpath: str = 'diffusion/defaults_H24_T20/2024-07-18:18-24'
args = Parser().parse_args('test')


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)


diffusion = diffusion_experiment.ema
diff_trainer = diffusion_experiment.trainer
_, testset = diffusion_experiment.dataset

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

batch_size = 1024
# test_dataloader = cycle(torch.utils.data.DataLoader(
#             testset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True
#         ))

test_dataloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)

observations = []
list_pred = []
list_gt = []

print(len(testset))

mae_list = []
mse_list = []
rmse_list = []

for sample in range(20):
    observations = []
    list_pred = []
    list_gt = []
    for i, batch in enumerate(test_dataloader):
        conditions = to_device(batch.conditions, 'cuda')

        samples = diffusion(conditions)

        trajectories = to_np(samples.trajectories)

        traj = trajectories.reshape((trajectories.shape[0], args.horizon, testset.locations, -1))

        gt_traj = batch.trajectories.numpy()
        gt_traj = gt_traj.reshape((trajectories.shape[0], args.horizon, testset.locations, -1))

        gt_no2 = gt_traj
        pred_no2 = traj

        list_pred.extend(pred_no2[..., 0])
        list_gt.extend(gt_no2[..., 0])

    list_pred = np.array(list_pred)[:, 12:, :] * 50
    list_gt = np.array(list_gt)[:, 12:, :] * 50

    mae = np.abs(list_pred - list_gt)
    mse = np.mean((list_pred - list_gt) ** 2)
    rmse = np.sqrt(np.mean(mse))

    mae_list.append(np.mean(mae))
    mse_list.append(np.mean(mse))
    rmse_list.append(rmse)

print("MAE: ", np.mean(mae_list), '+-', np.std(mae_list))
print("RMSE: ", np.mean(rmse_list), '+-', np.std(rmse_list))
print("MSE: ", np.mean(mse_list), '+-', np.std(mse_list))

    # print("MAE:", np.mean(mae))
    # print("MSE:", np.mean(mse))
    # print("RMSE:", np.sqrt(np.mean(mse)))

