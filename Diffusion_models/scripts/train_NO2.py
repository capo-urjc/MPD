import diffuser.utils as utils
from diffuser.models.helpers import (apply_conditioning_NO2)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

fn_condition = apply_conditioning_NO2 # apply_simplified_direction_coord_init_conditioning # apply_direction_coord_init_conditioning  # apply_direction_coord_init_conditioning  # apply_conditioning

class Parser(utils.ParserNO2): # TODO: change name of utils.ParserNO2 to Parser and remove original parser
    config: str = 'config.NO2'

    # PAPER DATA
    # test_path: str = "../Mad_Station/Mad_Station_2022.csv"
    # train_path: str = "../Mad_Station/Mad_Station_2019.csv"
    # path_correspondences: str = "../correspondences/correspondencesPaper.csv"

    # # OUR DATA
    test_path: str = "../DATA/2023_24.csv"
    train_path: str = "../DATA/2022_24.csv"
    path_correspondences: str = "../correspondences/correspondences.csv"

# --loader datasets.GoalDataset / datasets.SequenceDataset

args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.ConfigTrainTest(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    train_path=args.train_path,
    test_path=args.test_path,
    path_correspondences=args.path_correspondences,
    horizon=args.horizon,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

# render_config = utils.ConfigTrainTest(
#     args.renderer,
#     savepath=(args.savepath, 'render_config.pkl'),
#     env=args.dataset,
# )

dataset, testset = dataset_config()
# renderer, _ = render_config()

observation_dim = dataset.observation_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    # action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
    fn_condition=fn_condition,
    n_locations=dataset.locations
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, testset=testset)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)


