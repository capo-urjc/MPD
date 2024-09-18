# import socket
#
from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs/1'

base = {
    'diffusion': {
        # 'test_path': "../Mad_Station/Mad_Station_2022.csv",
        # 'train_path': "../Mad_Station/Mad_Station_2019.csv",

        'test_path': "../DATA/2023_24.csv",
        'train_path': "../DATA/2022_24.csv",

        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusionModel',
        'horizon': 24,
        'n_diffusion_steps': 20,
        'loss_weights':  {0:  1.,
                          1:  0.0002,
                          2:  0.0002,
                          3:  0.0002,
                          4:  0.0002,
                          5:  0.0002,
                          6:  0.0002,
                          # 7:  0.0005,
                          # 8:  0.0005,
                          # 9:  0.0005,
                          # 10: 0.0005
                          },
        'loss_discount': 0.9,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': None, #'utils.EMARenderer',

        ## dataset
        'loader': 'datasets.SequenceDatasetNO2NormalizedOUR',
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 24,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 2000,
        'loss_type': 'l1',
        'n_train_steps': 40000,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 1,
        'ema_decay':  0,  #0.995,
        'save_freq': 500,
        'sample_freq': 250,
        'n_saves': 20,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'test': {
        # 'test_path': "../Mad_Station/Mad_Station_2022.csv",
        # 'train_path': "../Mad_Station/Mad_Station_2019.csv",

        'test_path': "../DATA/2023_24.csv",
        'train_path': "../DATA/2022_24.csv",

        # 'guide': 'sampling.ValueGuide',
        'policy': 'sampling.DiffPolicy',
        'max_episode_length': 1000,
        'batch_size': 1024,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 24,
        'n_diffusion_steps': 20,
        'renderer': None,

        ## value function
        'discount': 0.997,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}/2024-09-17:09-29',

        'diffusion_epoch': 'latest',

        'verbose': True,
        'suffix': '0',

        'loader': 'datasets.SequenceDatasetNO2NormalizedOUR',
    },
}


