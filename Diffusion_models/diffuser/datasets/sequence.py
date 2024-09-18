from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
# from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from torch.utils.data import Dataset, DataLoader
from dataloaders.DiffusionDataloader import DiffusionDataloader
from dataloaders.DiffusionPollutionDataset import DiffusionPollutionDataset
import pandas as pd


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDataset(torch.utils.data.Dataset):

    # def __init__(self, env='hopper-medium-replay', horizon=64,
    #     normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
    #     max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
    #
    #
    #     self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
    #     self.env = env = load_environment(env)
    #     self.env.seed(seed)
    #
    #     self.horizon = horizon
    #     self.max_path_length = max_path_length
    #     self.use_padding = use_padding
    #     itr = sequence_dataset(env, self.preprocess_fn)
    #
    #     fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
    #     for i, episode in enumerate(itr):
    #         fields.add_path(episode)
    #     fields.finalize()
    #
    #     self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
    #     self.indices = self.make_indices(fields.path_lengths, horizon)
    #
    #     self.observation_dim = fields.observations.shape[-1]
    #     self.action_dim = fields.actions.shape[-1]
    #     self.fields = fields
    #     self.n_episodes = fields.n_episodes
    #     self.path_lengths = fields.path_lengths
    #     self.normalize()
    #
    #     print(fields)
    #     # shapes = {key: val.shape for key, val in self.fields.items()}
    #     # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

class SequenceDatasetNO2Normalized(SequenceDataset):
    def __init__(self, path_csv, path_correspondences, horizon=24, max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None, batch_size=32, **kwargs):

        # self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        # self.env = env = load_environment(env)
        # self.env.seed(seed)

        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        # fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        # for i, episode in enumerate(itr):
        #     fields.add_path(episode)
        # fields.finalize()

        # TODO: self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        # self.indices = self.make_indices(fields.path_lengths, horizon)

        # self.observation_dim = fields.observations.shape[-1]
        # self.action_dim = fields.actions.shape[-1]

        dict_correspondences: dict = pd.read_csv(path_correspondences).to_dict(orient='list')

        self.tpd = DiffusionDataloader(path=path_csv,
                                       correspondences=dict_correspondences,
                                       sq_len_to_train=horizon//2,
                                       sq_len_to_predict=horizon//2,
                                       interpolate="linear",
                                       transform=None,
                                       normalization=True,
                                       categorical=False
                                       )

        # self.dataloader = DataLoader(self.tpd, batch_size=batch_size, shuffle=True)

        self.observation_dim = self.tpd[0]['x'].shape[-1]  # TODO: renombrar observation_dim a input_shape
        # self.fields = fields
        # self.n_episodes = fields.n_episodes
        # self.path_lengths = fields.path_lengths

        # print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def __len__(self):
        return self.tpd.__len__()

    def __getitem__(self, idx):
        ittem =  self.tpd.__getitem__(idx)
        return Batch(ittem['x'], ittem['cond'])


class SequenceDatasetNO2NormalizedOUR(SequenceDataset):
    def __init__(self, path_csv, path_correspondences, horizon=24, max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None, batch_size=32, **kwargs):

        # self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        # self.env = env = load_environment(env)
        # self.env.seed(seed)

        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        # fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        # for i, episode in enumerate(itr):
        #     fields.add_path(episode)
        # fields.finalize()

        # TODO: self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        # self.indices = self.make_indices(fields.path_lengths, horizon)

        # self.observation_dim = fields.observations.shape[-1]
        # self.action_dim = fields.actions.shape[-1]

        dict_correspondences: dict = pd.read_csv(path_correspondences).to_dict(orient='list')

        locs: list = [(16, 1), (171, 1), (47, 2), (5, 2), (92, 5)]

        self.tpd = DiffusionPollutionDataset(path=path_csv,
                                             correspondences=dict_correspondences,
                                             sq_len_to_train=12,
                                             sq_len_to_predict=12,
                                             magnitudes_to_train=[1, 81, 82, 83, 86, 87, 88],
                                             magnitudes_to_predict=[1],
                                             locations_to_train=locs,
                                             locations_to_predict=locs,
                                             interpolate="linear",
                                             transform=None,
                                             )

        self.locations = len(locs)
        # self.dataloader = DataLoader(self.tpd, batch_size=batch_size, shuffle=True)

        self.observation_dim = self.tpd[0]['x'].shape[-1]  # TODO: renombrar observation_dim a input_shape
        # self.fields = fields
        # self.n_episodes = fields.n_episodes
        # self.path_lengths = fields.path_lengths

        # print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def __len__(self):
        return self.tpd.__len__()

    def __getitem__(self, idx):
        ittem =  self.tpd.__getitem__(idx)
        return Batch(ittem['x'], ittem['cond'])


