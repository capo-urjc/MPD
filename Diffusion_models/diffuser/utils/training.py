import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from tensorboardX import SummaryWriter


def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer=None,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        testset=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))

        if testset is not None:
            self.testset = testset
            self.test_dataloader = cycle(torch.utils.data.DataLoader(
                self.testset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
            ))
        else:
            self.testset = None
            self.test_dataloader = None

        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr, weight_decay=1e-5)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.summary_writer = SummaryWriter(os.path.join(results_folder, 'tensorboard'))

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    def train(self, n_train_steps):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=8000, eta_min=1e-6)

        timer = Timer()
        for step in range(n_train_steps):
            total_loss = 0.
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                total_loss += loss

            if step % 100 == 0:
                for key in infos.keys():
                    self.summary_writer.add_scalar('train_losses/'+ key, infos[key], global_step=self.step)

                self.summary_writer.add_scalar('loss', total_loss, global_step=self.step)

            self.optimizer.step()
            self.optimizer.zero_grad()

            scheduler.step()

            self.summary_writer.add_scalar('lr', self.get_lr(), global_step=self.step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                # infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                # print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
                print(f'{self.step}: {loss:8.4f} | t: {timer():8.4f}', flush=True)

            # TODO: # rendering the reference movement deactivated
            # print('self.sample_freq', self.sample_freq)
            if self.step == 0 and self.sample_freq and self.renderer is not None:
                self.render_reference(self.n_reference)
            # TODO: # rendering the reference movement deactivated
            if self.sample_freq and self.step % self.sample_freq == 0 and self.renderer is not None:
                self.render_samples()
                self.render_samples_w_displacement()


            self.step += 1

            if step % 100 == 0 and self.test_dataloader is not None:
                total_loss = 0.
                for i in range(self.gradient_accumulate_every):
                    batch = next(self.test_dataloader)
                    batch = batch_to_device(batch)

                    test_loss, infos = self.model.loss(*batch)
                    test_loss = test_loss / self.gradient_accumulate_every

                    total_loss += test_loss

                for key in infos.keys():
                    self.summary_writer.add_scalar('test_losses/'+ key, infos[key], global_step=self.step)
                self.summary_writer.add_scalar('test_loss', total_loss, global_step=self.step)

                # infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                # print(f'{self.step}: {total_loss:8.8f} | {infos_str} | t: {timer():8.4f}', flush=True)
                print(f'{self.step}: {total_loss:8.8f} | t: {timer():8.4f}', flush=True)

        label = self.step // self.label_freq * self.label_freq
        self.save(label)

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')

        if self.renderer is not None:
            self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=1, n_samples=1):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            normed_traj = to_np(batch.trajectories)[:, :, self.dataset.action_dim:]
            normed_traj = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_traj
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            normed_traj = self.dataset.normalizer.unnormalize(normed_traj, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            if self.renderer is not None:
                self.renderer.composite(savepath, observations, real_paths=normed_traj)

    def render_samples_w_displacement(self, batch_size=1, n_samples=1):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            normed_traj = to_np(batch.trajectories)[:, :, self.dataset.action_dim:]
            normed_traj = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_traj
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            normed_traj = self.dataset.normalizer.unnormalize(normed_traj, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}_w_displacement.png')
            if self.renderer is not None:
                self.renderer.composite_w_displacement(savepath, observations, real_paths=normed_traj)
