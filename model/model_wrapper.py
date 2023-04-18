import os

import numpy as np
import torch

from criterions.optim import Optimizer, Scheduler
from utils.misc import map_location


class ModelWrapper:
    def __init__(self, args, mode, working_dir, model, is_train, device):
        self._model = model.to(device)
        self._mode = mode
        self._args = args
        self._is_train = is_train
        self._save_dir = working_dir

        if self._is_train:
            self._init_train_vars()
            self._model.train()
        else:
            self._model.eval()
        self._device = device

    @property
    def is_train(self):
        return self._is_train

    def _init_train_vars(self):
        # infer learning rate before changing batch size
        init_lr = self._args.lr * self._args.batch_size / 256
        self._optimizer = Optimizer().get(self._model, self._args.optimizer, lr=init_lr,
                                          wd=self._args.weight_decay)
        self.lr_scheduler = Scheduler().get(self._args.lr_policy, self._optimizer, step_size=self._args.lr_decay_epochs)

    def load(self, without_optimiser=False):
        # load feature extractor
        self._load_network(self._model)
        if not without_optimiser:
            self._load_optimizer(self._optimizer)

    def existing(self):
        return self._check_model()

    def get_current_lr(self):
        lr = []
        for param_group in self._optimizer.param_groups:
            lr.append(param_group['lr'])
        print('current learning rate: {}'.format(np.unique(lr)))

    def save(self):
        """
        save network, the filename is specified with the sofar tasks and iteration
        """
        self._save_network(self._model)
        # save optimizers
        self._save_optimizer(self._optimizer)

    def _save_optimizer(self, optimizer):
        save_filename = f'{self._mode}-optimizer.pth'
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def _load_optimizer(self, optimizer):
        load_filename = f'{self._mode}-optimizer.pth'
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(load_path), 'Weights file %s not found!' % load_path

        optimizer.load_state_dict(torch.load(load_path))
        print('loaded optimizer: %s' % load_path)

    def _save_network(self, network):
        save_filename = f'{self._mode}-net.pth'
        save_path = os.path.join(self._save_dir, save_filename)
        save_dict = network.state_dict()
        torch.save(save_dict, save_path)
        print('saved net: %s' % save_path)

    def load_network(self, pretrained_checkpoint):
        assert os.path.exists(pretrained_checkpoint), 'Weights file %s not found ' % pretrained_checkpoint
        checkpoint = torch.load(pretrained_checkpoint, map_location=map_location(self._args.cuda))
        self._model.load_state_dict(checkpoint)
        print('loaded net: %s' % pretrained_checkpoint)

    def _check_model(self):
        load_filename = f'{self._mode}-net.pth'
        load_path = os.path.join(self._save_dir, load_filename)
        return os.path.exists(load_path)

    def _load_network(self, network):
        load_filename = f'{self._mode}-net.pth'
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(load_path), 'Weights file %s not found ' % load_path
        checkpoint = torch.load(load_path, map_location=map_location(self._args.cuda))
        network.load_state_dict(checkpoint)
        print('loaded net: %s' % load_path)

    def set_train(self):
        self._model.train()
        self._is_train = True

    def set_eval(self):
        self._model.eval()
        self._is_train = False

    def forward(self, images):
        images = [x.to(self._device, non_blocking=True) for x in images]
        with torch.set_grad_enabled(self._is_train):
            return self._model(images)

    def compute_loss(self, batch_data):
        images, target = batch_data
        images = [x.to(self._device, non_blocking=True) for x in images]
        target = [{k: v.to(self._device, non_blocking=True) for k, v in t.items()} for t in target]

        all_losses = {}
        with torch.set_grad_enabled(self._is_train):
            losses = self._model(images, target)
            all_losses.update(losses)

        return all_losses

    def optimise_params(self, all_losses):
        self._optimizer.zero_grad()
        loss = sum(all_losses.values()) / len(all_losses)
        loss.backward()
        self._optimizer.step()
