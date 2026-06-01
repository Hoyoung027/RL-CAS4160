import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions
import torch.nn as nn

from cas4160.infrastructure import pytorch_util as ptu


class RewardPredictor(nn.Module):
    """Base MLP reward predictor, which takes an observation and outputs a predicted reward.

    This class should implement the `forward` and `get_reward` methods. The `update` method should be written in the
    subclasses, since the reward prediction update rule differs for different algorithms.
    """
    def __init__(self, input_size: int, n_layers: int, layer_size: int, learning_rate: float):
        super().__init__()

        self.reward_net = ptu.build_mlp(
            input_size=input_size,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
            activation="tanh",
            output_activation="identity"
        ).to(ptu.device)

        self.optimizer = optim.Adam(self.reward_net.parameters(), learning_rate)

    def predict_rewards(self, obs: np.ndarray, acts: np.ndarray, training=False) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single predicted reward (as a numpy array)."""
        if not training:
            self.eval()
        else:
            self.train()

        batch = True
        if len(obs.shape) == 2:
            batch = False
            obs = obs.reshape((1, ) + obs.shape)
            acts = acts.reshape((1, ) + acts.shape)

        batch_size, n_frames, d = obs.shape
        _, _, d_act = acts.shape
        obs_flat = obs.reshape(batch_size * n_frames, d)
        obs_flat = ptu.from_numpy(obs_flat)
        acts_flat = acts.reshape(batch_size * n_frames, d_act)
        acts_flat = ptu.from_numpy(acts_flat)

        # TODO: predict reward using `reward_net`
        # HINT: you should concatenate observation and action

        inp = torch.cat([obs_flat, acts_flat], dim=-1)
        rewards_flat = self.reward_net(inp).squeeze(-1)

        rewards = rewards_flat.reshape(batch_size, n_frames)

        if not batch:
            rewards = rewards[0]
        if not training:
            rewards = ptu.to_numpy(rewards)
        return rewards

    def predict_preferences(self, obs1, acts1, obs2, acts2):
        # TODO: predict preferences probabiltiy between two trajectories
        # HINT: 1. Get reward for each time step using `self.predict_rewards`
        #       2. Sum the rewards
        #       3. Use Softmax to get probabiltiy

        reward_1 = self.predict_rewards(obs1, acts1, training=True)
        reward_2 = self.predict_rewards(obs2, acts2, training=True)

        reward_sum_1 = reward_1.sum(dim=1, keepdim=True)
        reward_sum_2 = reward_2.sum(dim=1, keepdim=True)
        pred = F.softmax(torch.cat([reward_sum_1, reward_sum_2], dim=1), dim=1)

        return pred

    def compute_loss_and_accuracy(self, obs1, acts1, obs2, acts2, prefs):
        # TODO: Calculate the reward funciton loss
        # HINT: 1. Get preferences
        #       2. Calculate loss function
        prefs = ptu.from_numpy(prefs)
        prefs = torch.concat([1 - prefs, prefs], dim=1)

        preds = self.predict_preferences(obs1, acts1, obs2, acts2)
        loss = -torch.sum(prefs * torch.log(preds + 1e-8), dim=1).mean()

        accuracy = (preds.argmax(dim=1) == prefs.argmax(dim=1)).float().mean()

        return loss, accuracy

    def train_step(self, obs1, acts1, obs2, acts2, prefs):
        # TODO: Perform the training step

        self.optimizer.zero_grad()
        loss, accuracy = self.compute_loss_and_accuracy(obs1, acts1, obs2, acts2, prefs)
        loss.backward()
        self.optimizer.step()

        return loss.item(), accuracy.item()
