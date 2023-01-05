from stable_baselines3.td3.policies import TD3Policy, Actor # right click to see
from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch as th
from torch import nn
import torch.functional as F

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule


class Network_Custom(nn.Module):
    def __init__(self, features_dim, embedding_dim, output_dim):
        super().__init__()
        self.input_layer_f_length = features_dim-embedding_dim # int(feature_dim/2)
        self.input_layer_c_length = embedding_dim # feature_dim - self.input_layer_f_length

        self.fc_f1 = nn.Linear(self.input_layer_f_length, 32) #supose your input shape is 100
        # self.fc_f2 = nn.Linear(32, 64) #supose your input shape is 100
        # self.fc_f3 = nn.Linear(64, 64) #supose your input shape is 100
        self.fc_f4 = nn.Linear(32, 32) #supose your input shape is 100

        self.fc_c1 = nn.Linear(self.input_layer_c_length, 32)
        self.fc_c2 = nn.Linear(32, 32)
        
        self.fc = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_layer):
        x = F.relu(self.fc_f1(input_layer[:,:self.input_layer_f_length]))
        # x = F.relu(self.fc_f1(torch.squeeze(input_layer[:,:self.input_layer_f_length], 0)))
        # x = F.relu(self.fc_f2(x))
        # x = F.relu(self.fc_f3(x))
        x = F.relu(self.fc_f4(x))

        y = F.relu(self.fc_c1(input_layer[:,self.input_layer_f_length:]))
        # y = F.relu(self.fc_c1(torch.squeeze(input_layer[:,self.input_layer_f_length:], 0)))
        y = F.relu(self.fc_c2(y))

        x = th.cat((x, y), dim=1)
        x = self.fc(x)

        x[:,1:] = self.softmax(x[:,1:])
        
        return 
class Actor_Custom(Actor):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        embedding_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn, 
            normalize_images=normalize_images,
        )

        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)


        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # # Deterministic action
        # self.mu = nn.Sequential(*actor_net)
        self.mu = Network_Custom(features_dim, embedding_dim, action_dim)
        
    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        # features = self.extract_features(obs)
        return self.mu(obs)
        

class TD3Policy_embedding(TD3Policy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor_Custom:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor_Custom(**actor_kwargs).to(self.device)