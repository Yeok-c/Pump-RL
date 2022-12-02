from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import DDPG
from stable_baselines3.td3.policies import TD3Policy
# from stable_baselines3.common.policies import ActorCriticPolicy
from policy import MultiInputNet


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        calib_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Module(
            MultiInputNet(feature_dim=feature_dim/2, calib_dim=feature_dim/2, output_dim=last_layer_dim_pi)
            )
        # self.policy_net = nn.Sequential(
        #     nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        # )
                
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomTD3Policy(TD3Policy): #ActorCriticPolicy
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        # lr_schedule: Callable[[float], float],
        # net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        # activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomTD3Policy, self).__init__(
            observation_space,
            action_space,
            # lr_schedule,
            # net_arch,
            # activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    # def make_actor(self) -> None:
    #     self.mlp_extractor = CustomNetwork(self.features_dim)
    #     pass

    # def _build_mlp_extractor(self) -> None:
    #     """
    #     Create the policy and value networks.
    #     Part of the layers can be shared.
    #     """
    #     # Note: If net_arch is None and some features extractor is used,
    #     #       net_arch here is an empty list and mlp_extractor does not
    #     #       really contain any layers (acts like an identity module).
    #     self.mlp_extractor = MlpExtractor(
    #         self.features_dim,
    #         net_arch=self.net_arch,
    #         activation_fn=self.activation_fn,
    #         device=self.device,
    #     )

    # Turned into
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

# test code
if __name__ == '__main__':
    model = DDPG(CustomTD3Policy, "Pendulum-v1", verbose=1)
    model.learn(5000) 