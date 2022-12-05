# import gym
# import torch as th
# from torch import nn

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

#         extractors = {}

#         total_concat_size = 0
#         # We need to know size of the output of this extractor,
#         # so go over all the spaces and compute output feature sizes
#         for key, subspace in observation_space.spaces.items():
#             if key == "image":
#                 # We will just downsample one channel of the image by 4x4 and flatten.
#                 # Assume the image is single-channel (subspace.shape[0] == 0)
#                 extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
#                 total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
#             elif key == "vector":
#                 # Run through a simple MLP
#                 extractors[key] = nn.Linear(subspace.shape[0], 16)
#                 total_concat_size += 16

#         self.extractors = nn.ModuleDict(extractors)

#         # Update the features dim manually
#         self._features_dim = total_concat_size

#     def forward(self, observations) -> th.Tensor:
#         encoded_tensor_list = []

#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         return th.cat(encoded_tensor_list, dim=1)


# class BaseFeaturesExtractor(nn.Module):
#     """
#     Base class that represents a features extractor.

#     :param observation_space:
#     :param features_dim: Number of features extracted.
#     """

#     def __init__(self, observation_space: gym.Space, features_dim: int = 0):
#         super().__init__()
#         assert features_dim > 0
#         self._observation_space = observation_space
#         self._features_dim = features_dim

#     @property
#     def features_dim(self) -> int:
#         return self._features_dim

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         raise NotImplementedError()

        
import torch
from torch import nn
import torch.nn.functional as F

class MultiInputNet(nn.Module):
    def __init__(self, feature_dim=55, embedding_dim=22, output_dim=1):
        super().__init__()
        self.input_layer_f_length = feature_dim-embedding_dim # int(feature_dim/2)
        self.input_layer_c_length = embedding_dim # feature_dim - self.input_layer_f_length

        self.fc_f1 = nn.Linear(self.input_layer_f_length, 32) #supose your input shape is 100
        self.fc_f2 = nn.Linear(32, 64) #supose your input shape is 100
        self.fc_f3 = nn.Linear(64, 64) #supose your input shape is 100
        self.fc_f4 = nn.Linear(64, 32) #supose your input shape is 100

        self.fc_c1 = nn.Linear(self.input_layer_c_length, 32)
        self.fc_c2 = nn.Linear(32, 32)
        
        self.fc = nn.Linear(64, output_dim)

    def forward(self, input_layer):
        x = F.relu(self.fc_f1(input_layer[:,:self.input_layer_f_length]))
        # x = F.relu(self.fc_f1(torch.squeeze(input_layer[:,:self.input_layer_f_length], 0)))
        x = F.relu(self.fc_f2(x))
        x = F.relu(self.fc_f3(x))
        x = F.relu(self.fc_f4(x))

        y = F.relu(self.fc_c1(input_layer[:,self.input_layer_f_length:]))
        # y = F.relu(self.fc_c1(torch.squeeze(input_layer[:,self.input_layer_f_length:], 0)))
        y = F.relu(self.fc_c2(y))

        x = torch.cat((x, y), dim=1)
        x = self.fc(x)
        return x


# test code
if __name__ == '__main__':
    net = MultiInputNet()
    print(net)
    # A = net.forward(torch.zeros(1,55))
    # print(A)