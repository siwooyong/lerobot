"""Value function used by Flow-Noise PPO."""

import torch
from torch import Tensor, nn


class ValueCritic(nn.Module):
    """πRL's π₀.₅ value head over mean VLM prefix features."""

    def __init__(self, context_dim: int, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        self.net = self._head(context_dim, hidden_dims)
        self._init_weights()

    @staticmethod
    def _head(context_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        input_dim = context_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1, bias=True))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                if module is self.net[-1]:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Predict a chunk value from the mean VLM prefix feature."""
        return self.net(features.to(self.net[0].weight)).squeeze(-1)
