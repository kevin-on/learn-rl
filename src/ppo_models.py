import math
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

import torch
from einops import rearrange
from torch import nn


class ActorCriticModel(Protocol):
    def __call__(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits, state_values) for a batch of observations."""
        raise NotImplementedError

    def parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError


class MLPActorCriticNet(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        num_actions: int,
        hidden_sizes: list[int],
    ) -> None:
        super().__init__()
        observation_size = math.prod(observation_shape)
        if observation_size <= 0:
            raise ValueError("observation_size must be positive.")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty.")
        if any(hidden_size <= 0 for hidden_size in hidden_sizes):
            raise ValueError("hidden_sizes values must be positive.")

        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
            input_size = hidden_size

        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_size, num_actions)
        self.value_head = nn.Linear(input_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observations = observations.to(dtype=torch.float32)
        observations = rearrange(observations, "batch ... -> batch (...)")
        features = self.trunk(observations)
        policy_logits = self.policy_head(features)
        state_values = self.value_head(features).squeeze(-1)
        return policy_logits, state_values


class AtariActorCriticNet(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        num_actions: int,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError("observation_shape must be (channels, height, width).")
        if any(dimension <= 0 for dimension in observation_shape):
            raise ValueError("observation_shape dimensions must be positive.")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        channels, height, width = observation_shape
        self.convs = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_output = self.convs(torch.zeros(1, channels, height, width))
        conv_output_size = int(conv_output.shape[1])

        self.fc = nn.Sequential(nn.Linear(conv_output_size, hidden_size), nn.ReLU())
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observations = observations.to(dtype=torch.float32) / 255.0
        features = self.fc(self.convs(observations))
        policy_logits = self.policy_head(features)
        state_values = self.value_head(features).squeeze(-1)
        return policy_logits, state_values


type PPOModelFactory = type[nn.Module]


PPO_MODEL_FACTORIES: dict[str, PPOModelFactory] = {
    "mlp": MLPActorCriticNet,
    "atari_cnn": AtariActorCriticNet,
}


def build_ppo_model(
    *,
    name: str,
    observation_shape: tuple[int, ...],
    num_actions: int,
    kwargs: Mapping[str, Any],
) -> nn.Module:
    factory = PPO_MODEL_FACTORIES.get(name)
    if factory is None:
        known_models = ", ".join(sorted(PPO_MODEL_FACTORIES))
        msg = f"Unknown PPO model {name!r}; expected one of: {known_models}."
        raise ValueError(msg)

    try:
        return factory(
            observation_shape=observation_shape,
            num_actions=num_actions,
            **dict(kwargs),
        )
    except TypeError as exc:
        msg = f"Invalid kwargs for PPO model {name!r}: {exc}"
        raise ValueError(msg) from exc
