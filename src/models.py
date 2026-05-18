import math
from collections.abc import Mapping
from typing import Any

import torch
from einops import rearrange
from torch import nn

from envs import ActionSpec, BoxActionSpec, DiscreteActionSpec
from policies import CategoricalPolicyDistribution, DiagGaussianPolicyDistribution

ORTHOGONAL_HIDDEN_GAIN = math.sqrt(2.0)
ORTHOGONAL_POLICY_HEAD_GAIN = 0.01
ORTHOGONAL_VALUE_HEAD_GAIN = 1.0


def validate_hidden_sizes(hidden_sizes: list[int]) -> None:
    if not hidden_sizes:
        raise ValueError("hidden_sizes must not be empty.")
    if any(hidden_size <= 0 for hidden_size in hidden_sizes):
        raise ValueError("hidden_sizes values must be positive.")


def build_mlp_layers(
    *,
    input_size: int,
    hidden_sizes: list[int],
    activation: str = "relu",
    layer_norm: bool = False,
) -> tuple[nn.Sequential, int]:
    if input_size <= 0:
        raise ValueError("input_size must be positive.")
    validate_hidden_sizes(hidden_sizes)

    layers: list[nn.Module] = []
    layer_input_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(layer_input_size, hidden_size))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(_activation_layer(activation))
        layer_input_size = hidden_size
    return nn.Sequential(*layers), layer_input_size


def _activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()

    msg = f"activation must be one of: relu, tanh; got {name!r}."
    raise ValueError(msg)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        num_actions: int,
        hidden_sizes: list[int],
        activation: str = "relu",
        layer_norm: bool = False,
        orthogonal_init: bool = False,
    ) -> None:
        super().__init__()
        observation_size = math.prod(observation_shape)
        if observation_size <= 0:
            raise ValueError("observation_size must be positive.")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")

        self.trunk, trunk_output_size = build_mlp_layers(
            input_size=observation_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.q_head = nn.Linear(trunk_output_size, num_actions)
        if orthogonal_init:
            _orthogonal_init_trunk(self.trunk)
            _orthogonal_init_linear(self.q_head, gain=ORTHOGONAL_VALUE_HEAD_GAIN)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(dtype=torch.float32)
        observations = rearrange(observations, "batch ... -> batch (...)")
        return self.q_head(self.trunk(observations))


class DiscreteActorCriticMLP(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        num_actions: int,
        hidden_sizes: list[int],
        activation: str = "relu",
        layer_norm: bool = False,
        orthogonal_init: bool = False,
    ) -> None:
        super().__init__()
        observation_size = math.prod(observation_shape)
        if observation_size <= 0:
            raise ValueError("observation_size must be positive.")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")

        self.policy_trunk, policy_output_size = build_mlp_layers(
            input_size=observation_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.value_trunk, value_output_size = build_mlp_layers(
            input_size=observation_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.policy_head = nn.Linear(policy_output_size, num_actions)
        self.value_head = nn.Linear(value_output_size, 1)
        if orthogonal_init:
            _orthogonal_init_trunk(self.policy_trunk)
            _orthogonal_init_trunk(self.value_trunk)
            _orthogonal_init_linear(
                self.policy_head,
                gain=ORTHOGONAL_POLICY_HEAD_GAIN,
            )
            _orthogonal_init_linear(
                self.value_head,
                gain=ORTHOGONAL_VALUE_HEAD_GAIN,
            )

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[CategoricalPolicyDistribution, torch.Tensor]:
        observations = observations.to(dtype=torch.float32)
        observations = rearrange(observations, "batch ... -> batch (...)")
        policy_features = self.policy_trunk(observations)
        value_features = self.value_trunk(observations)
        policy_logits = self.policy_head(policy_features)
        state_values = self.value_head(value_features).squeeze(-1)
        return CategoricalPolicyDistribution(policy_logits), state_values


class ContinuousActorCriticMLP(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        hidden_sizes: list[int],
        init_log_std: float,
        log_std_min: float,
        log_std_max: float,
        activation: str = "relu",
        layer_norm: bool = False,
        orthogonal_init: bool = False,
    ) -> None:
        super().__init__()
        observation_size = math.prod(observation_shape)
        action_size = math.prod(action_shape)
        if observation_size <= 0:
            raise ValueError("observation_size must be positive.")
        if action_size <= 0:
            raise ValueError("action_size must be positive.")

        self.action_shape = action_shape
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.policy_trunk, policy_output_size = build_mlp_layers(
            input_size=observation_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.value_trunk, value_output_size = build_mlp_layers(
            input_size=observation_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.mean_head = nn.Linear(policy_output_size, action_size)
        self.log_std = nn.Parameter(torch.full((action_size,), float(init_log_std)))
        self.value_head = nn.Linear(value_output_size, 1)
        if orthogonal_init:
            _orthogonal_init_trunk(self.policy_trunk)
            _orthogonal_init_trunk(self.value_trunk)
            _orthogonal_init_linear(
                self.mean_head,
                gain=ORTHOGONAL_POLICY_HEAD_GAIN,
            )
            _orthogonal_init_linear(
                self.value_head,
                gain=ORTHOGONAL_VALUE_HEAD_GAIN,
            )

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[DiagGaussianPolicyDistribution, torch.Tensor]:
        observations = observations.to(dtype=torch.float32)
        observations = rearrange(observations, "batch ... -> batch (...)")
        policy_features = self.policy_trunk(observations)
        value_features = self.value_trunk(observations)
        mean = self.mean_head(policy_features).reshape(
            observations.shape[0],
            *self.action_shape,
        )
        log_std = self.log_std.reshape(self.action_shape).expand_as(mean)
        state_values = self.value_head(value_features).squeeze(-1)
        return (
            DiagGaussianPolicyDistribution(
                mean=mean,
                log_std=log_std,
                log_std_min=self.log_std_min,
                log_std_max=self.log_std_max,
            ),
            state_values,
        )


class DDPGActorCriticMLP(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_spec: BoxActionSpec,
        hidden_sizes: list[int],
    ) -> None:
        super().__init__()
        validate_hidden_sizes(hidden_sizes)

        observation_size = math.prod(observation_shape)
        action_size = math.prod(action_spec.shape)
        if observation_size <= 0:
            raise ValueError("observation_size must be positive.")
        if action_size <= 0:
            raise ValueError("action_size must be positive.")

        action_low, action_high = _validated_box_action_bounds(
            action_spec,
            algorithm_name="DDPG",
        )

        self.action_shape = action_spec.shape
        self.actor = _build_mlp_with_output(
            input_size=observation_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size,
            output_activation=nn.Tanh(),
        )
        self.critic = _build_mlp_with_output(
            input_size=observation_size + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )
        self.register_buffer(
            "action_scale",
            (action_high - action_low) / 2.0,
        )
        self.register_buffer(
            "action_bias",
            (action_high + action_low) / 2.0,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.act(observations)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(dtype=torch.float32)
        observations = rearrange(observations, "batch ... -> batch (...)")
        normalized_actions = self.actor(observations).reshape(
            observations.shape[0],
            *self.action_shape,
        )
        return normalized_actions * self.action_scale + self.action_bias

    def q(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        observations = observations.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        observations = rearrange(observations, "batch ... -> batch (...)")
        actions = rearrange(actions, "batch ... -> batch (...)")
        return self.critic(torch.cat([observations, actions], dim=1))


class TD3ActorCriticMLP(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_spec: BoxActionSpec,
        hidden_sizes: list[int],
    ) -> None:
        super().__init__()
        validate_hidden_sizes(hidden_sizes)

        observation_size = math.prod(observation_shape)
        action_size = math.prod(action_spec.shape)
        if observation_size <= 0:
            raise ValueError("observation_size must be positive.")
        if action_size <= 0:
            raise ValueError("action_size must be positive.")

        action_low, action_high = _validated_box_action_bounds(
            action_spec,
            algorithm_name="TD3",
        )

        self.action_shape = action_spec.shape
        self.actor = _build_mlp_with_output(
            input_size=observation_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size,
            output_activation=nn.Tanh(),
        )
        self.critic1 = _build_mlp_with_output(
            input_size=observation_size + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )
        self.critic2 = _build_mlp_with_output(
            input_size=observation_size + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )
        self.register_buffer(
            "action_scale",
            (action_high - action_low) / 2.0,
        )
        self.register_buffer(
            "action_bias",
            (action_high + action_low) / 2.0,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.act(observations)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(dtype=torch.float32)
        observations = rearrange(observations, "batch ... -> batch (...)")
        normalized_actions = self.actor(observations).reshape(
            observations.shape[0],
            *self.action_shape,
        )
        return normalized_actions * self.action_scale + self.action_bias

    def q1(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.critic1(self._critic_input(observations, actions))

    def q2(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.critic2(self._critic_input(observations, actions))

    def q_pair(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critic_input = self._critic_input(observations, actions)
        return self.critic1(critic_input), self.critic2(critic_input)

    def _critic_input(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        observations = observations.to(dtype=torch.float32)
        actions = actions.to(dtype=torch.float32)
        observations = rearrange(observations, "batch ... -> batch (...)")
        actions = rearrange(actions, "batch ... -> batch (...)")
        return torch.cat([observations, actions], dim=1)


def _validated_box_action_bounds(
    action_spec: BoxActionSpec,
    *,
    algorithm_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_low = torch.as_tensor(action_spec.low, dtype=torch.float32)
    action_high = torch.as_tensor(action_spec.high, dtype=torch.float32)
    if action_low.shape != action_spec.shape or action_high.shape != action_spec.shape:
        msg = "action bounds must match action_spec.shape."
        raise ValueError(msg)
    if not torch.all(torch.isfinite(action_low)) or not torch.all(
        torch.isfinite(action_high)
    ):
        raise ValueError(f"{algorithm_name} requires finite Box action bounds.")
    if not torch.all(action_low < action_high):
        raise ValueError("action_spec.low must be less than action_spec.high.")
    return action_low, action_high


def _build_mlp_with_output(
    *,
    input_size: int,
    hidden_sizes: list[int],
    output_size: int,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    trunk, trunk_output_size = build_mlp_layers(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
    )
    layers = [
        *trunk.children(),
        nn.Linear(trunk_output_size, output_size),
    ]
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class ActorCriticCNN(nn.Module):
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

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[CategoricalPolicyDistribution, torch.Tensor]:
        observations = observations.to(dtype=torch.float32) / 255.0
        features = self.fc(self.convs(observations))
        policy_logits = self.policy_head(features)
        state_values = self.value_head(features).squeeze(-1)
        return CategoricalPolicyDistribution(policy_logits), state_values


type ModelFactory = type[nn.Module]


def _orthogonal_init_trunk(trunk: nn.Module) -> None:
    for module in trunk.modules():
        if isinstance(module, nn.Linear):
            _orthogonal_init_linear(module, gain=ORTHOGONAL_HIDDEN_GAIN)


def _orthogonal_init_linear(module: nn.Linear, *, gain: float) -> None:
    nn.init.orthogonal_(module.weight, gain=gain)
    nn.init.constant_(module.bias, 0.0)


Q_MODEL_FACTORIES: dict[str, ModelFactory] = {
    "mlp": QNetwork,
}


DISCRETE_ACTOR_CRITIC_FACTORIES: dict[str, ModelFactory] = {
    "discrete_mlp": DiscreteActorCriticMLP,
    "atari_cnn": ActorCriticCNN,
}


BOX_ACTOR_CRITIC_FACTORIES: dict[str, ModelFactory] = {
    "continuous_mlp": ContinuousActorCriticMLP,
}


DDPG_ACTOR_CRITIC_FACTORIES: dict[str, ModelFactory] = {
    "ddpg_mlp": DDPGActorCriticMLP,
}


TD3_ACTOR_CRITIC_FACTORIES: dict[str, ModelFactory] = {
    "td3_mlp": TD3ActorCriticMLP,
}


def build_q_model(
    *,
    name: str,
    observation_shape: tuple[int, ...],
    num_actions: int,
    kwargs: Mapping[str, Any],
) -> nn.Module:
    factory = Q_MODEL_FACTORIES.get(name)
    if factory is None:
        known_models = ", ".join(sorted(Q_MODEL_FACTORIES))
        msg = f"Unknown Q model {name!r}; expected one of: {known_models}."
        raise ValueError(msg)

    try:
        return factory(
            observation_shape=observation_shape,
            num_actions=num_actions,
            **dict(kwargs),
        )
    except TypeError as exc:
        msg = f"Invalid kwargs for Q model {name!r}: {exc}"
        raise ValueError(msg) from exc


def build_actor_critic_model(
    *,
    name: str,
    observation_shape: tuple[int, ...],
    action_spec: ActionSpec,
    kwargs: Mapping[str, Any],
) -> nn.Module:
    try:
        if isinstance(action_spec, DiscreteActionSpec):
            factory = DISCRETE_ACTOR_CRITIC_FACTORIES.get(name)
            if factory is None:
                known_models = ", ".join(sorted(DISCRETE_ACTOR_CRITIC_FACTORIES))
                msg = (
                    f"Unknown discrete actor-critic model {name!r}; "
                    f"expected one of: {known_models}."
                )
                raise ValueError(msg)
            return factory(
                observation_shape=observation_shape,
                num_actions=action_spec.num_actions,
                **dict(kwargs),
            )

        if isinstance(action_spec, BoxActionSpec):
            factory = BOX_ACTOR_CRITIC_FACTORIES.get(name)
            if factory is None:
                known_models = ", ".join(sorted(BOX_ACTOR_CRITIC_FACTORIES))
                msg = (
                    f"Unknown Box actor-critic model {name!r}; "
                    f"expected one of: {known_models}."
                )
                raise ValueError(msg)
            return factory(
                observation_shape=observation_shape,
                action_shape=action_spec.shape,
                **dict(kwargs),
            )

        msg = f"Unsupported action spec: {type(action_spec).__name__}."
        raise ValueError(msg)
    except TypeError as exc:
        msg = f"Invalid kwargs for actor-critic model {name!r}: {exc}"
        raise ValueError(msg) from exc


def build_ddpg_actor_critic_model(
    *,
    name: str,
    observation_shape: tuple[int, ...],
    action_spec: ActionSpec,
    kwargs: Mapping[str, Any],
) -> nn.Module:
    if not isinstance(action_spec, BoxActionSpec):
        raise ValueError("DDPG requires a Box action space.")

    factory = DDPG_ACTOR_CRITIC_FACTORIES.get(name)
    if factory is None:
        known_models = ", ".join(sorted(DDPG_ACTOR_CRITIC_FACTORIES))
        msg = f"Unknown DDPG actor-critic model {name!r}; expected one of: {known_models}."
        raise ValueError(msg)

    try:
        return factory(
            observation_shape=observation_shape,
            action_spec=action_spec,
            **dict(kwargs),
        )
    except TypeError as exc:
        msg = f"Invalid kwargs for DDPG actor-critic model {name!r}: {exc}"
        raise ValueError(msg) from exc


def build_td3_actor_critic_model(
    *,
    name: str,
    observation_shape: tuple[int, ...],
    action_spec: ActionSpec,
    kwargs: Mapping[str, Any],
) -> nn.Module:
    if not isinstance(action_spec, BoxActionSpec):
        raise ValueError("TD3 requires a Box action space.")

    factory = TD3_ACTOR_CRITIC_FACTORIES.get(name)
    if factory is None:
        known_models = ", ".join(sorted(TD3_ACTOR_CRITIC_FACTORIES))
        msg = (
            f"Unknown TD3 actor-critic model {name!r}; expected one of: {known_models}."
        )
        raise ValueError(msg)

    try:
        return factory(
            observation_shape=observation_shape,
            action_spec=action_spec,
            **dict(kwargs),
        )
    except TypeError as exc:
        msg = f"Invalid kwargs for TD3 actor-critic model {name!r}: {exc}"
        raise ValueError(msg) from exc
