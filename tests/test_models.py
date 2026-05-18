import pytest
import torch

from envs import BoxActionSpec, DiscreteActionSpec
from models import (
    build_actor_critic_model,
    build_ddpg_actor_critic_model,
    build_q_model,
)
from policies import CategoricalPolicyDistribution, DiagGaussianPolicyDistribution


def test_q_network_output_shape() -> None:
    model = build_q_model(
        name="mlp",
        observation_shape=(4,),
        num_actions=2,
        kwargs={"hidden_sizes": [8]},
    )
    output = model(torch.zeros(3, 4))
    assert output.shape == (3, 2)


def test_q_network_supports_layer_norm_kwargs() -> None:
    model = build_q_model(
        name="mlp",
        observation_shape=(4,),
        num_actions=2,
        kwargs={
            "hidden_sizes": [8],
            "activation": "relu",
            "layer_norm": True,
            "orthogonal_init": True,
        },
    )

    output = model(torch.zeros(3, 4))

    assert output.shape == (3, 2)
    assert any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())


def test_discrete_actor_critic_mlp_output_shapes() -> None:
    model = build_actor_critic_model(
        name="discrete_mlp",
        observation_shape=(4,),
        action_spec=DiscreteActionSpec(num_actions=2),
        kwargs={"hidden_sizes": [8]},
    )
    dist, values = model(torch.zeros(3, 4))
    assert isinstance(dist, CategoricalPolicyDistribution)
    assert dist.sample().shape == (3,)
    assert dist.log_prob(torch.zeros(3, dtype=torch.int64)).shape == (3,)
    assert dist.entropy().shape == (3,)
    assert values.shape == (3,)
    assert hasattr(model, "policy_trunk")
    assert hasattr(model, "value_trunk")
    assert not hasattr(model, "trunk")


def test_discrete_actor_critic_mlp_supports_relu_layer_norm() -> None:
    model = build_actor_critic_model(
        name="discrete_mlp",
        observation_shape=(4,),
        action_spec=DiscreteActionSpec(num_actions=2),
        kwargs={
            "hidden_sizes": [8],
            "activation": "relu",
            "layer_norm": True,
            "orthogonal_init": True,
        },
    )

    dist, values = model(torch.zeros(3, 4))

    assert isinstance(dist, CategoricalPolicyDistribution)
    assert values.shape == (3,)
    assert (
        sum(isinstance(module, torch.nn.LayerNorm) for module in model.modules()) == 2
    )


def test_continuous_actor_critic_mlp_output_shapes() -> None:
    model = build_actor_critic_model(
        name="continuous_mlp",
        observation_shape=(3,),
        action_spec=BoxActionSpec(
            shape=(2,),
            low=torch.full((2,), -1.0).numpy(),
            high=torch.full((2,), 1.0).numpy(),
            dtype=torch.full((2,), 0.0).numpy().dtype,
        ),
        kwargs={
            "hidden_sizes": [8],
            "init_log_std": 0.0,
            "log_std_min": -20.0,
            "log_std_max": 2.0,
        },
    )
    dist, values = model(torch.zeros(3, 3))
    assert isinstance(dist, DiagGaussianPolicyDistribution)
    actions = dist.sample()
    assert actions.shape == (3, 2)
    assert dist.deterministic().shape == (3, 2)
    assert dist.log_prob(actions).shape == (3,)
    assert dist.entropy().shape == (3,)
    assert values.shape == (3,)
    assert hasattr(model, "policy_trunk")
    assert hasattr(model, "value_trunk")
    assert not hasattr(model, "trunk")


def test_continuous_actor_critic_mlp_supports_relu_layer_norm() -> None:
    model = build_actor_critic_model(
        name="continuous_mlp",
        observation_shape=(3,),
        action_spec=BoxActionSpec(
            shape=(2,),
            low=torch.full((2,), -1.0).numpy(),
            high=torch.full((2,), 1.0).numpy(),
            dtype=torch.full((2,), 0.0).numpy().dtype,
        ),
        kwargs={
            "hidden_sizes": [8],
            "activation": "relu",
            "layer_norm": True,
            "orthogonal_init": True,
            "init_log_std": 0.0,
        },
    )

    dist, values = model(torch.zeros(3, 3))

    assert isinstance(dist, DiagGaussianPolicyDistribution)
    assert dist.deterministic().shape == (3, 2)
    assert values.shape == (3,)
    assert (
        sum(isinstance(module, torch.nn.LayerNorm) for module in model.modules()) == 2
    )


def test_ddpg_actor_critic_mlp_output_shapes_and_bounds() -> None:
    action_spec = BoxActionSpec(
        shape=(2,),
        low=torch.tensor([-2.0, -1.0]).numpy(),
        high=torch.tensor([2.0, 3.0]).numpy(),
        dtype=torch.full((2,), 0.0).numpy().dtype,
    )
    model = build_ddpg_actor_critic_model(
        name="ddpg_mlp",
        observation_shape=(3,),
        action_spec=action_spec,
        kwargs={"hidden_sizes": [8, 8]},
    )

    observations = torch.zeros(4, 3)
    actions = model.act(observations)
    q_values = model.q(observations, actions)

    assert actions.shape == (4, 2)
    assert torch.all(actions >= torch.tensor([-2.0, -1.0]))
    assert torch.all(actions <= torch.tensor([2.0, 3.0]))
    assert q_values.shape == (4, 1)
    assert hasattr(model, "actor")
    assert hasattr(model, "critic")
    assert isinstance(model.actor[0], torch.nn.Linear)
    assert model.actor[0].in_features == 3
    assert isinstance(model.actor[-1], torch.nn.Tanh)
    assert isinstance(model.critic[0], torch.nn.Linear)
    assert model.critic[0].in_features == 5
    assert not hasattr(model, "state_layer")
    assert not hasattr(model, "post_action_trunk")


def test_ddpg_actor_critic_mlp_rejects_empty_hidden_sizes() -> None:
    action_spec = BoxActionSpec(
        shape=(1,),
        low=torch.tensor([-1.0]).numpy(),
        high=torch.tensor([1.0]).numpy(),
        dtype=torch.full((1,), 0.0).numpy().dtype,
    )

    with pytest.raises(ValueError, match="hidden_sizes must not be empty"):
        build_ddpg_actor_critic_model(
            name="ddpg_mlp",
            observation_shape=(3,),
            action_spec=action_spec,
            kwargs={"hidden_sizes": []},
        )


def test_ddpg_actor_critic_mlp_requires_hidden_sizes() -> None:
    action_spec = BoxActionSpec(
        shape=(1,),
        low=torch.tensor([-1.0]).numpy(),
        high=torch.tensor([1.0]).numpy(),
        dtype=torch.full((1,), 0.0).numpy().dtype,
    )

    with pytest.raises(ValueError, match="Invalid kwargs"):
        build_ddpg_actor_critic_model(
            name="ddpg_mlp",
            observation_shape=(3,),
            action_spec=action_spec,
            kwargs={},
        )


def test_atari_cnn_remains_shared() -> None:
    model = build_actor_critic_model(
        name="atari_cnn",
        observation_shape=(4, 84, 84),
        action_spec=DiscreteActionSpec(num_actions=2),
        kwargs={"hidden_size": 8},
    )
    dist, values = model(torch.zeros(3, 4, 84, 84))
    assert isinstance(dist, CategoricalPolicyDistribution)
    assert dist.sample().shape == (3,)
    assert dist.log_prob(torch.zeros(3, dtype=torch.int64)).shape == (3,)
    assert dist.entropy().shape == (3,)
    assert values.shape == (3,)
    assert hasattr(model, "convs")
    assert hasattr(model, "fc")
