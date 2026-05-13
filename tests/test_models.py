import torch

from envs import BoxActionSpec, DiscreteActionSpec
from models import build_actor_critic_model, build_q_model
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


def test_actor_critic_mlp_output_shapes() -> None:
    model = build_actor_critic_model(
        name="mlp",
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


def test_continuous_actor_critic_mlp_output_shapes() -> None:
    model = build_actor_critic_model(
        name="mlp",
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
