import torch

from models import build_actor_critic_model, build_q_model


def test_q_network_output_shape() -> None:
    model = build_q_model(
        name="mlp",
        observation_shape=(4,),
        num_actions=2,
        kwargs={"hidden_sizes": [8]},
    )
    output = model(torch.zeros(3, 4))
    assert output.shape == (3, 2)


def test_shared_actor_critic_mlp_output_shapes() -> None:
    model = build_actor_critic_model(
        name="shared_mlp",
        observation_shape=(4,),
        num_actions=2,
        kwargs={"hidden_sizes": [8]},
    )
    logits, values = model(torch.zeros(3, 4))
    assert logits.shape == (3, 2)
    assert values.shape == (3,)


def test_unshared_actor_critic_mlp_output_shapes() -> None:
    model = build_actor_critic_model(
        name="unshared_mlp",
        observation_shape=(4,),
        num_actions=2,
        kwargs={"hidden_sizes": [8]},
    )
    logits, values = model(torch.zeros(3, 4))
    assert logits.shape == (3, 2)
    assert values.shape == (3,)


def test_actor_critic_mlp_alias_remains_shared() -> None:
    model = build_actor_critic_model(
        name="mlp",
        observation_shape=(4,),
        num_actions=2,
        kwargs={"hidden_sizes": [8]},
    )
    assert hasattr(model, "trunk")
