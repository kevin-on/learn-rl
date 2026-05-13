import torch

from rl_math import compute_discounted_rollout_returns, compute_gae


def test_compute_gae_handles_terminal_bootstrap() -> None:
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    terminated = torch.tensor([[False], [True]])
    truncated = torch.zeros_like(terminated)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        discount_factor=0.9,
        gae_lambda=1.0,
    )

    expected = torch.tensor([[1.9], [1.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_compute_discounted_rollout_returns_splits_done_boundaries() -> None:
    rewards = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    next_values = torch.tensor(
        [
            [11.0, 101.0],
            [12.0, 102.0],
            [13.0, 103.0],
            [14.0, 104.0],
        ]
    )
    terminated = torch.tensor(
        [
            [False, False],
            [True, False],
            [False, False],
            [False, False],
        ]
    )
    truncated = torch.tensor(
        [
            [False, False],
            [False, True],
            [False, False],
            [False, False],
        ]
    )

    returns = compute_discounted_rollout_returns(
        rewards=rewards,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        discount_factor=0.5,
    )

    expected = torch.tensor(
        [
            [2.0, 45.5],
            [2.0, 71.0],
            [8.5, 76.0],
            [11.0, 92.0],
        ]
    )
    torch.testing.assert_close(returns, expected)
