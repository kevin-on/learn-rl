from typing import Protocol

import torch


class PolicyDistribution(Protocol):
    def sample(self) -> torch.Tensor:
        raise NotImplementedError

    def deterministic(self) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError


class CategoricalPolicyDistribution:
    """Discrete policy with logits (batch, num_actions).

    Samples and deterministic actions are shaped (batch,). Log probabilities and
    entropies are also shaped (batch,).
    """

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self._distribution = torch.distributions.Categorical(logits=logits)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def deterministic(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(actions.to(dtype=torch.int64))

    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy()


class DiagGaussianPolicyDistribution:
    """Continuous policy with mean/log_std/actions (batch, *action_shape).

    Samples and deterministic actions keep that shape. Log probabilities and
    entropies are summed over action dimensions and shaped (batch,).
    """

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        self.mean = mean
        self.log_std = torch.clamp(log_std, log_std_min, log_std_max)
        self._distribution = torch.distributions.Normal(mean, self.log_std.exp())

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def deterministic(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self._distribution.log_prob(actions.to(dtype=self.mean.dtype))
        return _sum_action_dimensions(log_prob)

    def entropy(self) -> torch.Tensor:
        return _sum_action_dimensions(self._distribution.entropy())


def _sum_action_dimensions(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        msg = "continuous policy tensors must include batch and action dimensions."
        raise ValueError(msg)
    return values.flatten(start_dim=1).sum(dim=1)
