import queue
import random
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.context import SpawnContext, SpawnProcess
from multiprocessing.queues import Queue
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Lock
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from envs import DiscreteActionSpec
from models import build_actor_critic_model
from rl_math import clip_grad_tensors, compute_discounted_returns
from task_adapter import State, VectorTaskAdapter, make_task_adapter


@dataclass(frozen=True)
class A3CLog:
    global_step: int
    worker_id: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    grad_norm: float | None
    rollout_length: int
    episode_return: float | None = None
    episode_length: int | None = None


type A3CLogFn = Callable[[A3CLog], None]


class A3C:
    def __init__(
        self,
        *,
        env_id: str,
        state_size: int,
        num_actions: int,
        model_name: str,
        model_kwargs: dict[str, Any],
        num_workers: int,
        learning_rate: float,
        value_loss_coef: float,
        discount_factor: float,
        rollout_steps: int,
        max_grad_norm: float | None,
        entropy_coef: float,
        rmsprop_alpha: float,
        rmsprop_eps: float,
    ) -> None:
        validate_a3c_hyperparameters(
            state_size=state_size,
            num_actions=num_actions,
            num_workers=num_workers,
            learning_rate=learning_rate,
            value_loss_coef=value_loss_coef,
            discount_factor=discount_factor,
            rollout_steps=rollout_steps,
            max_grad_norm=max_grad_norm,
            entropy_coef=entropy_coef,
            rmsprop_alpha=rmsprop_alpha,
            rmsprop_eps=rmsprop_eps,
        )

        self.env_id = env_id
        self.state_size = state_size
        self.num_actions = num_actions
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs)
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.value_loss_coef = value_loss_coef
        self.discount_factor = discount_factor
        self.rollout_steps = rollout_steps
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.rmsprop_alpha = rmsprop_alpha
        self.rmsprop_eps = rmsprop_eps

        self.global_model = build_actor_critic_model(
            name=self.model_name,
            observation_shape=(state_size,),
            action_spec=DiscreteActionSpec(num_actions=num_actions),
            kwargs=self.model_kwargs,
        ).cpu()
        self.global_model.share_memory()

        self._ctx: SpawnContext = mp.get_context("spawn")
        self._global_step: Synchronized = self._ctx.Value("q", 0)
        self._update_lock: Lock = self._ctx.Lock()
        self._square_avgs = [
            torch.zeros_like(param, memory_format=torch.preserve_format).share_memory_()
            for param in self.global_model.parameters()
        ]

    def train(
        self,
        *,
        num_steps: int,
        seed: int,
        log_fn: A3CLogFn | None = None,
    ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")

        with self._update_lock:
            self._global_step.value = 0
            for square_avg in self._square_avgs:
                square_avg.zero_()

        log_queue: Queue[A3CLog] = self._ctx.Queue()
        processes: list[SpawnProcess] = [
            self._ctx.Process(
                target=worker_main,
                kwargs={
                    "worker_id": worker_id,
                    "env_id": self.env_id,
                    "state_size": self.state_size,
                    "num_actions": self.num_actions,
                    "model_name": self.model_name,
                    "model_kwargs": self.model_kwargs,
                    "global_model": self.global_model,
                    "square_avgs": self._square_avgs,
                    "global_step": self._global_step,
                    "update_lock": self._update_lock,
                    "log_queue": log_queue,
                    "num_steps": num_steps,
                    "seed": seed + worker_id,
                    "learning_rate": self.learning_rate,
                    "value_loss_coef": self.value_loss_coef,
                    "discount_factor": self.discount_factor,
                    "rollout_steps": self.rollout_steps,
                    "max_grad_norm": self.max_grad_norm,
                    "entropy_coef": self.entropy_coef,
                    "rmsprop_alpha": self.rmsprop_alpha,
                    "rmsprop_eps": self.rmsprop_eps,
                },
            )
            for worker_id in range(self.num_workers)
        ]

        for process in processes:
            process.start()

        try:
            self._consume_logs(processes=processes, log_queue=log_queue, log_fn=log_fn)
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise
        finally:
            for process in processes:
                process.join()

        failed_processes = [
            process.exitcode
            for process in processes
            if process.exitcode not in (0, None)
        ]
        if failed_processes:
            msg = f"A3C worker process failed with exit code(s): {failed_processes}"
            raise RuntimeError(msg)

    def snapshot_state_dict(self) -> dict[str, torch.Tensor]:
        with self._update_lock:
            return {
                name: value.detach().clone()
                for name, value in self.global_model.state_dict().items()
            }

    def _consume_logs(
        self,
        *,
        processes: list[SpawnProcess],
        log_queue: Queue[A3CLog],
        log_fn: A3CLogFn | None,
    ) -> None:
        while any(process.is_alive() for process in processes):
            try:
                log = log_queue.get(timeout=0.2)
            except queue.Empty:
                self._raise_for_failed_process(processes)
                continue

            if log_fn is not None:
                log_fn(log)
            self._raise_for_failed_process(processes)

        while True:
            try:
                log = log_queue.get_nowait()
            except queue.Empty:
                break

            if log_fn is not None:
                log_fn(log)

    @staticmethod
    def _raise_for_failed_process(processes: list[SpawnProcess]) -> None:
        failed_processes = [
            process.exitcode
            for process in processes
            if process.exitcode not in (0, None)
        ]
        if failed_processes:
            msg = f"A3C worker process failed with exit code(s): {failed_processes}"
            raise RuntimeError(msg)


def worker_main(
    *,
    worker_id: int,
    env_id: str,
    state_size: int,
    num_actions: int,
    model_name: str,
    model_kwargs: dict[str, Any],
    global_model: nn.Module,
    square_avgs: list[torch.Tensor],
    global_step: Synchronized,
    update_lock: Lock,
    log_queue: Queue[A3CLog],
    num_steps: int,
    seed: int,
    learning_rate: float,
    value_loss_coef: float,
    discount_factor: float,
    rollout_steps: int,
    max_grad_norm: float | None,
    entropy_coef: float,
    rmsprop_alpha: float,
    rmsprop_eps: float,
) -> None:
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_id)
    env.action_space.seed(seed)
    task_adapter = make_task_adapter(env, env_id)
    local_model = build_actor_critic_model(
        name=model_name,
        observation_shape=(state_size,),
        action_spec=DiscreteActionSpec(num_actions=num_actions),
        kwargs=model_kwargs,
    ).cpu()

    observation, _info = env.reset(seed=seed)
    state = task_adapter.encode_observation(observation)
    episode_return = 0.0
    episode_length = 0

    try:
        while True:
            with update_lock:
                if global_step.value >= num_steps:
                    break
                local_model.load_state_dict(global_model.state_dict())

            rollout = _collect_rollout(
                model=local_model,
                state=state,
                env=env,
                task_adapter=task_adapter,
                rollout_steps=rollout_steps,
            )
            state = rollout.next_state
            episode_return += rollout.reward_sum
            episode_length += rollout.length

            loss_info = _compute_loss(
                model=local_model,
                states=rollout.states,
                rewards=rollout.rewards,
                log_probs=rollout.log_probs,
                values=rollout.values,
                entropies=rollout.entropies,
                rollout_end_state=rollout.next_state,
                terminal=rollout.done,
                discount_factor=discount_factor,
                value_loss_coef=value_loss_coef,
                entropy_coef=entropy_coef,
            )

            local_model.zero_grad(set_to_none=True)
            loss_info.loss.backward()
            grads = [param.grad for param in local_model.parameters()]

            with update_lock:
                grad_norm = clip_grad_tensors(grads, max_grad_norm)
                _shared_rmsprop_step(
                    params=list(global_model.parameters()),
                    grads=grads,
                    square_avgs=square_avgs,
                    learning_rate=learning_rate,
                    alpha=rmsprop_alpha,
                    eps=rmsprop_eps,
                )
                global_step.value += rollout.length
                step = int(global_step.value)

            completed_episode_return = None
            completed_episode_length = None
            if rollout.done:
                completed_episode_return = episode_return
                completed_episode_length = episode_length
                observation, _info = env.reset()
                state = task_adapter.encode_observation(observation)
                episode_return = 0.0
                episode_length = 0

            log_queue.put(
                A3CLog(
                    global_step=step,
                    worker_id=worker_id,
                    loss=float(loss_info.loss.item()),
                    policy_loss=float(loss_info.policy_loss.item()),
                    value_loss=float(loss_info.value_loss.item()),
                    entropy=float(loss_info.entropy.item()),
                    grad_norm=grad_norm,
                    rollout_length=rollout.length,
                    episode_return=completed_episode_return,
                    episode_length=completed_episode_length,
                )
            )
    finally:
        env.close()


@dataclass(frozen=True)
class _Rollout:
    states: list[State]
    rewards: list[float]
    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]
    entropies: list[torch.Tensor]
    next_state: State
    done: bool
    length: int
    reward_sum: float


@dataclass(frozen=True)
class _LossInfo:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor


def _collect_rollout(
    *,
    model: nn.Module,
    state: State,
    env: gym.Env[NDArray[np.float32], int],
    task_adapter: VectorTaskAdapter,
    rollout_steps: int,
) -> _Rollout:
    states: list[State] = []
    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    done = False
    reward_sum = 0.0

    while len(states) < rollout_steps and not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        dist, value = model(state_tensor)
        _validate_model_outputs(
            values=value,
            batch_size=1,
        )

        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        entropy = dist.entropy()
        env_action = task_adapter.action_index_to_env_action(int(action_index.item()))

        next_observation, reward, terminated, truncated, _info = env.step(env_action)
        next_state = task_adapter.encode_observation(next_observation)
        done = terminated or truncated

        states.append(state)
        rewards.append(float(reward))
        log_probs.append(log_prob.squeeze(0))
        values.append(value.squeeze(0))
        entropies.append(entropy.squeeze(0))
        reward_sum += float(reward)
        state = next_state

    return _Rollout(
        states=states,
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        entropies=entropies,
        next_state=state,
        done=done,
        length=len(states),
        reward_sum=reward_sum,
    )


def _compute_loss(
    *,
    model: nn.Module,
    states: list[State],
    rewards: list[float],
    log_probs: list[torch.Tensor],
    values: list[torch.Tensor],
    entropies: list[torch.Tensor],
    rollout_end_state: State,
    terminal: bool,
    discount_factor: float,
    value_loss_coef: float,
    entropy_coef: float,
) -> _LossInfo:
    returns = _compute_rollout_returns(
        model=model,
        rewards=rewards,
        rollout_end_state=rollout_end_state,
        terminal=terminal,
        discount_factor=discount_factor,
    )
    values_tensor = torch.stack(values).reshape_as(returns)
    log_probs_tensor = torch.stack(log_probs).reshape_as(returns)
    entropies_tensor = torch.stack(entropies)

    advantages = returns - values_tensor
    policy_loss = -(log_probs_tensor * advantages.detach()).mean()
    value_loss = F.mse_loss(values_tensor, returns)
    entropy = entropies_tensor.mean()
    loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
    return _LossInfo(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
    )


@torch.no_grad()
def _compute_rollout_returns(
    *,
    model: nn.Module,
    rewards: list[float],
    rollout_end_state: State,
    terminal: bool,
    discount_factor: float,
) -> torch.Tensor:
    if terminal:
        bootstrap_value = torch.zeros((), dtype=torch.float32)
    else:
        state_tensor = torch.as_tensor(
            rollout_end_state, dtype=torch.float32
        ).unsqueeze(0)
        _dist, value = model(state_tensor)
        bootstrap_value = value.reshape(())

    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
    return compute_discounted_returns(
        rewards=rewards_tensor,
        bootstrap_value=bootstrap_value,
        discount_factor=discount_factor,
    )


@torch.no_grad()
def _shared_rmsprop_step(
    *,
    params: list[torch.nn.Parameter],
    grads: list[torch.Tensor | None],
    square_avgs: list[torch.Tensor],
    learning_rate: float,
    alpha: float,
    eps: float,
) -> None:
    for param, grad, square_avg in zip(params, grads, square_avgs, strict=True):
        if grad is None:
            continue

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1.0 - alpha)
        denominator = square_avg.sqrt().add_(eps)
        param.addcdiv_(grad, denominator, value=-learning_rate)


def _validate_model_outputs(
    *,
    values: torch.Tensor,
    batch_size: int,
) -> None:
    if values.shape != (batch_size,):
        msg = (
            "Actor-critic model value head must return values with shape "
            f"({batch_size},), got {values.shape}."
        )
        raise ValueError(msg)


def validate_actor_critic_model_shape(
    *,
    state_size: int,
    num_actions: int,
) -> None:
    if state_size <= 0:
        raise ValueError("state_size must be positive.")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive.")


def validate_a3c_hyperparameters(
    *,
    state_size: int,
    num_actions: int,
    num_workers: int,
    learning_rate: float,
    value_loss_coef: float,
    discount_factor: float,
    rollout_steps: int,
    max_grad_norm: float | None,
    entropy_coef: float,
    rmsprop_alpha: float,
    rmsprop_eps: float,
) -> None:
    validate_actor_critic_model_shape(
        state_size=state_size,
        num_actions=num_actions,
    )
    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if value_loss_coef < 0.0:
        raise ValueError("value_loss_coef must be non-negative.")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or None.")
    if entropy_coef < 0.0:
        raise ValueError("entropy_coef must be non-negative.")
    if not 0.0 <= rmsprop_alpha < 1.0:
        raise ValueError("rmsprop_alpha must be in [0, 1).")
    if rmsprop_eps <= 0.0:
        raise ValueError("rmsprop_eps must be positive.")
