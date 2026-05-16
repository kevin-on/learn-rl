import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import ExperimentRunConfig
from envs import (
    DiscreteVecEnv,
    EnvPoolVecEnv,
    NormalizeObservationVecEnv,
    RunningMeanStd,
    VecEnv,
)


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    msg = f"device must be one of: auto, cpu, cuda; got {requested_device}"
    raise ValueError(msg)


def set_random_seeds(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise ValueError("model must have at least one parameter.") from exc
    return parameter.device


def create_run_dir(
    config: ExperimentRunConfig,
    requested_run_dir: Path | None,
) -> Path:
    if requested_run_dir is not None:
        run_dir = requested_run_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_name = f"{timestamp}-{config.experiment.name}-seed{config.seed}"
        run_dir = Path(config.experiment.run_root) / run_name

    run_dir.mkdir(parents=True, exist_ok=requested_run_dir is not None)
    return run_dir


def envpool_kwargs(
    config: ExperimentRunConfig, *, evaluation: bool = False
) -> dict[str, int | bool]:
    atari = config.env.atari
    if atari is None:
        return {}

    kwargs: dict[str, int | bool] = {
        "stack_num": atari.stack_num,
        "frame_skip": atari.frame_skip,
        "noop_max": atari.noop_max,
        "episodic_life": atari.episodic_life,
        "reward_clip": atari.reward_clip,
        "img_height": atari.img_height,
        "img_width": atari.img_width,
        "gray_scale": atari.gray_scale,
    }
    if evaluation:
        kwargs["episodic_life"] = False
        kwargs["reward_clip"] = False
    return kwargs


def make_envpool_env(
    config: ExperimentRunConfig,
    *,
    num_envs: int,
    seed: int,
    evaluation: bool = False,
    observation_rms: RunningMeanStd | None = None,
) -> VecEnv:
    env: VecEnv = EnvPoolVecEnv(
        env_id=config.env.id,
        num_envs=num_envs,
        seed=seed,
        env_kwargs=envpool_kwargs(config, evaluation=evaluation),
    )
    observation_normalization = config.env.observation_normalization
    if observation_normalization is None:
        return env

    if evaluation and observation_rms is None:
        msg = "normalized evaluation envs must share training observation stats."
        raise ValueError(msg)

    return NormalizeObservationVecEnv(
        env,
        training=not evaluation,
        observation_rms=observation_rms,
        clip=observation_normalization.clip,
        epsilon=observation_normalization.epsilon,
    )


def observation_normalization_stats(env: VecEnv) -> RunningMeanStd | None:
    if isinstance(env, NormalizeObservationVecEnv):
        return env.observation_rms
    return None


def observation_normalization_state(
    env: VecEnv,
) -> dict[str, np.ndarray | float] | None:
    observation_rms = observation_normalization_stats(env)
    if observation_rms is None:
        return None

    return {
        "mean": np.array(observation_rms.mean, copy=True),
        "var": np.array(observation_rms.var, copy=True),
        "count": float(observation_rms.count),
    }


def running_mean_std_from_state(state: dict[str, object]) -> RunningMeanStd:
    mean = np.asarray(state["mean"], dtype=np.float64)
    var = np.asarray(state["var"], dtype=np.float64)
    count = float(state["count"])
    if mean.shape != var.shape:
        msg = f"observation normalization mean/var shape mismatch: {mean.shape} != {var.shape}"
        raise ValueError(msg)

    observation_rms = RunningMeanStd(shape=tuple(int(size) for size in mean.shape))
    observation_rms.mean = np.array(mean, copy=True)
    observation_rms.var = np.array(var, copy=True)
    observation_rms.count = count
    return observation_rms


def save_observation_normalization_stats(env: VecEnv, path: Path) -> Path | None:
    observation_rms = observation_normalization_stats(env)
    if observation_rms is None:
        return None

    np.savez(
        path,
        mean=observation_rms.mean,
        var=observation_rms.var,
        count=np.asarray(observation_rms.count, dtype=np.float64),
    )
    return path


@torch.no_grad()
def evaluate_q_policy(
    *,
    q_net: torch.nn.Module,
    env: DiscreteVecEnv,
    num_episodes: int,
) -> list[float]:
    if env.num_envs != 1:
        raise ValueError("evaluation env must use num_envs=1.")
    was_training = q_net.training
    q_net.eval()
    device = next(q_net.parameters()).device
    episode_returns: list[float] = []

    for _episode_index in range(num_episodes):
        observation = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            observation_tensor = torch.as_tensor(observation, device=device)
            q_values = q_net(observation_tensor)
            if q_values.shape != (1, env.num_actions):
                msg = (
                    "Q-network action dimension must match the env: "
                    f"expected (1, {env.num_actions}), got {q_values.shape}"
                )
                raise ValueError(msg)
            action_index = int(q_values.argmax(dim=1).item())
            step = env.step(np.asarray([action_index], dtype=np.int32))
            observation = step.observation
            episode_return += float(step.reward[0])
            done = bool(step.terminated[0] or step.truncated[0])

        episode_returns.append(episode_return)

    if was_training:
        q_net.train()
    return episode_returns


@torch.no_grad()
def evaluate_actor_critic_policy(
    *,
    model: torch.nn.Module,
    env: VecEnv,
    num_episodes: int,
) -> list[float]:
    if env.num_envs != 1:
        raise ValueError("evaluation env must use num_envs=1.")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    episode_returns: list[float] = []

    for _episode_index in range(num_episodes):
        observation = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            observation_tensor = torch.as_tensor(observation, device=device)
            dist, _value = model(observation_tensor)
            action = dist.deterministic()
            step = env.step(action.cpu().numpy())
            observation = step.observation
            episode_return += float(step.reward[0])
            done = bool(step.terminated[0] or step.truncated[0])

        episode_returns.append(episode_return)

    if was_training:
        model.train()
    return episode_returns
