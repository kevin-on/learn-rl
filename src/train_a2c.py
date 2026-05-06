import argparse
import random
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from a2c import A2C, A2CLog
from config import A2CConfig, load_a2c_config, save_config
from metrics import JSONLMetricsLogger
from plot_metrics import plot_metrics
from task_adapter import VectorTaskAdapter, make_task_adapter


def build_mlp(input_size: int, output_size: int, hidden_sizes: list[int]) -> nn.Module:
    layers: list[nn.Module] = []
    layer_input_size = input_size
    for hidden_size in hidden_sizes:
        layers.extend([nn.Linear(layer_input_size, hidden_size), nn.ReLU()])
        layer_input_size = hidden_size
    layers.append(nn.Linear(layer_input_size, output_size))
    return nn.Sequential(*layers)


def build_policy_net(
    state_size: int, num_actions: int, hidden_sizes: list[int]
) -> nn.Module:
    return build_mlp(state_size, num_actions, hidden_sizes)


def build_value_net(state_size: int, hidden_sizes: list[int]) -> nn.Module:
    return build_mlp(state_size, 1, hidden_sizes)


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    if requested_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("--device mps was requested, but MPS is not available.")

    msg = f"device must be one of: auto, cpu, cuda, mps; got {requested_device}"
    raise ValueError(msg)


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_policy(
    policy_net: nn.Module,
    task_adapter: VectorTaskAdapter,
    num_episodes: int,
    seed: int,
) -> list[float]:
    was_training = policy_net.training
    policy_net.eval()
    device = next(policy_net.parameters()).device
    episode_returns: list[float] = []

    for episode_index in range(num_episodes):
        observation, _info = task_adapter.env.reset(seed=seed + episode_index)
        state = task_adapter.encode_observation(observation)
        done = False
        episode_return = 0.0

        while not done:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            logits = policy_net(state_tensor)
            if logits.ndim != 2 or logits.shape[1] != task_adapter.num_actions:
                msg = (
                    "Policy network action dimension must match the task adapter: "
                    f"expected {task_adapter.num_actions}, got shape {logits.shape}"
                )
                raise ValueError(msg)
            action_index = int(logits.argmax(dim=1).item())
            env_action = task_adapter.action_index_to_env_action(action_index)

            observation, reward, terminated, truncated, _info = task_adapter.env.step(
                env_action
            )
            state = task_adapter.encode_observation(observation)
            episode_return += float(reward)
            done = terminated or truncated

        episode_returns.append(episode_return)

    if was_training:
        policy_net.train()

    return episode_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train A2C on a supported Gymnasium discrete-action task."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML experiment config.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Output directory for resolved config, metrics JSONL, and plots.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="PyTorch device to use. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set train.steps=1000.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the metrics plot after training.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> A2CConfig:
    config = load_a2c_config(args.config, overrides=args.overrides)
    if args.no_plot:
        config = replace(config, logging=replace(config.logging, save_plot=False))
    return config


def create_run_dir(config: A2CConfig, requested_run_dir: Path | None) -> Path:
    if requested_run_dir is not None:
        run_dir = requested_run_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_name = f"{timestamp}-{config.experiment.name}-seed{config.seed}"
        run_dir = Path(config.experiment.run_root) / run_name

    run_dir.mkdir(parents=True, exist_ok=requested_run_dir is not None)
    return run_dir


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    set_random_seeds(config.seed)
    device = choose_device(args.device)
    run_dir = create_run_dir(config, args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    save_config(config, run_dir / "config.yaml")

    train_env = gym.wrappers.RecordEpisodeStatistics(gym.make(config.env.id))
    train_env.action_space.seed(config.seed)
    train_adapter = make_task_adapter(train_env, config.env.id)

    eval_env = gym.make(config.env.id)
    eval_env.action_space.seed(config.eval.seed)
    eval_adapter = make_task_adapter(eval_env, config.env.id)

    policy_net = build_policy_net(
        train_adapter.state_size,
        train_adapter.num_actions,
        config.model.hidden_sizes,
    ).to(device)
    value_net = build_value_net(
        train_adapter.state_size,
        config.model.hidden_sizes,
    ).to(device)
    agent = A2C(
        train_adapter,
        policy_net,
        value_net,
        policy_learning_rate=config.train.policy_learning_rate,
        value_learning_rate=config.train.value_learning_rate,
        discount_factor=config.train.discount_factor,
        rollout_steps=config.train.rollout_steps,
        max_grad_norm=config.train.max_grad_norm,
    )

    recent_returns: deque[float] = deque(maxlen=20)

    def log_training(agent: A2C, log: A2CLog) -> None:
        step = log.step_index + 1
        record = {
            "step": step,
            "loss": log.loss,
            "policy_loss": log.policy_loss,
            "value_loss": log.value_loss,
            "grad_norm": log.grad_norm,
        }

        if log.loss is not None and step % config.logging.loss_every_steps == 0:
            grad_norm_text = (
                "" if log.grad_norm is None else f" grad_norm={log.grad_norm:.4f}"
            )
            policy_loss_text = (
                "" if log.policy_loss is None else f" policy_loss={log.policy_loss:.4f}"
            )
            value_loss_text = (
                "" if log.value_loss is None else f" value_loss={log.value_loss:.4f}"
            )
            print(
                f"step={step:6d} loss={log.loss:.4f}"
                f"{policy_loss_text}{value_loss_text}{grad_norm_text}"
            )

        if "episode" in log.info:
            episode = log.info["episode"]
            episode_return = float(episode["r"])
            episode_length = int(episode["l"])
            recent_returns.append(episode_return)
            mean_return = float(np.mean(recent_returns))
            record.update(
                train_episode_return=episode_return,
                train_episode_return_mean20=mean_return,
                train_episode_length=episode_length,
            )
            print(
                f"step={step:6d} "
                f"train_return={episode_return:6.1f} "
                f"mean20_return={mean_return:6.1f} "
                f"episode_length={episode_length:3d}"
            )

        metrics.write(**record)

    def run_evaluation(agent: A2C, step_index: int) -> None:
        step = step_index + 1
        if step % config.eval.every_steps != 0:
            return

        returns = evaluate_policy(
            agent.policy_net,
            eval_adapter,
            num_episodes=config.eval.episodes,
            seed=config.eval.seed,
        )
        eval_mean_return = float(np.mean(returns))
        eval_std_return = float(np.std(returns))
        eval_best_return = float(np.max(returns))
        metrics.write(
            step=step,
            eval_seed=config.eval.seed,
            eval_mean_return=eval_mean_return,
            eval_std_return=eval_std_return,
            eval_best_return=eval_best_return,
        )
        print(
            f"step={step:6d} "
            f"eval_mean_return={eval_mean_return:6.1f} "
            f"eval_std_return={eval_std_return:6.1f} "
            f"eval_best_return={eval_best_return:6.1f}"
        )

    print(
        f"Training {config.env.id} for {config.train.steps} A2C steps on {device}. "
        f"Run directory: {run_dir}"
    )
    with JSONLMetricsLogger(metrics_path) as metrics:
        try:
            agent.train(
                num_steps=config.train.steps,
                env_seed=config.seed,
                log_fn=log_training,
                eval_fn=run_evaluation,
            )
        finally:
            train_env.close()
            eval_env.close()

    if config.logging.save_plot:
        plot_path = run_dir / "metrics.png"
        plot_metrics(metrics_path, plot_path, title=f"{config.env.id} A2C")
        print(f"Saved metrics plot to {plot_path}")


if __name__ == "__main__":
    main()
