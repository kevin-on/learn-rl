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

from config import CartPoleDQNConfig, load_config, save_config
from dqn import DQN, DQNLog
from metrics import JSONLMetricsLogger
from plot_metrics import plot_metrics
from schedules import ExplorationRateSchedule
from task_adapter import CartPoleTaskAdapter


def build_q_net(num_actions: int, hidden_sizes: list[int]) -> nn.Module:
    layers: list[nn.Module] = []
    input_size = 4
    for hidden_size in hidden_sizes:
        layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
        input_size = hidden_size
    layers.append(nn.Linear(input_size, num_actions))
    return nn.Sequential(*layers)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_policy(
    q_net: nn.Module,
    task_adapter: CartPoleTaskAdapter,
    num_episodes: int,
    seed: int,
) -> list[float]:
    was_training = q_net.training
    q_net.eval()
    device = next(q_net.parameters()).device
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
            q_values = q_net(state_tensor)
            action_index = int(q_values.argmax(dim=1).item())
            env_action = task_adapter.action_index_to_env_action(action_index)

            observation, reward, terminated, truncated, _info = task_adapter.env.step(
                env_action
            )
            state = task_adapter.encode_observation(observation)
            episode_return += float(reward)
            done = terminated or truncated

        episode_returns.append(episode_return)

    if was_training:
        q_net.train()

    return episode_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on CartPole.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cartpole_dqn.yaml"),
        help="Path to a YAML experiment config.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Output directory for resolved config, metrics JSONL, and plots.",
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


def resolve_config(args: argparse.Namespace) -> CartPoleDQNConfig:
    config = load_config(args.config, overrides=args.overrides)
    if args.no_plot:
        config = replace(config, logging=replace(config.logging, save_plot=False))
    return config


def create_run_dir(config: CartPoleDQNConfig, requested_run_dir: Path | None) -> Path:
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
    device = choose_device()
    run_dir = create_run_dir(config, args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    save_config(config, run_dir / "config.yaml")

    train_env = gym.wrappers.RecordEpisodeStatistics(gym.make(config.env.id))
    train_env.action_space.seed(config.seed)
    train_adapter = CartPoleTaskAdapter(train_env)

    eval_env = gym.make(config.env.id)
    eval_env.action_space.seed(config.seed + config.train.steps)
    eval_adapter = CartPoleTaskAdapter(eval_env)

    exploration_schedule = ExplorationRateSchedule(
        schedule=config.exploration.schedule,
        start=config.exploration.start,
        end=config.exploration.end,
        decay_steps=config.exploration.decay_steps,
    )
    q_net = build_q_net(train_adapter.num_actions, config.model.hidden_sizes).to(device)
    agent = DQN(
        train_adapter,
        q_net,
        learning_rate=config.train.learning_rate,
        discount_factor=config.train.discount_factor,
        soft_update_rate=config.train.soft_update_rate,
        buffer_capacity=config.train.buffer_capacity,
    )

    recent_returns: deque[float] = deque(maxlen=20)

    def log_training(agent: DQN, log: DQNLog) -> None:
        step = log.step_index + 1
        record = {
            "step": step,
            "loss": log.loss,
            "epsilon": log.exploration_rate,
        }

        if log.loss is not None and step % config.logging.loss_every_steps == 0:
            print(
                f"step={step:6d} loss={log.loss:.4f} epsilon={log.exploration_rate:.3f}"
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
                f"episode_length={episode_length:3d} "
                f"epsilon={log.exploration_rate:.3f}"
            )

        metrics.write(**record)

    def run_evaluation(agent: DQN, step_index: int, exploration_rate: float) -> None:
        step = step_index + 1
        if step % config.eval.every_steps != 0:
            return

        returns = evaluate_policy(
            agent.online_q_net,
            eval_adapter,
            num_episodes=config.eval.episodes,
            seed=config.seed + step,
        )
        eval_mean_return = float(np.mean(returns))
        eval_std_return = float(np.std(returns))
        eval_best_return = float(np.max(returns))
        metrics.write(
            step=step,
            epsilon=exploration_rate,
            eval_mean_return=eval_mean_return,
            eval_std_return=eval_std_return,
            eval_best_return=eval_best_return,
        )
        print(
            f"step={step:6d} "
            f"eval_mean_return={eval_mean_return:6.1f} "
            f"eval_std_return={eval_std_return:6.1f} "
            f"eval_best_return={eval_best_return:6.1f} "
            f"epsilon={exploration_rate:.3f}"
        )

    print(
        f"Training {config.env.id} for {config.train.steps} steps on {device}. "
        f"Run directory: {run_dir}"
    )
    with JSONLMetricsLogger(metrics_path) as metrics:
        try:
            agent.train(
                num_steps=config.train.steps,
                batch_size=config.train.batch_size,
                exploration_rate_fn=exploration_schedule.value,
                env_seed=config.seed,
                log_fn=log_training,
                eval_fn=run_evaluation,
            )
        finally:
            train_env.close()
            eval_env.close()

    if config.logging.save_plot:
        plot_path = run_dir / "metrics.png"
        plot_metrics(metrics_path, plot_path, title=f"{config.env.id} DQN")
        print(f"Saved metrics plot to {plot_path}")


if __name__ == "__main__":
    main()
