import argparse
import random
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from a3c import A3C, A3CLog, ActorCriticNet
from config import A3CConfig, load_a3c_config, save_config
from metrics import JSONLMetricsLogger
from plot_metrics import plot_metrics
from task_adapter import VectorTaskAdapter, make_task_adapter


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inspect_env(env_id: str) -> tuple[int, int]:
    env = gym.make(env_id)
    try:
        task_adapter = make_task_adapter(env, env_id)
        return task_adapter.state_size, task_adapter.num_actions
    finally:
        env.close()


@torch.no_grad()
def evaluate_policy(
    model: ActorCriticNet,
    task_adapter: VectorTaskAdapter,
    num_episodes: int,
    seed: int,
) -> list[float]:
    was_training = model.training
    model.eval()
    episode_returns: list[float] = []

    for episode_index in range(num_episodes):
        observation, _info = task_adapter.env.reset(seed=seed + episode_index)
        state = task_adapter.encode_observation(observation)
        done = False
        episode_return = 0.0

        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, _value = model(state_tensor)
            if logits.shape != (1, task_adapter.num_actions):
                msg = (
                    "Policy head action dimension must match the task adapter: "
                    f"expected (1, {task_adapter.num_actions}), got {logits.shape}"
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
        model.train()

    return episode_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CPU A3C on a supported Gymnasium discrete-action task."
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


def resolve_config(args: argparse.Namespace) -> A3CConfig:
    config = load_a3c_config(args.config, overrides=args.overrides)
    if args.no_plot:
        config = replace(config, logging=replace(config.logging, save_plot=False))
    return config


def create_run_dir(config: A3CConfig, requested_run_dir: Path | None) -> Path:
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
    run_dir = create_run_dir(config, args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    save_config(config, run_dir / "config.yaml")

    state_size, num_actions = inspect_env(config.env.id)
    agent = A3C(
        env_id=config.env.id,
        state_size=state_size,
        num_actions=num_actions,
        hidden_sizes=config.model.hidden_sizes,
        num_workers=config.train.num_workers,
        learning_rate=config.train.learning_rate,
        discount_factor=config.train.discount_factor,
        rollout_steps=config.train.rollout_steps,
        max_grad_norm=config.train.max_grad_norm,
        entropy_coef=config.train.entropy_coef,
        rmsprop_alpha=config.train.rmsprop_alpha,
        rmsprop_eps=config.train.rmsprop_eps,
    )

    eval_env = gym.make(config.env.id)
    eval_env.action_space.seed(config.eval.seed)
    eval_adapter = make_task_adapter(eval_env, config.env.id)
    eval_model = ActorCriticNet(
        state_size=state_size,
        num_actions=num_actions,
        hidden_sizes=config.model.hidden_sizes,
    )
    recent_returns: deque[float] = deque(maxlen=20)
    next_loss_step = config.logging.loss_every_steps
    next_eval_step = config.eval.every_steps

    def log_training(log: A3CLog) -> None:
        nonlocal next_loss_step, next_eval_step

        record = {
            "step": log.global_step,
            "worker_id": log.worker_id,
            "loss": log.loss,
            "policy_loss": log.policy_loss,
            "value_loss": log.value_loss,
            "entropy": log.entropy,
            "grad_norm": log.grad_norm,
            "rollout_length": log.rollout_length,
        }

        if log.episode_return is not None and log.episode_length is not None:
            recent_returns.append(log.episode_return)
            mean_return = float(np.mean(recent_returns))
            record.update(
                train_episode_return=log.episode_return,
                train_episode_return_mean20=mean_return,
                train_episode_length=log.episode_length,
            )
            print(
                f"step={log.global_step:6d} "
                f"worker={log.worker_id:2d} "
                f"train_return={log.episode_return:6.1f} "
                f"mean20_return={mean_return:6.1f} "
                f"episode_length={log.episode_length:3d}"
            )

        if log.global_step >= next_loss_step:
            while next_loss_step <= log.global_step:
                next_loss_step += config.logging.loss_every_steps
            grad_norm_text = (
                "" if log.grad_norm is None else f" grad_norm={log.grad_norm:.4f}"
            )
            print(
                f"step={log.global_step:6d} "
                f"loss={log.loss:.4f} "
                f"policy_loss={log.policy_loss:.4f} "
                f"value_loss={log.value_loss:.4f} "
                f"entropy={log.entropy:.4f}"
                f"{grad_norm_text}"
            )

        metrics.write(**record)

        if log.global_step >= next_eval_step:
            while next_eval_step <= log.global_step:
                next_eval_step += config.eval.every_steps

            eval_model.load_state_dict(agent.snapshot_state_dict())
            returns = evaluate_policy(
                eval_model,
                eval_adapter,
                num_episodes=config.eval.episodes,
                seed=config.eval.seed,
            )
            eval_mean_return = float(np.mean(returns))
            eval_std_return = float(np.std(returns))
            eval_best_return = float(np.max(returns))
            metrics.write(
                step=log.global_step,
                eval_seed=config.eval.seed,
                eval_mean_return=eval_mean_return,
                eval_std_return=eval_std_return,
                eval_best_return=eval_best_return,
            )
            print(
                f"step={log.global_step:6d} "
                f"eval_mean_return={eval_mean_return:6.1f} "
                f"eval_std_return={eval_std_return:6.1f} "
                f"eval_best_return={eval_best_return:6.1f}"
            )

    print(
        f"Training {config.env.id} for {config.train.steps} A3C steps on CPU "
        f"with {config.train.num_workers} workers. Run directory: {run_dir}"
    )
    with JSONLMetricsLogger(metrics_path) as metrics:
        try:
            agent.train(
                num_steps=config.train.steps,
                seed=config.seed,
                log_fn=log_training,
            )
        finally:
            eval_env.close()

    if config.logging.save_plot:
        plot_path = run_dir / "metrics.png"
        plot_metrics(metrics_path, plot_path, title=f"{config.env.id} A3C")
        print(f"Saved metrics plot to {plot_path}")


if __name__ == "__main__":
    main()
