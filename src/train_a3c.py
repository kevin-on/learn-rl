import argparse
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from a3c import A3C, A3CLog
from config import A3CConfig, load_a3c_config, save_config
from envs import DiscreteActionSpec, EnvPoolVecEnv
from experiment import (
    create_run_dir,
    evaluate_actor_critic_policy,
    set_random_seeds,
)
from metrics import JSONLMetricsLogger
from models import build_actor_critic_model
from plot_metrics import plot_metrics


def inspect_env(env_id: str) -> tuple[int, int]:
    env = EnvPoolVecEnv(env_id=env_id, num_envs=1, seed=0)
    try:
        return int(np.prod(env.observation_shape)), env.num_actions
    finally:
        env.close()


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
        model_name=config.model.name,
        model_kwargs=config.model.kwargs,
        num_workers=config.train.num_workers,
        learning_rate=config.train.learning_rate,
        value_loss_coef=config.train.value_loss_coef,
        discount_factor=config.train.discount_factor,
        rollout_steps=config.train.rollout_steps,
        max_grad_norm=config.train.max_grad_norm,
        entropy_coef=config.train.entropy_coef,
        rmsprop_alpha=config.train.rmsprop_alpha,
        rmsprop_eps=config.train.rmsprop_eps,
    )

    eval_env = EnvPoolVecEnv(env_id=config.env.id, num_envs=1, seed=config.eval.seed)
    eval_model = build_actor_critic_model(
        name=config.model.name,
        observation_shape=eval_env.observation_shape,
        action_spec=DiscreteActionSpec(num_actions=num_actions),
        kwargs=config.model.kwargs,
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
            returns = evaluate_actor_critic_policy(
                model=eval_model,
                env=eval_env,
                num_episodes=config.eval.episodes,
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
