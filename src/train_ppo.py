import argparse
from collections import deque
from pathlib import Path

import numpy as np

from config import PPOConfig, load_ppo_config, save_config
from experiment import (
    choose_device,
    create_run_dir,
    evaluate_actor_critic_policy,
    make_envpool_env,
    observation_normalization_stats,
    save_observation_normalization_stats,
    set_random_seeds,
)
from metrics import JSONLMetricsLogger
from models import build_actor_critic_model
from plot_metrics import plot_metrics
from ppo import PPO, PPOLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train clipped PPO with GAE on EnvPool Gymnasium environments."
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
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch device to use. 'auto' prefers CUDA, then CPU.",
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


def resolve_config(args: argparse.Namespace) -> PPOConfig:
    config = load_ppo_config(args.config, overrides=args.overrides)
    if args.no_plot:
        config = config.model_copy(
            update={"logging": config.logging.model_copy(update={"save_plot": False})}
        )
    return config


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    set_random_seeds(config.seed)
    device = choose_device(args.device)
    run_dir = create_run_dir(config, args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    save_config(config, run_dir / "config.yaml")

    train_env = make_envpool_env(
        config,
        num_envs=config.env.num_envs,
        seed=config.seed,
    )
    model = build_actor_critic_model(
        name=config.model.name,
        observation_shape=train_env.observation_shape,
        action_spec=train_env.action_spec,
        kwargs=config.model.kwargs,
    ).to(device)

    eval_env = make_envpool_env(
        config,
        num_envs=1,
        seed=config.eval.seed,
        evaluation=True,
        observation_rms=observation_normalization_stats(train_env),
    )

    agent = PPO(
        train_env,
        model,
        learning_rate=config.train.learning_rate,
        rollout_steps=config.train.rollout_steps,
        minibatch_size=config.train.minibatch_size,
        epochs=config.train.epochs,
        discount_factor=config.train.discount_factor,
        gae_lambda=config.train.gae_lambda,
        clip_coef=config.train.clip_coef,
        value_coef=config.train.value_coef,
        entropy_coef=config.train.entropy_coef,
        max_grad_norm=config.train.max_grad_norm,
    )

    recent_returns: deque[float] = deque(maxlen=20)
    next_loss_step = config.logging.loss_every_steps
    next_eval_step = config.eval.every_steps

    def log_training(_agent: PPO, log: PPOLog) -> None:
        nonlocal next_loss_step, next_eval_step

        record = {
            "step": log.step,
            "update": log.update,
            "loss": log.stats.loss,
            "policy_loss": log.stats.policy_loss,
            "value_loss": log.stats.value_loss,
            "entropy": log.stats.entropy,
            "clip_fraction": log.stats.clip_fraction,
            "grad_norm": log.stats.grad_norm,
            "rollout_steps": log.rollout_steps,
        }
        metrics.write(**record)

        if log.step >= next_loss_step:
            while next_loss_step <= log.step:
                next_loss_step += config.logging.loss_every_steps
            grad_norm_text = (
                ""
                if log.stats.grad_norm is None
                else f" grad_norm={log.stats.grad_norm:.4f}"
            )
            print(
                f"step={log.step:6d} "
                f"loss={log.stats.loss:.4f} "
                f"policy_loss={log.stats.policy_loss:.4f} "
                f"value_loss={log.stats.value_loss:.4f} "
                f"entropy={log.stats.entropy:.4f} "
                f"clip_fraction={log.stats.clip_fraction:.3f}"
                f"{grad_norm_text}"
            )

        for episode in log.episodes:
            recent_returns.append(episode.episode_return)
            mean_return = float(np.mean(recent_returns))
            metrics.write(
                step=log.step,
                env_id=episode.env_id,
                train_episode_return=episode.episode_return,
                train_episode_return_mean20=mean_return,
                train_episode_length=episode.episode_length,
            )
            print(
                f"step={log.step:6d} "
                f"env={episode.env_id:3d} "
                f"train_return={episode.episode_return:6.1f} "
                f"mean20_return={mean_return:6.1f} "
                f"episode_length={episode.episode_length:4d}"
            )

        if log.step >= next_eval_step:
            while next_eval_step <= log.step:
                next_eval_step += config.eval.every_steps

            returns = evaluate_actor_critic_policy(
                model=model,
                env=eval_env,
                num_episodes=config.eval.episodes,
            )
            eval_mean_return = float(np.mean(returns))
            eval_std_return = float(np.std(returns))
            eval_best_return = float(np.max(returns))
            metrics.write(
                step=log.step,
                eval_seed=config.eval.seed,
                eval_mean_return=eval_mean_return,
                eval_std_return=eval_std_return,
                eval_best_return=eval_best_return,
            )
            print(
                f"step={log.step:6d} "
                f"eval_mean_return={eval_mean_return:6.1f} "
                f"eval_std_return={eval_std_return:6.1f} "
                f"eval_best_return={eval_best_return:6.1f}"
            )

    print(
        f"Training {config.env.id} for at least {config.train.steps} environment "
        f"steps on {device} with {config.env.num_envs} EnvPool envs. "
        f"Run directory: {run_dir}"
    )
    with JSONLMetricsLogger(metrics_path) as metrics:
        try:
            agent.train(num_steps=config.train.steps, log_fn=log_training)
        finally:
            train_env.close()
            eval_env.close()

    obs_norm_path = save_observation_normalization_stats(
        train_env, run_dir / "observation_normalization.npz"
    )
    if obs_norm_path is not None:
        print(f"Saved observation normalization stats to {obs_norm_path}")

    if config.logging.save_plot:
        plot_path = run_dir / "metrics.png"
        plot_metrics(metrics_path, plot_path, title=f"{config.env.id} PPO")
        print(f"Saved metrics plot to {plot_path}")


if __name__ == "__main__":
    main()
