import argparse
from collections import deque
from pathlib import Path

import numpy as np

from checkpoints import (
    apply_resume_overrides,
    build_checkpoint_payload,
    checkpoint_paths,
    config_from_checkpoint,
    load_checkpoint,
    resume_checkpoint_path,
    save_checkpoint,
)
from config import DQNConfig, load_config, save_config
from dqn import DQN, DQNLog
from experiment import (
    choose_device,
    create_run_dir,
    evaluate_q_policy,
    make_envpool_env,
    set_random_seeds,
)
from metrics import JSONLMetricsLogger
from models import build_q_model
from plot_metrics import plot_metrics
from schedules import ExplorationRateSchedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DQN on an EnvPool discrete-action task."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a YAML experiment config.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Run directory to resume from. Loads checkpoints/last.pt.",
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
    args = parser.parse_args()
    if args.resume is None and args.config is None:
        parser.error("--config is required unless --resume is provided.")
    if args.resume is not None and args.config is not None:
        parser.error("--config cannot be used with --resume.")
    if args.resume is not None and args.run_dir is not None:
        parser.error("--run-dir cannot be used with --resume.")
    return args


def resolve_config(args: argparse.Namespace) -> DQNConfig:
    config = load_config(args.config, overrides=args.overrides)
    if args.no_plot:
        config = config.model_copy(
            update={"logging": config.logging.model_copy(update={"save_plot": False})}
        )
    return config


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = None
    if args.resume is not None:
        run_dir = args.resume
        checkpoint_path = resume_checkpoint_path(run_dir)
        checkpoint = load_checkpoint(
            checkpoint_path,
            map_location=device,
            expected_algorithm="dqn",
        )
        config = config_from_checkpoint(checkpoint)
        if not isinstance(config, DQNConfig):
            raise TypeError("Expected DQN config in checkpoint.")
        config = apply_resume_overrides(
            config,
            overrides=args.overrides,
            checkpoint_step=int(checkpoint["step"]),
        )
        if args.no_plot:
            config = config.model_copy(
                update={
                    "logging": config.logging.model_copy(update={"save_plot": False})
                }
            )
        append_metrics = True
    else:
        config = resolve_config(args)
        run_dir = create_run_dir(config, args.run_dir)
        save_config(config, run_dir / "config.yaml")
        append_metrics = False

    set_random_seeds(config.seed)
    metrics_path = run_dir / "metrics.jsonl"
    last_checkpoint_path, best_checkpoint_path = checkpoint_paths(run_dir)

    train_env = make_envpool_env(
        config,
        num_envs=config.env.num_envs,
        seed=config.seed,
    )
    eval_env = make_envpool_env(
        config,
        num_envs=1,
        seed=config.eval.seed,
        evaluation=True,
    )

    exploration_schedule = ExplorationRateSchedule(
        schedule=config.exploration.schedule,
        start=config.exploration.start,
        end=config.exploration.end,
        decay_steps=config.exploration.decay_steps,
    )
    q_net = build_q_model(
        name=config.model.name,
        observation_shape=train_env.observation_shape,
        num_actions=train_env.num_actions,
        kwargs=config.model.kwargs,
    ).to(device)
    agent = DQN(
        train_env,
        q_net,
        learning_rate=config.train.learning_rate,
        discount_factor=config.train.discount_factor,
        soft_update_rate=config.train.soft_update_rate,
        buffer_capacity=config.train.buffer_capacity,
        batch_size=config.train.batch_size,
        learning_starts=config.train.learning_starts,
        max_grad_norm=config.train.max_grad_norm,
    )
    if checkpoint is not None:
        agent.load_checkpoint_state(checkpoint)

    if config.train.steps <= agent.step:
        msg = (
            "train.steps must be larger than the checkpoint step when resuming; "
            f"got train.steps={config.train.steps}, checkpoint step={agent.step}."
        )
        raise ValueError(msg)

    recent_returns: deque[float] = deque(maxlen=20)
    next_loss_step = config.logging.loss_every_steps
    next_eval_step = config.eval.every_steps
    while next_loss_step <= agent.step:
        next_loss_step += config.logging.loss_every_steps
    while next_eval_step <= agent.step:
        next_eval_step += config.eval.every_steps
    if checkpoint is not None and best_checkpoint_path.exists():
        best_eval_mean_return = checkpoint.get("best_eval_mean_return")
        best_step = checkpoint.get("best_step")
    else:
        best_eval_mean_return = None
        best_step = None

    def checkpoint_payload() -> dict:
        return build_checkpoint_payload(
            algorithm="dqn",
            config=config,
            agent_state=agent.checkpoint_state(),
            observation_normalization=None,
            best_eval_mean_return=best_eval_mean_return,
            best_step=best_step,
        )

    def save_last_checkpoint() -> None:
        save_checkpoint(checkpoint_payload(), last_checkpoint_path)

    def log_training(agent: DQN, log: DQNLog) -> None:
        nonlocal best_eval_mean_return, best_step, next_loss_step, next_eval_step

        metrics.write(
            step=log.step,
            loss=log.loss,
            grad_norm=log.grad_norm,
            epsilon=log.exploration_rate,
        )

        if log.loss is not None and log.step >= next_loss_step:
            while next_loss_step <= log.step:
                next_loss_step += config.logging.loss_every_steps
            grad_norm_text = (
                "" if log.grad_norm is None else f" grad_norm={log.grad_norm:.4f}"
            )
            print(
                f"step={log.step:6d} loss={log.loss:.4f}{grad_norm_text} "
                f"epsilon={log.exploration_rate:.3f}"
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
                f"episode_length={episode.episode_length:4d} "
                f"epsilon={log.exploration_rate:.3f}"
            )

        if log.step >= next_eval_step:
            while next_eval_step <= log.step:
                next_eval_step += config.eval.every_steps

            returns = evaluate_q_policy(
                q_net=agent.online_q_net,
                env=eval_env,
                num_episodes=config.eval.episodes,
            )
            eval_mean_return = float(np.mean(returns))
            eval_std_return = float(np.std(returns))
            eval_best_return = float(np.max(returns))
            metrics.write(
                step=log.step,
                epsilon=log.exploration_rate,
                eval_seed=config.eval.seed,
                eval_mean_return=eval_mean_return,
                eval_std_return=eval_std_return,
                eval_best_return=eval_best_return,
            )
            print(
                f"step={log.step:6d} "
                f"eval_mean_return={eval_mean_return:6.1f} "
                f"eval_std_return={eval_std_return:6.1f} "
                f"eval_best_return={eval_best_return:6.1f} "
                f"epsilon={log.exploration_rate:.3f}"
            )
            if (
                best_eval_mean_return is None
                or eval_mean_return > best_eval_mean_return
            ):
                best_eval_mean_return = eval_mean_return
                best_step = log.step
                save_checkpoint(checkpoint_payload(), best_checkpoint_path)
            save_last_checkpoint()

    print(
        f"Training {config.env.id} for at least {config.train.steps} DQN env "
        f"steps on {device} with {config.env.num_envs} EnvPool envs. "
        f"Run directory: {run_dir}"
    )
    with JSONLMetricsLogger(metrics_path, append=append_metrics) as metrics:
        try:
            agent.train(
                num_steps=config.train.steps,
                exploration_rate_fn=exploration_schedule.value,
                log_fn=log_training,
            )
        finally:
            save_last_checkpoint()
            train_env.close()
            eval_env.close()

    if config.logging.save_plot:
        plot_path = run_dir / "metrics.png"
        plot_metrics(metrics_path, plot_path, title=f"{config.env.id} DQN")
        print(f"Saved metrics plot to {plot_path}")


if __name__ == "__main__":
    main()
