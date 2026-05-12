import argparse
import random
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import PPOConfig, load_ppo_config, save_config
from metrics import JSONLMetricsLogger
from plot_metrics import plot_metrics
from ppo import PPO, PPOLog
from ppo_env import EnvPoolPPOVecEnv, PPOVecEnv
from ppo_models import build_ppo_model


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def envpool_kwargs(
    config: PPOConfig, *, evaluation: bool = False
) -> dict[str, int | bool]:
    kwargs: dict[str, int | bool] = {}
    if config.env.kind == "atari":
        kwargs.update(
            stack_num=config.env.stack_num,
            frame_skip=config.env.frame_skip,
            noop_max=config.env.noop_max,
            episodic_life=config.env.episodic_life,
            reward_clip=config.env.reward_clip,
            img_height=config.env.img_height,
            img_width=config.env.img_width,
            gray_scale=config.env.gray_scale,
        )
        if evaluation:
            kwargs["episodic_life"] = False
            kwargs["reward_clip"] = False

    return kwargs


def build_model(
    *,
    config: PPOConfig,
    observation_shape: tuple[int, ...],
    num_actions: int,
) -> torch.nn.Module:
    return build_ppo_model(
        name=config.model.name,
        observation_shape=observation_shape,
        num_actions=num_actions,
        kwargs=config.model.kwargs,
    )


@torch.no_grad()
def evaluate_policy(
    *,
    model: torch.nn.Module,
    env: PPOVecEnv,
    num_episodes: int,
) -> list[float]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    episode_returns: list[float] = []
    assert env.num_envs == 1

    for _episode_index in range(num_episodes):
        observation = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            observation_tensor = torch.as_tensor(observation, device=device)
            logits, _value = model(observation_tensor)
            assert logits.shape == (1, env.num_actions)

            action_index = int(logits.argmax(dim=1).item())
            step = env.step(np.asarray([action_index], dtype=np.int32))
            observation = step.observation
            episode_return += float(step.reward[0])
            done = bool(step.terminated[0] or step.truncated[0])

        episode_returns.append(episode_return)

    if was_training:
        model.train()

    return episode_returns


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
        config = replace(config, logging=replace(config.logging, save_plot=False))
    return config


def create_run_dir(config: PPOConfig, requested_run_dir: Path | None) -> Path:
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

    train_env = EnvPoolPPOVecEnv(
        env_id=config.env.id,
        num_envs=config.env.num_envs,
        seed=config.seed,
        env_kwargs=envpool_kwargs(config),
    )
    model = build_model(
        config=config,
        observation_shape=train_env.observation_shape,
        num_actions=train_env.num_actions,
    ).to(device)

    eval_env = EnvPoolPPOVecEnv(
        env_id=config.env.id,
        num_envs=1,
        seed=config.eval.seed,
        env_kwargs=envpool_kwargs(config, evaluation=True),
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

            returns = evaluate_policy(
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

    if config.logging.save_plot:
        plot_path = run_dir / "metrics.png"
        plot_metrics(metrics_path, plot_path, title=f"{config.env.id} PPO")
        print(f"Saved metrics plot to {plot_path}")


if __name__ == "__main__":
    main()
