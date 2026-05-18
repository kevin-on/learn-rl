import argparse
from pathlib import Path

import numpy as np

from checkpoints import config_from_checkpoint, load_checkpoint
from config import A2CConfig, DDPGConfig, DQNConfig, PPOConfig, TD3Config
from experiment import (
    choose_device,
    evaluate_actor_critic_policy,
    evaluate_ddpg_policy,
    evaluate_q_policy,
    make_envpool_env,
    running_mean_std_from_state,
    set_random_seeds,
)
from models import (
    build_actor_critic_model,
    build_ddpg_actor_critic_model,
    build_q_model,
    build_td3_actor_critic_model,
)
from videos import EpisodeVideoRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DQN, DDPG, TD3, A2C, or PPO checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a .pt checkpoint file.")
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of evaluation episodes. Defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Evaluation seed. Defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch device to use. 'auto' prefers CUDA, then CPU.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Directory for MP4 videos. When set, evaluation envs render rgb_array frames.",
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        help="Number of evaluated episodes to save. Defaults to 1 when --video-dir is set.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frames per second for saved videos. Defaults to 30.",
    )
    parser.add_argument(
        "--video-frame-stride",
        type=int,
        default=1,
        help="Save every Nth rendered frame. Defaults to 1.",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=28,
        help="H.264 CRF quality value in [0, 51]; higher is smaller. Defaults to 28.",
    )
    parser.add_argument(
        "--video-preset",
        default="medium",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="H.264 encoder preset. Slower presets usually produce smaller files.",
    )
    parser.add_argument(
        "--video-encoder-workers",
        type=int,
        default=1,
        help="Number of background video encoder workers. Defaults to 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.video_episodes is not None and args.video_episodes <= 0:
        raise ValueError("--video-episodes must be positive.")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    if args.video_frame_stride <= 0:
        raise ValueError("--video-frame-stride must be positive.")
    if not 0 <= args.video_crf <= 51:
        raise ValueError("--video-crf must be in [0, 51].")
    if args.video_encoder_workers <= 0:
        raise ValueError("--video-encoder-workers must be positive.")
    if args.video_episodes is not None and args.video_dir is None:
        raise ValueError("--video-episodes requires --video-dir.")

    device = choose_device(args.device)
    checkpoint_path = args.checkpoint
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = config_from_checkpoint(checkpoint)
    episodes = args.episodes or config.eval.episodes
    seed = args.seed if args.seed is not None else config.eval.seed
    set_random_seeds(seed)
    video_recorder = (
        None
        if args.video_dir is None
        else EpisodeVideoRecorder(
            video_dir=args.video_dir,
            max_episodes=args.video_episodes or 1,
            fps=args.video_fps,
            frame_stride=args.video_frame_stride,
            crf=args.video_crf,
            preset=args.video_preset,
            encoder_workers=args.video_encoder_workers,
        )
    )

    try:
        evaluation_error: BaseException | None = None
        if isinstance(config, DQNConfig):
            returns = evaluate_dqn_checkpoint(
                checkpoint=checkpoint,
                config=config,
                episodes=episodes,
                seed=seed,
                device=device,
                video_recorder=video_recorder,
            )
        elif isinstance(config, DDPGConfig):
            returns = evaluate_ddpg_checkpoint(
                checkpoint=checkpoint,
                config=config,
                episodes=episodes,
                seed=seed,
                device=device,
                video_recorder=video_recorder,
            )
        elif isinstance(config, TD3Config):
            returns = evaluate_td3_checkpoint(
                checkpoint=checkpoint,
                config=config,
                episodes=episodes,
                seed=seed,
                device=device,
                video_recorder=video_recorder,
            )
        elif isinstance(config, A2CConfig | PPOConfig):
            returns = evaluate_actor_critic_checkpoint(
                checkpoint=checkpoint,
                config=config,
                episodes=episodes,
                seed=seed,
                device=device,
                video_recorder=video_recorder,
            )
        else:
            msg = f"Unsupported checkpoint config type: {type(config).__name__}"
            raise TypeError(msg)
    except BaseException as exc:
        evaluation_error = exc
        raise
    finally:
        _close_video_recorder(video_recorder, primary_error=evaluation_error)

    eval_mean_return = float(np.mean(returns))
    eval_std_return = float(np.std(returns))
    eval_best_return = float(np.max(returns))
    print(f"checkpoint={checkpoint_path}")
    print(f"algorithm={checkpoint['algorithm']} step={checkpoint['step']}")
    print(f"episodes={episodes} seed={seed}")
    print(
        f"eval_mean_return={eval_mean_return:.3f} "
        f"eval_std_return={eval_std_return:.3f} "
        f"eval_best_return={eval_best_return:.3f}"
    )
    if video_recorder is not None:
        for video in video_recorder.videos:
            metadata = " ".join(
                f"{key}={value}" for key, value in sorted(video.metadata.items())
            )
            metadata_text = f" {metadata}" if metadata else ""
            print(
                f"video={video.path} episode={video.recording_id}"
                f"{metadata_text} frames={video.num_frames}"
            )


def _close_video_recorder(
    video_recorder: EpisodeVideoRecorder | None,
    *,
    primary_error: BaseException | None = None,
) -> None:
    if video_recorder is None:
        return

    try:
        video_recorder.close()
    except BaseException as close_error:
        if primary_error is None:
            raise
        primary_error.add_note(f"video recorder close failed: {close_error!r}")


def evaluate_dqn_checkpoint(
    *,
    checkpoint: dict,
    config: DQNConfig,
    episodes: int,
    seed: int,
    device,
    video_recorder: EpisodeVideoRecorder | None = None,
) -> list[float]:
    env = make_envpool_env(
        config,
        num_envs=1,
        seed=seed,
        evaluation=True,
        render_mode="rgb_array" if video_recorder is not None else None,
    )
    try:
        q_net = build_q_model(
            name=config.model.name,
            observation_shape=env.observation_shape,
            num_actions=env.num_actions,
            kwargs=config.model.kwargs,
        ).to(device)
        q_net.load_state_dict(checkpoint["model_state"])
        return evaluate_q_policy(
            q_net=q_net,
            env=env,
            num_episodes=episodes,
            video_recorder=video_recorder,
        )
    finally:
        env.close()


def evaluate_ddpg_checkpoint(
    *,
    checkpoint: dict,
    config: DDPGConfig,
    episodes: int,
    seed: int,
    device,
    video_recorder: EpisodeVideoRecorder | None = None,
) -> list[float]:
    env = make_envpool_env(
        config,
        num_envs=1,
        seed=seed,
        evaluation=True,
        render_mode="rgb_array" if video_recorder is not None else None,
    )
    try:
        model = build_ddpg_actor_critic_model(
            name=config.model.name,
            observation_shape=env.observation_shape,
            action_spec=env.action_spec,
            kwargs=config.model.kwargs,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        return evaluate_ddpg_policy(
            model=model,
            env=env,
            num_episodes=episodes,
            video_recorder=video_recorder,
        )
    finally:
        env.close()


def evaluate_td3_checkpoint(
    *,
    checkpoint: dict,
    config: TD3Config,
    episodes: int,
    seed: int,
    device,
    video_recorder: EpisodeVideoRecorder | None = None,
) -> list[float]:
    env = make_envpool_env(
        config,
        num_envs=1,
        seed=seed,
        evaluation=True,
        render_mode="rgb_array" if video_recorder is not None else None,
    )
    try:
        model = build_td3_actor_critic_model(
            name=config.model.name,
            observation_shape=env.observation_shape,
            action_spec=env.action_spec,
            kwargs=config.model.kwargs,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        return evaluate_ddpg_policy(
            model=model,
            env=env,
            num_episodes=episodes,
            video_recorder=video_recorder,
        )
    finally:
        env.close()


def evaluate_actor_critic_checkpoint(
    *,
    checkpoint: dict,
    config: A2CConfig | PPOConfig,
    episodes: int,
    seed: int,
    device,
    video_recorder: EpisodeVideoRecorder | None = None,
) -> list[float]:
    observation_rms = (
        None
        if checkpoint["observation_normalization"] is None
        else running_mean_std_from_state(checkpoint["observation_normalization"])
    )
    env = make_envpool_env(
        config,
        num_envs=1,
        seed=seed,
        evaluation=True,
        observation_rms=observation_rms,
        render_mode="rgb_array" if video_recorder is not None else None,
    )
    try:
        model = build_actor_critic_model(
            name=config.model.name,
            observation_shape=env.observation_shape,
            action_spec=env.action_spec,
            kwargs=config.model.kwargs,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        return evaluate_actor_critic_policy(
            model=model,
            env=env,
            num_episodes=episodes,
            video_recorder=video_recorder,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
