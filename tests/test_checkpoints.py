import sys
from pathlib import Path

import pytest
import torch

from a2c import A2C
from checkpoints import (
    build_checkpoint_payload,
    load_checkpoint,
    resume_checkpoint_path,
    save_checkpoint,
)
from config import (
    A2CConfig,
    A2CTrainConfig,
    DQNConfig,
    DQNTrainConfig,
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    ExplorationConfig,
    LoggingConfig,
    ModelConfig,
    PPOConfig,
    PPOTrainConfig,
    save_config,
)
from dqn import DQN
from envs import EnvPoolVecEnv
from evaluate_checkpoint import (
    evaluate_actor_critic_checkpoint,
    evaluate_dqn_checkpoint,
)
from metrics import JSONLMetricsLogger
from models import build_actor_critic_model, build_q_model
from ppo import PPO
from train_a2c import main as train_a2c_main
from train_dqn import main as train_dqn_main
from train_ppo import main as train_ppo_main


def tiny_dqn_config(run_root: str = "runs") -> DQNConfig:
    return DQNConfig(
        experiment=ExperimentConfig(name="test", run_root=run_root),
        seed=123,
        env=EnvConfig(id="CartPole-v1", num_envs=1),
        model=ModelConfig(name="mlp", kwargs={"hidden_sizes": [8]}),
        train=DQNTrainConfig(
            steps=4,
            batch_size=2,
            buffer_capacity=16,
            learning_starts=0,
            learning_rate=0.001,
            discount_factor=0.99,
            soft_update_rate=0.005,
            max_grad_norm=1.0,
        ),
        exploration=ExplorationConfig(
            schedule="constant",
            start=1.0,
            end=1.0,
            decay_steps=1,
        ),
        eval=EvalConfig(every_steps=2, episodes=1, seed=10000),
        logging=LoggingConfig(loss_every_steps=2),
    )


def tiny_a2c_config(run_root: str = "runs") -> A2CConfig:
    return A2CConfig(
        experiment=ExperimentConfig(name="test", run_root=run_root),
        seed=123,
        env=EnvConfig(id="CartPole-v1", num_envs=2),
        model=ModelConfig(name="discrete_mlp", kwargs={"hidden_sizes": [8]}),
        train=A2CTrainConfig(
            steps=4,
            learning_rate=0.001,
            value_loss_coef=0.5,
            discount_factor=0.99,
            rollout_steps=2,
            max_grad_norm=1.0,
        ),
        eval=EvalConfig(every_steps=2, episodes=1, seed=10000),
        logging=LoggingConfig(loss_every_steps=2),
    )


def tiny_ppo_config(run_root: str = "runs") -> PPOConfig:
    return PPOConfig(
        experiment=ExperimentConfig(name="test", run_root=run_root),
        seed=123,
        env=EnvConfig(id="CartPole-v1", num_envs=2),
        model=ModelConfig(name="discrete_mlp", kwargs={"hidden_sizes": [8]}),
        train=PPOTrainConfig(
            steps=4,
            learning_rate=0.001,
            discount_factor=0.99,
            gae_lambda=0.95,
            rollout_steps=2,
            minibatch_size=4,
            epochs=1,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=1.0,
        ),
        eval=EvalConfig(every_steps=2, episodes=1, seed=10000),
        logging=LoggingConfig(loss_every_steps=2),
    )


def test_checkpoint_helper_round_trips_payload(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "algorithm": "dqn",
        "step": 3,
        "update": None,
        "config": {"experiment": {"name": "test"}},
        "model_state": {},
        "optimizer_state": {},
        "algorithm_state": {},
        "observation_normalization": None,
        "best_eval_mean_return": None,
        "best_step": None,
    }
    path = tmp_path / "checkpoints" / "last.pt"

    save_checkpoint(payload, path)
    loaded = load_checkpoint(path)

    assert loaded["schema_version"] == 1
    assert loaded["algorithm"] == "dqn"
    assert loaded["step"] == 3
    assert resume_checkpoint_path(tmp_path) == path


def test_resume_checkpoint_path_requires_run_directory_layout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find resume checkpoint"):
        resume_checkpoint_path(tmp_path)

    checkpoint_path = tmp_path / "checkpoints" / "last.pt"
    save_checkpoint(
        {
            "schema_version": 1,
            "algorithm": "dqn",
            "step": 3,
            "update": None,
            "config": {"experiment": {"name": "test"}},
            "model_state": {},
            "optimizer_state": {},
            "algorithm_state": {},
            "observation_normalization": None,
            "best_eval_mean_return": None,
            "best_step": None,
        },
        checkpoint_path,
    )

    with pytest.raises(NotADirectoryError, match="--resume must point"):
        resume_checkpoint_path(checkpoint_path)
    with pytest.raises(IsADirectoryError, match="checkpoint path must be a file"):
        load_checkpoint(tmp_path)


def test_dqn_checkpoint_state_restores_replay_buffer() -> None:
    env = EnvPoolVecEnv(env_id="CartPole-v1", num_envs=1, seed=1)
    restored_env = EnvPoolVecEnv(env_id="CartPole-v1", num_envs=1, seed=2)
    try:
        model = build_q_model(
            name="mlp",
            observation_shape=env.observation_shape,
            num_actions=env.num_actions,
            kwargs={"hidden_sizes": [8]},
        )
        agent = DQN(
            env,
            model,
            learning_rate=1e-3,
            discount_factor=0.99,
            soft_update_rate=0.005,
            buffer_capacity=16,
            batch_size=2,
            learning_starts=0,
            max_grad_norm=1.0,
        )
        agent.train(num_steps=4, exploration_rate_fn=lambda _step: 1.0)

        restored_model = build_q_model(
            name="mlp",
            observation_shape=restored_env.observation_shape,
            num_actions=restored_env.num_actions,
            kwargs={"hidden_sizes": [8]},
        )
        restored_agent = DQN(
            restored_env,
            restored_model,
            learning_rate=1e-3,
            discount_factor=0.99,
            soft_update_rate=0.005,
            buffer_capacity=16,
            batch_size=2,
            learning_starts=0,
            max_grad_norm=1.0,
        )
        restored_agent.load_checkpoint_state(agent.checkpoint_state())

        assert restored_agent.step == agent.step
        assert len(restored_agent.replay_buffer) == len(agent.replay_buffer)
        for name, value in agent.target_q_net.state_dict().items():
            assert torch.equal(restored_agent.target_q_net.state_dict()[name], value)
    finally:
        env.close()
        restored_env.close()


def test_a2c_and_ppo_checkpoint_state_restore_step_and_update() -> None:
    a2c_config = tiny_a2c_config()
    ppo_config = tiny_ppo_config()
    a2c_env = EnvPoolVecEnv(env_id=a2c_config.env.id, num_envs=2, seed=1)
    restored_a2c_env = EnvPoolVecEnv(env_id=a2c_config.env.id, num_envs=2, seed=2)
    ppo_env = EnvPoolVecEnv(env_id=ppo_config.env.id, num_envs=2, seed=3)
    restored_ppo_env = EnvPoolVecEnv(env_id=ppo_config.env.id, num_envs=2, seed=4)
    try:
        a2c_model = build_actor_critic_model(
            name=a2c_config.model.name,
            observation_shape=a2c_env.observation_shape,
            action_spec=a2c_env.action_spec,
            kwargs=a2c_config.model.kwargs,
        )
        a2c_agent = A2C(
            a2c_env,
            a2c_model,
            learning_rate=1e-3,
            value_loss_coef=0.5,
            discount_factor=0.99,
            rollout_steps=2,
            max_grad_norm=1.0,
        )
        a2c_agent.train(num_steps=4)
        restored_a2c_model = build_actor_critic_model(
            name=a2c_config.model.name,
            observation_shape=restored_a2c_env.observation_shape,
            action_spec=restored_a2c_env.action_spec,
            kwargs=a2c_config.model.kwargs,
        )
        restored_a2c_agent = A2C(
            restored_a2c_env,
            restored_a2c_model,
            learning_rate=1e-3,
            value_loss_coef=0.5,
            discount_factor=0.99,
            rollout_steps=2,
            max_grad_norm=1.0,
        )
        restored_a2c_agent.load_checkpoint_state(a2c_agent.checkpoint_state())

        ppo_model = build_actor_critic_model(
            name=ppo_config.model.name,
            observation_shape=ppo_env.observation_shape,
            action_spec=ppo_env.action_spec,
            kwargs=ppo_config.model.kwargs,
        )
        ppo_agent = PPO(
            ppo_env,
            ppo_model,
            learning_rate=1e-3,
            rollout_steps=2,
            minibatch_size=4,
            epochs=1,
            discount_factor=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=1.0,
        )
        ppo_agent.train(num_steps=4)
        restored_ppo_model = build_actor_critic_model(
            name=ppo_config.model.name,
            observation_shape=restored_ppo_env.observation_shape,
            action_spec=restored_ppo_env.action_spec,
            kwargs=ppo_config.model.kwargs,
        )
        restored_ppo_agent = PPO(
            restored_ppo_env,
            restored_ppo_model,
            learning_rate=1e-3,
            rollout_steps=2,
            minibatch_size=4,
            epochs=1,
            discount_factor=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=1.0,
        )
        restored_ppo_agent.load_checkpoint_state(ppo_agent.checkpoint_state())

        assert restored_a2c_agent.step == a2c_agent.step
        assert restored_a2c_agent.update == a2c_agent.update
        assert restored_ppo_agent.step == ppo_agent.step
        assert restored_ppo_agent.update == ppo_agent.update
    finally:
        a2c_env.close()
        restored_a2c_env.close()
        ppo_env.close()
        restored_ppo_env.close()


def test_metrics_logger_append_mode(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"

    with JSONLMetricsLogger(metrics_path) as metrics:
        metrics.write(step=1, loss=1.0)
    with JSONLMetricsLogger(metrics_path, append=True) as metrics:
        metrics.write(step=2, loss=0.5)

    assert len(metrics_path.read_text(encoding="utf-8").splitlines()) == 2


def test_dqn_train_saves_last_best_and_resume_appends(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tiny_dqn_config(run_root=str(tmp_path / "runs"))
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "run"

    save_config(config, config_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dqn.py",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--no-plot",
        ],
    )
    train_dqn_main()

    last_path = run_dir / "checkpoints" / "last.pt"
    best_path = run_dir / "checkpoints" / "best.pt"
    assert last_path.exists()
    assert best_path.exists()
    initial_line_count = len(
        (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dqn.py",
            "--resume",
            str(run_dir),
            "--set",
            "train.steps=8",
            "--no-plot",
        ],
    )
    train_dqn_main()

    resumed_line_count = len(
        (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )
    checkpoint = load_checkpoint(last_path, expected_algorithm="dqn")
    assert resumed_line_count > initial_line_count
    assert checkpoint["step"] >= 8


def test_dqn_train_does_not_write_best_checkpoint_before_first_eval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tiny_dqn_config(run_root=str(tmp_path / "runs"))
    config = config.model_copy(
        update={
            "eval": config.eval.model_copy(update={"every_steps": 100}),
        }
    )
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "run"

    save_config(config, config_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dqn.py",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--no-plot",
        ],
    )
    train_dqn_main()

    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert not (run_dir / "checkpoints" / "best.pt").exists()


def test_dqn_resume_recreates_best_checkpoint_on_next_eval_when_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tiny_dqn_config(run_root=str(tmp_path / "runs"))
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "run"

    save_config(config, config_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dqn.py",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--no-plot",
        ],
    )
    train_dqn_main()

    last_path = run_dir / "checkpoints" / "last.pt"
    best_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = load_checkpoint(last_path, expected_algorithm="dqn")
    checkpoint["best_eval_mean_return"] = 1_000_000.0
    checkpoint["best_step"] = checkpoint["step"]
    save_checkpoint(checkpoint, last_path)
    best_path.unlink()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dqn.py",
            "--resume",
            str(run_dir),
            "--set",
            "train.steps=8",
            "--no-plot",
        ],
    )
    train_dqn_main()

    recreated_best = load_checkpoint(best_path, expected_algorithm="dqn")
    assert recreated_best["step"] >= 6
    assert recreated_best["best_eval_mean_return"] < 1_000_000.0
    assert recreated_best["best_step"] == recreated_best["step"]


def test_a2c_train_saves_last_best_and_resume_appends(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tiny_a2c_config(run_root=str(tmp_path / "runs"))
    config_path = tmp_path / "a2c_config.yaml"
    run_dir = tmp_path / "a2c_run"

    save_config(config, config_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_a2c.py",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--no-plot",
        ],
    )
    train_a2c_main()

    last_path = run_dir / "checkpoints" / "last.pt"
    best_path = run_dir / "checkpoints" / "best.pt"
    assert last_path.exists()
    assert best_path.exists()
    initial_line_count = len(
        (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_a2c.py",
            "--resume",
            str(run_dir),
            "--set",
            "train.steps=8",
            "--no-plot",
        ],
    )
    train_a2c_main()

    resumed_line_count = len(
        (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )
    checkpoint = load_checkpoint(last_path, expected_algorithm="a2c")
    assert resumed_line_count > initial_line_count
    assert checkpoint["step"] >= 8
    assert checkpoint["update"] >= 2


def test_ppo_train_saves_last_best_and_resume_appends(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tiny_ppo_config(run_root=str(tmp_path / "runs"))
    config_path = tmp_path / "ppo_config.yaml"
    run_dir = tmp_path / "ppo_run"

    save_config(config, config_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--no-plot",
        ],
    )
    train_ppo_main()

    last_path = run_dir / "checkpoints" / "last.pt"
    best_path = run_dir / "checkpoints" / "best.pt"
    assert last_path.exists()
    assert best_path.exists()
    initial_line_count = len(
        (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--resume",
            str(run_dir),
            "--set",
            "train.steps=8",
            "--no-plot",
        ],
    )
    train_ppo_main()

    resumed_line_count = len(
        (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )
    checkpoint = load_checkpoint(last_path, expected_algorithm="ppo")
    assert resumed_line_count > initial_line_count
    assert checkpoint["step"] >= 8
    assert checkpoint["update"] >= 2


@pytest.mark.parametrize("algorithm", ["dqn", "a2c", "ppo"])
def test_resumed_training_matches_uninterrupted_result(
    algorithm: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_factory = {
        "dqn": tiny_dqn_config,
        "a2c": tiny_a2c_config,
        "ppo": tiny_ppo_config,
    }[algorithm]
    train_main = {
        "dqn": train_dqn_main,
        "a2c": train_a2c_main,
        "ppo": train_ppo_main,
    }[algorithm]
    eval_checkpoint = {
        "dqn": evaluate_dqn_checkpoint,
        "a2c": evaluate_actor_critic_checkpoint,
        "ppo": evaluate_actor_critic_checkpoint,
    }[algorithm]

    run_root = str(tmp_path / "runs")
    base_config = config_factory(run_root)
    continuous_config = base_config.model_copy(
        update={"train": base_config.train.model_copy(update={"steps": 8})}
    )
    pause_config = config_factory(run_root)
    continuous_config_path = tmp_path / f"{algorithm}_continuous.yaml"
    pause_config_path = tmp_path / f"{algorithm}_pause.yaml"
    continuous_run_dir = tmp_path / f"{algorithm}_continuous"
    resumed_run_dir = tmp_path / f"{algorithm}_resumed"

    save_config(continuous_config, continuous_config_path)
    save_config(pause_config, pause_config_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            f"train_{algorithm}.py",
            "--config",
            str(continuous_config_path),
            "--run-dir",
            str(continuous_run_dir),
            "--no-plot",
        ],
    )
    train_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            f"train_{algorithm}.py",
            "--config",
            str(pause_config_path),
            "--run-dir",
            str(resumed_run_dir),
            "--no-plot",
        ],
    )
    train_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            f"train_{algorithm}.py",
            "--resume",
            str(resumed_run_dir),
            "--set",
            "train.steps=8",
            "--no-plot",
        ],
    )
    train_main()

    continuous_checkpoint = load_checkpoint(
        continuous_run_dir / "checkpoints" / "last.pt",
        expected_algorithm=algorithm,
    )
    resumed_checkpoint = load_checkpoint(
        resumed_run_dir / "checkpoints" / "last.pt",
        expected_algorithm=algorithm,
    )
    continuous_returns = eval_checkpoint(
        checkpoint=continuous_checkpoint,
        config=continuous_config,
        episodes=3,
        seed=54321,
        device=torch.device("cpu"),
    )
    resumed_returns = eval_checkpoint(
        checkpoint=resumed_checkpoint,
        config=continuous_config,
        episodes=3,
        seed=54321,
        device=torch.device("cpu"),
    )

    continuous_mean = sum(continuous_returns) / len(continuous_returns)
    resumed_mean = sum(resumed_returns) / len(resumed_returns)
    assert continuous_checkpoint["step"] == resumed_checkpoint["step"] == 8
    assert continuous_checkpoint["update"] == resumed_checkpoint["update"]
    assert abs(continuous_mean - resumed_mean) <= 2.0


def test_evaluate_checkpoint_helpers_return_episode_returns() -> None:
    dqn_config = tiny_dqn_config()
    a2c_config = tiny_a2c_config()
    dqn_env = EnvPoolVecEnv(env_id=dqn_config.env.id, num_envs=1, seed=1)
    a2c_env = EnvPoolVecEnv(env_id=a2c_config.env.id, num_envs=2, seed=2)
    try:
        dqn_model = build_q_model(
            name=dqn_config.model.name,
            observation_shape=dqn_env.observation_shape,
            num_actions=dqn_env.num_actions,
            kwargs=dqn_config.model.kwargs,
        )
        dqn_agent = DQN(
            dqn_env,
            dqn_model,
            learning_rate=1e-3,
            discount_factor=0.99,
            soft_update_rate=0.005,
            buffer_capacity=16,
            batch_size=2,
            learning_starts=0,
            max_grad_norm=1.0,
        )
        dqn_agent.train(num_steps=4, exploration_rate_fn=lambda _step: 1.0)
        dqn_checkpoint = build_checkpoint_payload(
            algorithm="dqn",
            config=dqn_config,
            agent_state=dqn_agent.checkpoint_state(),
            observation_normalization=None,
            best_eval_mean_return=None,
            best_step=None,
        )

        a2c_model = build_actor_critic_model(
            name=a2c_config.model.name,
            observation_shape=a2c_env.observation_shape,
            action_spec=a2c_env.action_spec,
            kwargs=a2c_config.model.kwargs,
        )
        a2c_agent = A2C(
            a2c_env,
            a2c_model,
            learning_rate=1e-3,
            value_loss_coef=0.5,
            discount_factor=0.99,
            rollout_steps=2,
            max_grad_norm=1.0,
        )
        a2c_agent.train(num_steps=4)
        a2c_checkpoint = build_checkpoint_payload(
            algorithm="a2c",
            config=a2c_config,
            agent_state=a2c_agent.checkpoint_state(),
            observation_normalization=None,
            best_eval_mean_return=None,
            best_step=None,
        )

        assert (
            len(
                evaluate_dqn_checkpoint(
                    checkpoint=dqn_checkpoint,
                    config=dqn_config,
                    episodes=1,
                    seed=100,
                    device=torch.device("cpu"),
                )
            )
            == 1
        )
        assert (
            len(
                evaluate_actor_critic_checkpoint(
                    checkpoint=a2c_checkpoint,
                    config=a2c_config,
                    episodes=1,
                    seed=101,
                    device=torch.device("cpu"),
                )
            )
            == 1
        )
    finally:
        dqn_env.close()
        a2c_env.close()
