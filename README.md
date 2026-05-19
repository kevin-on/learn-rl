# learn-rl

Small reinforcement-learning experiments for Gymnasium-compatible EnvPool tasks.

Implemented algorithms:

- DQN
- DDPG
- TD3
- SAC
- A2C
- A3C
- PPO

## Train

Each algorithm has its own entrypoint:

```sh
uv run python src/train_dqn.py --config configs/cartpole_dqn.yaml
uv run python src/train_ddpg.py --config configs/pendulum_ddpg.yaml
uv run python src/train_td3.py --config configs/pendulum_td3.yaml
uv run python src/train_sac.py --config configs/pendulum_sac.yaml
uv run python src/train_a2c.py --config configs/cartpole_a2c.yaml
uv run python src/train_a3c.py --config configs/cartpole_a3c.yaml
uv run python src/train_ppo.py --config configs/cartpole_ppo.yaml
```

DQN, DDPG, TD3, SAC, A2C, and PPO train through a shared EnvPool vector-environment
interface. A3C keeps its worker-process training loop, while sharing the common
model and math utilities where practical.

Each run writes:

- `config.yaml`: the resolved config used for the run
- `metrics.jsonl`: training, evaluation, and algorithm-specific metrics
- `metrics.png`: a plot of returns and optimization metrics
- `checkpoints/last.pt`: the latest resumable DQN/DDPG/TD3/SAC/A2C/PPO checkpoint
- `checkpoints/best.pt`: the best DQN/DDPG/TD3/SAC/A2C/PPO checkpoint by evaluation
  mean return after at least one evaluation has run
- `checkpoints/step_<step>.pt`: optional DQN/DDPG/TD3/SAC/A2C/PPO periodic
  checkpoints when `checkpoint.every_steps` is set

By default outputs go under `runs/`, which is git-ignored.

## Resume and Evaluate Checkpoints

DQN, DDPG, TD3, SAC, A2C, and PPO resume from a run directory by loading
`checkpoints/last.pt`:

```sh
uv run python src/train_dqn.py --resume runs/<run-name> --set train.steps=20000
```

Save additional periodic checkpoints every K environment steps:

```yaml
checkpoint:
  every_steps: 50000
```

The CLI flag can override that value for a specific run:

```sh
uv run python src/train_dqn.py --config configs/cartpole_dqn.yaml --checkpoint-every-steps 50000
```

Evaluate a saved DQN/DDPG/TD3/SAC/A2C/PPO checkpoint:

```sh
uv run python src/evaluate_checkpoint.py runs/<run-name>/checkpoints/best.pt --episodes 20
```

Save MP4 videos from checkpoint evaluation:

```sh
uv run python src/evaluate_checkpoint.py runs/<run-name>/checkpoints/best.pt \
  --episodes 5 \
  --video-dir runs/<run-name>/videos \
  --video-episodes 2 \
  --video-crf 28 \
  --video-preset medium
```

Video capture uses EnvPool `rgb_array` rendering and saves only the first
episode by default when `--video-dir` is set. Use `--video-frame-stride` and
`--video-fps` to reduce frame count, and `--video-encoder-workers` to encode
completed episode videos in parallel.

## Override Hyperparameters

Use `--set` with dotted config keys:

```sh
uv run python src/train_dqn.py --config configs/cartpole_dqn.yaml --set train.steps=1000 --set seed=456
```

Vectorized DQN/DDPG/TD3/SAC/A2C/PPO runs can override `env.num_envs`:

```sh
uv run python src/train_a2c.py --config configs/cartpole_a2c.yaml --set env.num_envs=4
```

DQN/DDPG/TD3 exploration settings live under `train.exploration`:

```sh
uv run python src/train_dqn.py --config configs/cartpole_dqn.yaml --set train.exploration.end=0.05
```

For list values, quote the shell argument:

```sh
uv run python src/train_dqn.py --config configs/cartpole_dqn.yaml --set 'model.kwargs.hidden_sizes=[64, 64]'
```

## Replot Existing Metrics

```sh
uv run python src/plot_metrics.py runs/<run-name>/metrics.jsonl
```

## Checks

```sh
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```
