# DDPG Continuous Control Results

Date: 2026-05-17

This note records the DDPG runs for classic continuous-control tasks and MuJoCo
locomotion tasks. Public comparison numbers are from SB3/RL Zoo model cards,
RL Zoo benchmark/config tables, and CleanRL model cards.

## Sources

- RL Zoo benchmark table:
  https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
- SB3 Pendulum DDPG:
  https://huggingface.co/sb3/ddpg-Pendulum-v1
- SB3 LunarLanderContinuous DDPG:
  https://huggingface.co/sb3/ddpg-LunarLanderContinuous-v2
- SB3 MountainCarContinuous DDPG:
  https://huggingface.co/sb3/ddpg-MountainCarContinuous-v0
- RL Zoo DDPG MuJoCo hyperparameters:
  https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ddpg.yml
- Spinning Up DDPG reference implementation defaults:
  https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ddpg/ddpg.html
- CleanRL DDPG MuJoCo cards:
  https://huggingface.co/cleanrl/HalfCheetah-v4-ddpg_continuous_action-seed1
  https://huggingface.co/cleanrl/Hopper-v4-ddpg_continuous_action-seed1
  https://huggingface.co/cleanrl/Walker2d-v4-ddpg_continuous_action-seed1
  https://huggingface.co/cleanrl/Ant-v4-ddpg_continuous_action-seed1

## Summary

### Classic Continuous Control

| Environment | Run directory | Our result | Web comparison | Notes |
| --- | --- | ---: | ---: | --- |
| `Pendulum-v1` | `runs/pendulum_ddpg_rlzoo_adapted_seed123_20k` | `-158.219 +/- 85.525` over 750 eval episodes | RL Zoo: `-152.099 +/- 94.282`; SB3 card: `-211.65 +/- 134.05` | Close to RL Zoo. Earlier implementation used OU noise, not exact SB3 normal-noise warm-up semantics. |
| `LunarLanderContinuous-v2` | `runs/lunar_lander_continuous_ddpg_sb3_warmup_seed123_300k` | best training eval: `218.415 +/- 59.899` at 160k; fresh 100-episode eval: `145.969 +/- 122.296` | RL Zoo: `230.217 +/- 92.372`; SB3 card: `223.87 +/- 80.41` | Learns the task but is seed-sensitive/high variance. |
| `MountainCarContinuous-v0` | `runs/mountain_car_continuous_ddpg_sb3_warmup_seed123_300k` | best/final training eval: `93.618 +/- 0.031`; fresh 100-episode eval: `93.634 +/- 0.053` | RL Zoo: `93.512 +/- 0.048`; SB3 card: `93.51 +/- 0.05` | Matches SB3/RL Zoo closely. |

### MuJoCo Locomotion

| Environment | Selected run directory | Best training eval | Fresh 100-episode eval | CleanRL comparison | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `HalfCheetah-v4` | `runs/half_cheetah_ddpg_rlzoo_mujoco_seed123_1m` | `11423.716 +/- 151.925` at 875k | best checkpoint: `11248.164 +/- 835.464`; last checkpoint: `11305.299 +/- 806.737` | `10923.04 +/- 157.23` | RL Zoo-style `[400, 300]`, lr `1e-3`, normal noise `0.1` worked very well. |
| `Hopper-v4` | `runs/hopper_ddpg_cleanrl_seed123_1m` | `2059.876 +/- 596.884` at 325k | `2128.379 +/- 580.835` | `1675.53 +/- 1038.51` | RL Zoo-style lr `1e-3` collapsed early; CleanRL-style lr `3e-4`, `[256, 256]`, and 25k warm-up was better. |
| `Walker2d-v4` | `runs/walker2d_ddpg_paper_seed123_1m` | `1194.622 +/- 404.723` at 775k | `1081.602 +/- 446.024` | `1129.42 +/- 1251.17` | CleanRL/RL Zoo/Spinning Up variants were weak or unstable; paper-style split lr and OU noise was the best. |
| `Ant-v4` | `runs/ant_ddpg_cleanrl_seed123_1m` | `994.348 +/- 2.332` at 25k | `995.695 +/- 2.831` | `655.97 +/- 381.88` | CleanRL-style config solved the comparison target early, but later 50k eval collapsed, so keep the best checkpoint. |

## Config Notes

- DDPG model architecture: CleanRL-style actor/critic MLP with hidden sizes
  selected per task.
- `LunarLanderContinuous-v2` uses SB3/RL Zoo-style normal action noise:
  `noise_type=normal`, `sigma=0.1`, `learning_starts=10000`.
- `MountainCarContinuous-v0` uses SB3/RL Zoo-style OU action noise:
  `noise_type=ornstein-uhlenbeck`, `sigma=0.5`, `learning_starts=100`.
- The Pendulum run was done before adding exact SB3-style `noise_type` and
  random-action warm-up support, so it should be treated as RL Zoo-inspired
  rather than exact RL Zoo reproduction.
- `HalfCheetah-v4` uses the RL Zoo MuJoCo recipe in
  `configs/half_cheetah_ddpg.yaml`: `[400, 300]`, actor/critic lr `1e-3`,
  batch `256`, `learning_starts=10000`, tau `0.005`, normal noise `0.1`.
- `Hopper-v4` and `Ant-v4` use the CleanRL-style recipe in
  `configs/hopper_ddpg.yaml` and `configs/ant_ddpg.yaml`: `[256, 256]`,
  actor/critic lr `3e-4`, batch `256`, `learning_starts=25000`, tau `0.005`,
  normal noise `0.1`.
- `Walker2d-v4` needed a more conservative paper-style recipe in
  `configs/walker2d_ddpg.yaml`: `[400, 300]`, actor lr `1e-4`, critic lr
  `1e-3`, critic weight decay `0.01`, batch `64`, tau `0.001`, OU noise
  `0.2`.

## Key Insights

- DDPG is much more task-sensitive than the PPO/A2C runs in this repo. The same
  MuJoCo defaults that were excellent on `HalfCheetah-v4` were unstable on
  `Hopper-v4` and weak on `Walker2d-v4`.
- Fresh 100-episode evaluation matters. `Walker2d-v4` crossed the CleanRL mean
  on a 10-episode training eval at 450k, but the fresh 100-episode check was
  only `739.453 +/- 626.169`; the later 775k checkpoint was more robust.
- The best checkpoint can be materially better than the final checkpoint. This
  was clear for `Hopper-v4` and especially `Ant-v4`, where the 25k checkpoint
  was strong but the 50k eval had already collapsed.
- Normal action noise with a longer random warm-up worked well for
  `Hopper-v4` and `Ant-v4`, matching the CleanRL-style recipe. `Walker2d-v4`
  benefited from older DDPG settings: slower actor updates via lower actor lr,
  smaller tau, smaller batch, critic weight decay, and OU noise.
- `HalfCheetah-v4` remains the cleanest MuJoCo DDPG task in these runs: the
  RL Zoo-style config exceeded the CleanRL card and stayed strong through the
  final checkpoint.

## Verification

- `uv run ruff check src tests`
- `uv run pytest`

Latest full verification after adding DDPG noise type and learning warm-up:
`64 passed`.

MuJoCo config validation after the Hopper/Walker2d/Ant sweep: loaded
`configs/half_cheetah_ddpg.yaml`, `configs/hopper_ddpg.yaml`,
`configs/walker2d_ddpg.yaml`, and `configs/ant_ddpg.yaml` with
`load_ddpg_config`.
