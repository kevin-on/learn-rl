# DDPG Continuous Control Results

Date: 2026-05-17

This note records the first DDPG runs for Pendulum, LunarLanderContinuous, and
MountainCarContinuous. Public comparison numbers are from SB3/RL Zoo model cards
and RL Zoo benchmark tables.

## Sources

- RL Zoo benchmark table:
  https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
- SB3 Pendulum DDPG:
  https://huggingface.co/sb3/ddpg-Pendulum-v1
- SB3 LunarLanderContinuous DDPG:
  https://huggingface.co/sb3/ddpg-LunarLanderContinuous-v2
- SB3 MountainCarContinuous DDPG:
  https://huggingface.co/sb3/ddpg-MountainCarContinuous-v0

## Summary

| Environment | Run directory | Our result | Web comparison | Notes |
| --- | --- | ---: | ---: | --- |
| `Pendulum-v1` | `runs/pendulum_ddpg_rlzoo_adapted_seed123_20k` | `-158.219 +/- 85.525` over 750 eval episodes | RL Zoo: `-152.099 +/- 94.282`; SB3 card: `-211.65 +/- 134.05` | Close to RL Zoo. Earlier implementation used OU noise, not exact SB3 normal-noise warm-up semantics. |
| `LunarLanderContinuous-v2` | `runs/lunar_lander_continuous_ddpg_sb3_warmup_seed123_300k` | best training eval: `218.415 +/- 59.899` at 160k; fresh 100-episode eval: `145.969 +/- 122.296` | RL Zoo: `230.217 +/- 92.372`; SB3 card: `223.87 +/- 80.41` | Learns the task but is seed-sensitive/high variance. |
| `MountainCarContinuous-v0` | `runs/mountain_car_continuous_ddpg_sb3_warmup_seed123_300k` | best/final training eval: `93.618 +/- 0.031`; fresh 100-episode eval: `93.634 +/- 0.053` | RL Zoo: `93.512 +/- 0.048`; SB3 card: `93.51 +/- 0.05` | Matches SB3/RL Zoo closely. |

## Config Notes

- DDPG model architecture: CleanRL-style actor/critic MLP with hidden sizes
  `[400, 300]`.
- `LunarLanderContinuous-v2` uses SB3/RL Zoo-style normal action noise:
  `noise_type=normal`, `sigma=0.1`, `learning_starts=10000`.
- `MountainCarContinuous-v0` uses SB3/RL Zoo-style OU action noise:
  `noise_type=ornstein-uhlenbeck`, `sigma=0.5`, `learning_starts=100`.
- The Pendulum run was done before adding exact SB3-style `noise_type` and
  random-action warm-up support, so it should be treated as RL Zoo-inspired
  rather than exact RL Zoo reproduction.

## Verification

- `uv run ruff check src tests`
- `uv run pytest`

Latest full verification after adding DDPG noise type and learning warm-up:
`64 passed`.
