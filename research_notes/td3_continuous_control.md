# TD3 Continuous Control Results

Date: 2026-05-19

This note records the TD3 runs for the same continuous-control task set used in
the DDPG sweep. All runs used `src/train_td3.py`, CPU execution, seed `123`, and
config-only changes. The selected YAML configs are saved under `configs/*td3.yaml`.

Public comparison numbers come from RL Baselines3 Zoo/SB3 model cards, CleanRL
TD3 benchmark documentation, and one Ant-v4 Hugging Face model card. They are
orientation targets rather than exact apples-to-apples results because seeds,
evaluation protocols, hardware, Gym/Gymnasium versions, and logging conventions
differ.

## Summary

| Environment | Selected run | Best train eval | Final train eval | Fresh 100-episode eval | Best web comparison |
|---|---|---:|---:|---:|---:|
| `Pendulum-v1` | `runs/pendulum_td3_rlzoo_seed123_20k` | `-98.280 +/- 48.258` at 20k | `-98.280 +/- 48.258` | `-151.987 +/- 78.454` | RL Zoo TD3: `-151.855 +/- 90.227` |
| `MountainCarContinuous-v0` | `runs/mountain_car_continuous_td3_high_exploration_sigma1_seed123_300k` | `93.506 +/- 0.076` at 190k | `93.457 +/- 0.066` | `93.455 +/- 0.090` | RL Zoo TD3: `93.483 +/- 0.075` |
| `LunarLanderContinuous-v2` | `runs/lunar_lander_continuous_td3_rlzoo_seed123_300k` | `228.577 +/- 41.475` at 260k | `7.112 +/- 165.248` | `193.143 +/- 107.689` | SB3 card: `222.26 +/- 79.64` |
| `HalfCheetah-v4` | `runs/half_cheetah_td3_cleanrl_seed123_1m` | `9031.970 +/- 119.736` at 950k | `9014.811 +/- 58.198` | `8991.635 +/- 93.401` | CleanRL card: `10122.22 +/- 119.98` |
| `Hopper-v4` | `runs/hopper_td3_cleanrl_seed123_1m` | `3362.156 +/- 38.703` at 900k | `3284.403 +/- 8.370` | `3355.799 +/- 32.800` | CleanRL docs: `3134.61 +/- 360.18` |
| `Walker2d-v4` | `runs/walker2d_td3_rlzoo_seed123_1m` | `4440.958 +/- 27.895` at 700k | `4298.679 +/- 42.500` | `4447.541 +/- 31.913` | CleanRL docs: `4057.59 +/- 658.78` |
| `Ant-v4` | `runs/ant_td3_cleanrl_seed123_1m` | `5285.597 +/- 161.495` at 975k | `5265.959 +/- 151.676` | `5278.787 +/- 339.475` | HF TD3 seed4: `4917.41 +/- 599.22` |

## Selected Configs

| Environment | Config | Main settings |
|---|---|---|
| `Pendulum-v1` | `configs/pendulum_td3.yaml` | RL Zoo-style `[400, 300]`, lr `1e-3`, `learning_starts=10000`, normal action noise `0.1`, 20k steps |
| `MountainCarContinuous-v0` | `configs/mountain_car_continuous_td3.yaml` | RL Zoo-style `[400, 300]`, lr `1e-3`, `learning_starts=100`, higher action noise `1.0`, 300k steps |
| `LunarLanderContinuous-v2` | `configs/lunar_lander_continuous_td3.yaml` | RL Zoo-style `[400, 300]`, lr `1e-3`, `learning_starts=10000`, action noise `0.1`, 300k steps |
| `HalfCheetah-v4` | `configs/half_cheetah_td3.yaml` | CleanRL-style `[256, 256]`, lr `3e-4`, `learning_starts=25000`, action noise `0.1`, 1M steps |
| `Hopper-v4` | `configs/hopper_td3.yaml` | CleanRL-style `[256, 256]`, lr `3e-4`, `learning_starts=25000`, action noise `0.1`, 1M steps |
| `Walker2d-v4` | `configs/walker2d_td3.yaml` | RL Zoo-style `[400, 300]`, lr `1e-3`, `learning_starts=10000`, action noise `0.1`, 1M steps |
| `Ant-v4` | `configs/ant_td3.yaml` | CleanRL-style `[256, 256]`, lr `3e-4`, `learning_starts=25000`, action noise `0.1`, 1M steps |

## Findings

TD3 was broadly stronger and easier to tune than DDPG on this task set. The
final selected runs are comparable to the public references on every task, and
they exceed the referenced web numbers for Hopper, Walker2d, and Ant in the
single-seed checks.

MountainCarContinuous needed more exploration than the direct RL Zoo-style TD3
setting in this implementation. The first `sigma=0.5` run did not solve the
task, while `sigma=1.0` reached the same range as the SB3/RL Zoo result.

Walker2d was the other major correction. The original CleanRL-style
`[256, 256]`, lr `3e-4`, 25k warm-up config learned poorly. Switching the
canonical `configs/walker2d_td3.yaml` to the RL Zoo-style `[400, 300]`, lr
`1e-3`, 10k warm-up recipe produced the best Walker2d run.

LunarLanderContinuous is the main unstable result. The best checkpoint at 260k
is competitive with the SB3 card, but the final checkpoint collapsed to a very
low mean return. Use `checkpoints/best.pt` for this task, not `checkpoints/last.pt`.

For MuJoCo, CleanRL-style settings worked well for HalfCheetah, Hopper, and Ant.
Walker2d was the exception and preferred the larger RL Zoo-style actor/critic.

## Verification

The best and final train eval values above were read from each run's
`metrics.jsonl`. The final reusable YAMLs were loaded successfully with
`load_td3_config` through:

```bash
PYTHONPATH=src uv run python - <<'PY'
from pathlib import Path
from config import load_td3_config
for path in sorted(Path("configs").glob("*td3*.yaml")):
    load_td3_config(path)
PY
```

## Sources

- RL Zoo benchmark table:
  https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
- SB3 TD3 Pendulum model card:
  https://huggingface.co/sb3/td3-Pendulum-v1
- SB3 TD3 MountainCarContinuous model card:
  https://huggingface.co/sb3/td3-MountainCarContinuous-v0
- SB3 TD3 LunarLanderContinuous model card:
  https://huggingface.co/sb3/td3-LunarLanderContinuous-v2
- CleanRL TD3 benchmark docs:
  https://docs.cleanrl.dev/rl-algorithms/td3/
- CleanRL HalfCheetah-v4 TD3 model card:
  https://huggingface.co/cleanrl/HalfCheetah-v4-td3_continuous_action-seed1
- CleanRL Hopper-v4 TD3 model card:
  https://huggingface.co/cleanrl/Hopper-v4-td3_continuous_action-seed1
- CleanRL Walker2d-v4 TD3 model card:
  https://huggingface.co/cleanrl/Walker2d-v4-td3_continuous_action-seed1
- Ant-v4 TD3 model card used for the Ant comparison:
  https://huggingface.co/sdpkjc/Ant-v4-td3_continuous_action-seed4
