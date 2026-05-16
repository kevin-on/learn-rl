import argparse
import json
from pathlib import Path
from typing import Any


def load_metric_records(metrics_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with metrics_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "step" not in record:
                msg = f"{metrics_path}:{line_number} is missing required key 'step'."
                raise ValueError(msg)
            records.append(record)
    return records


def plot_metrics(
    metrics_path: Path, output_path: Path, title: str | None = None
) -> None:
    records = load_metric_records(metrics_path)
    if not records:
        msg = f"No metric records found in {metrics_path}."
        raise ValueError(msg)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    if title is not None:
        fig.suptitle(title)

    _plot_returns(axes[0], records)
    _plot_loss(axes[1], records)
    _plot_auxiliary_metric(axes[2], records)

    axes[2].set_xlabel("environment step")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_returns(ax: Any, records: list[dict[str, Any]]) -> None:
    train_steps, train_returns = _series(records, "train_episode_return")
    mean_steps, mean_returns = _series(records, "train_episode_return_mean20")
    eval_records = [record for record in records if "eval_mean_return" in record]

    if train_steps:
        ax.plot(
            train_steps,
            train_returns,
            color="#4c78a8",
            alpha=0.35,
            linewidth=1.0,
            label="train episode return",
        )
    if mean_steps:
        ax.plot(
            mean_steps,
            mean_returns,
            color="#1f4e79",
            linewidth=1.8,
            label="train mean20 return",
        )
    if eval_records:
        eval_steps = [int(record["step"]) for record in eval_records]
        eval_means = [float(record["eval_mean_return"]) for record in eval_records]
        eval_stds = [
            float(record.get("eval_std_return", 0.0)) for record in eval_records
        ]
        lower = [mean - std for mean, std in zip(eval_means, eval_stds, strict=True)]
        upper = [mean + std for mean, std in zip(eval_means, eval_stds, strict=True)]
        ax.plot(
            eval_steps,
            eval_means,
            color="#f58518",
            marker="o",
            linewidth=1.8,
            label="eval mean return",
        )
        ax.fill_between(eval_steps, lower, upper, color="#f58518", alpha=0.16)

    _finish_axis(ax, ylabel="return")


def _plot_loss(ax: Any, records: list[dict[str, Any]]) -> None:
    loss_steps, losses = _series(records, "loss")
    policy_loss_steps, policy_losses = _series(records, "policy_loss")
    value_loss_steps, value_losses = _series(records, "value_loss")
    actor_loss_steps, actor_losses = _series(records, "actor_loss")
    critic_loss_steps, critic_losses = _series(records, "critic_loss")
    if loss_steps:
        ax.plot(loss_steps, losses, color="#54a24b", linewidth=1.0, label="loss")
    if policy_loss_steps:
        ax.plot(
            policy_loss_steps,
            policy_losses,
            color="#4c78a8",
            alpha=0.8,
            linewidth=1.0,
            label="policy loss",
        )
    if value_loss_steps:
        ax.plot(
            value_loss_steps,
            value_losses,
            color="#f58518",
            alpha=0.8,
            linewidth=1.0,
            label="value loss",
        )
    if actor_loss_steps:
        ax.plot(
            actor_loss_steps,
            actor_losses,
            color="#4c78a8",
            alpha=0.8,
            linewidth=1.0,
            label="actor loss",
        )
    if critic_loss_steps:
        ax.plot(
            critic_loss_steps,
            critic_losses,
            color="#f58518",
            alpha=0.8,
            linewidth=1.0,
            label="critic loss",
        )
    _finish_axis(ax, ylabel="loss")


def _plot_auxiliary_metric(ax: Any, records: list[dict[str, Any]]) -> None:
    epsilon_steps, epsilons = _series(records, "epsilon")
    if epsilon_steps:
        ax.plot(
            epsilon_steps,
            epsilons,
            color="#b279a2",
            linewidth=1.5,
            label="epsilon",
        )
        _finish_axis(ax, ylabel="epsilon")
        return

    grad_norm_steps, grad_norms = _series(records, "grad_norm")
    if grad_norm_steps:
        ax.plot(
            grad_norm_steps,
            grad_norms,
            color="#b279a2",
            linewidth=1.5,
            label="grad norm",
        )
    _finish_axis(ax, ylabel="grad norm")


def _series(records: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    points = [
        (int(record["step"]), float(record[key])) for record in records if key in record
    ]
    if not points:
        return [], []
    steps, values = zip(*points, strict=True)
    return list(steps), list(values)


def _finish_axis(ax: Any, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RL metrics from a JSONL file.")
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output_path = args.output or args.metrics_path.with_name("metrics.png")
    plot_metrics(args.metrics_path, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
