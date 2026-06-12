from __future__ import annotations
from pathlib import Path


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    labels: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """Plot a confusion matrix heatmap and save to file.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels.
        labels: Ordered label names for axis ticks.
        output_path: Destination PNG path.
        title: Plot title.
        figsize: Figure size in inches.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric: str,
    output_path: Path,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
    ylim: tuple[float, float] | None = None,
) -> None:
    """Bar chart comparing multiple models on a single metric.

    Args:
        results: {model_name: {metric_name: value, ...}}.
        metric: Which metric key to plot.
        output_path: Destination PNG path.
        title: Plot title (defaults to metric name).
        figsize: Figure size in inches.
        ylim: Optional y-axis (min, max).
    """
    import matplotlib.pyplot as plt

    names = list(results.keys())
    values = [results[n].get(metric, 0.0) for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, values)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_title(title or metric)
    ax.set_ylabel(metric)
    if ylim:
        ax.set_ylim(ylim)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
