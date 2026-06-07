"""
visualise.py - Built-in visualisation module for NumCompute-Stream.

Provides reusable plotting functions backed by matplotlib.  All functions
accept an optional ``ax`` parameter so callers can embed plots in larger
figures, and an optional ``save_path`` to write the figure to disk.

Functions
---------
plot_metric_over_time        : Line plot of a single metric across chunks.
compare_models               : Side-by-side comparison of two metric series.
plot_predictions_vs_ground_truth : Scatter of predicted vs actual labels.
plot_confusion_matrix        : Heatmap of a confusion matrix.
plot_feature_importance      : Bar chart of feature importances.
plot_streaming_dashboard     : Multi-panel summary dashboard.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path, show):
    """Save figure to disk or display it inline."""
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig


def _make_ax(ax, figsize):
    """Return (fig, ax), creating a new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_metric_over_time
# ---------------------------------------------------------------------------

def plot_metric_over_time(metric_values, title="Metric over Time",
                          ylabel="Value", xlabel="Chunk",
                          color="#2196F3", marker="o",
                          ax=None, save_path=None, show=False):
    """
    Plot a single metric (e.g. accuracy) across streaming chunks.

    Parameters
    ----------
    metric_values : list or np.ndarray  One value per chunk.
    title         : str    Plot title.
    ylabel        : str    Y-axis label.
    xlabel        : str    X-axis label (default 'Chunk').
    color         : str    Line colour (default blue).
    marker        : str    Marker style (default 'o').
    ax            : matplotlib.axes.Axes or None
    save_path     : str or None  If set, save figure here.
    show          : bool   Call plt.show() (default False).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_ax(ax, figsize=(9, 4))
    x = np.arange(len(metric_values))
    ax.plot(x, metric_values, color=color, marker=marker,
            linewidth=2, markersize=5, markerfacecolor="white",
            markeredgewidth=1.5)
    ax.fill_between(x, metric_values, alpha=0.08, color=color)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(-0.3, max(len(metric_values) - 0.7, 0.7))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

def compare_models(metric1, metric2, labels=("Model A", "Model B"),
                   title="Model Comparison", ylabel="Accuracy",
                   xlabel="Chunk", colors=("#2196F3", "#F44336"),
                   ax=None, save_path=None, show=False):
    """
    Overlay two metric series to compare streaming model performance.

    Parameters
    ----------
    metric1 : list or np.ndarray  First model's per-chunk metrics.
    metric2 : list or np.ndarray  Second model's per-chunk metrics.
    labels  : tuple of str        Legend labels for each model.
    title   : str
    ylabel  : str
    xlabel  : str
    colors  : tuple of str        Line colours.
    ax      : matplotlib.axes.Axes or None
    save_path : str or None
    show    : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _make_ax(ax, figsize=(10, 4))
    x1 = np.arange(len(metric1))
    x2 = np.arange(len(metric2))

    ax.plot(x1, metric1, color=colors[0], marker="o", linewidth=2,
            markersize=5, label=labels[0],
            markerfacecolor="white", markeredgewidth=1.5)
    ax.plot(x2, metric2, color=colors[1], marker="s", linewidth=2,
            markersize=5, label=labels[1],
            markerfacecolor="white", markeredgewidth=1.5)

    ax.fill_between(x1, metric1, alpha=0.07, color=colors[0])
    ax.fill_between(x2, metric2, alpha=0.07, color=colors[1])

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# plot_predictions_vs_ground_truth
# ---------------------------------------------------------------------------

def plot_predictions_vs_ground_truth(y_true, y_pred,
                                     title="Predictions vs Ground Truth",
                                     ax=None, save_path=None, show=False):
    """
    Visualise predictions against ground-truth labels for the latest chunk.

    For classification: a scatter plot where colour encodes whether each
    prediction is correct (green) or incorrect (red).

    Parameters
    ----------
    y_true    : array-like  True labels.
    y_pred    : array-like  Predicted labels.
    title     : str
    ax        : matplotlib.axes.Axes or None
    save_path : str or None
    show      : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = (y_true == y_pred)
    n = len(y_true)
    x = np.arange(n)

    fig, ax = _make_ax(ax, figsize=(10, 4))
    ax.scatter(x[correct], y_true[correct], color="#4CAF50", s=30,
               label="Correct", alpha=0.8, zorder=3)
    ax.scatter(x[~correct], y_true[~correct], color="#F44336", s=30,
               label="Wrong (true)", alpha=0.8, zorder=3)
    ax.scatter(x[~correct], y_pred[~correct], color="#FF9800", s=15,
               marker="x", label="Wrong (pred)", alpha=0.9, zorder=4)

    acc = correct.mean()
    ax.set_title(f"{title}  (acc={acc:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample index", fontsize=11)
    ax.set_ylabel("Class label", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix",
                          cmap="Blues", ax=None, save_path=None, show=False):
    """
    Render a confusion matrix as an annotated heatmap.

    Parameters
    ----------
    cm          : np.ndarray  Shape (n_classes, n_classes).
    class_names : list or None  Labels for rows/columns.
    title       : str
    cmap        : str   Matplotlib colormap name.
    ax          : matplotlib.axes.Axes or None
    save_path   : str or None
    show        : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = np.asarray(cm)
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    fig, ax = _make_ax(ax, figsize=(max(4, n * 1.4), max(3.5, n * 1.2)))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

def plot_feature_importance(importances, feature_names=None,
                            title="Feature Importances",
                            color="#7C4DFF",
                            ax=None, save_path=None, show=False):
    """
    Bar chart of feature importances (e.g. from a Random Forest).

    Parameters
    ----------
    importances   : array-like  One value per feature.
    feature_names : list or None  Feature labels.
    title         : str
    color         : str
    ax            : matplotlib.axes.Axes or None
    save_path     : str or None
    show          : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    importances = np.asarray(importances, dtype=float)
    n = len(importances)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n)]

    order = np.argsort(importances)[::-1]
    sorted_imp = importances[order]
    sorted_names = [feature_names[i] for i in order]

    fig, ax = _make_ax(ax, figsize=(max(6, n * 0.6), 4))
    bars = ax.bar(range(n), sorted_imp, color=color, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Importance", fontsize=11)
    ax.set_xlabel("Feature", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# plot_streaming_dashboard
# ---------------------------------------------------------------------------

def plot_streaming_dashboard(chunk_accuracies, cumulative_accuracies,
                             fit_times, y_true_last, y_pred_last,
                             cm=None, model_name="Model",
                             save_path=None, show=False):
    """
    Multi-panel streaming summary dashboard.

    Panels
    ------
    1. Per-chunk accuracy over time.
    2. Cumulative accuracy over time.
    3. Fit time per chunk (ms).
    4. Predictions vs ground truth on the last chunk.
    5. Confusion matrix (optional).

    Parameters
    ----------
    chunk_accuracies      : list  Per-chunk accuracy values.
    cumulative_accuracies : list  Running cumulative accuracy values.
    fit_times             : list  Per-chunk fit durations in seconds.
    y_true_last           : array-like  True labels from the last chunk.
    y_pred_last           : array-like  Predicted labels from the last chunk.
    cm                    : np.ndarray or None  Cumulative confusion matrix.
    model_name            : str   Title prefix.
    save_path             : str or None
    show                  : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = 5 if cm is not None else 4
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"{model_name} — Streaming Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: chunk accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    plot_metric_over_time(chunk_accuracies, title="Chunk Accuracy",
                          ylabel="Accuracy", color="#2196F3", ax=ax1)

    # Panel 2: cumulative accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    plot_metric_over_time(cumulative_accuracies, title="Cumulative Accuracy",
                          ylabel="Accuracy", color="#4CAF50", ax=ax2)

    # Panel 3: fit time
    ax3 = fig.add_subplot(gs[0, 2])
    fit_ms = [t * 1e3 for t in fit_times]
    plot_metric_over_time(fit_ms, title="Fit Time per Chunk",
                          ylabel="Time (ms)", color="#FF9800",
                          marker="^", ax=ax3)

    # Panel 4: predictions vs truth (last chunk)
    ax4 = fig.add_subplot(gs[1, 0:2])
    plot_predictions_vs_ground_truth(y_true_last, y_pred_last,
                                     title="Last Chunk: Predictions vs Truth",
                                     ax=ax4)

    # Panel 5: confusion matrix (optional)
    if cm is not None:
        ax5 = fig.add_subplot(gs[1, 2])
        plot_confusion_matrix(cm, title="Cumulative Confusion Matrix", ax=ax5)

    fig.tight_layout()
    return _save_or_show(fig, save_path, show)
