import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _save_or_show(fig, save_path, show):
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig

def _make_ax(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax

def plot_metric_over_time(metric_values, title="Metric over Time",
                          ylabel="Value", xlabel="Chunk",
                          color="#2196F3", marker="o",
                          ax=None, save_path=None, show=False):
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

def compare_models(metric1, metric2, labels=("Model A", "Model B"),
                   title="Model Comparison", ylabel="Accuracy",
                   xlabel="Chunk", colors=("#2196F3", "#F44336"),
                   ax=None, save_path=None, show=False):
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

def plot_predictions_vs_ground_truth(y_true, y_pred,
                                     title="Predictions vs Ground Truth",
                                     ax=None, save_path=None, show=False):
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

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix",
                          cmap="Blues", ax=None, save_path=None, show=False):
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

def plot_feature_importance(importances, feature_names=None,
                            title="Feature Importances",
                            color="#7C4DFF",
                            ax=None, save_path=None, show=False):
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

def plot_streaming_dashboard(chunk_accuracies, cumulative_accuracies,
                             fit_times, y_true_last, y_pred_last,
                             cm=None, model_name="Model",
                             save_path=None, show=False):
    n_panels = 5 if cm is not None else 4
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"{model_name} — Streaming Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_metric_over_time(chunk_accuracies, title="Chunk Accuracy",
                          ylabel="Accuracy", color="#2196F3", ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_metric_over_time(cumulative_accuracies, title="Cumulative Accuracy",
                          ylabel="Accuracy", color="#4CAF50", ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    fit_ms = [t * 1e3 for t in fit_times]
    plot_metric_over_time(fit_ms, title="Fit Time per Chunk",
                          ylabel="Time (ms)", color="#FF9800",
                          marker="^", ax=ax3)

    ax4 = fig.add_subplot(gs[1, 0:2])
    plot_predictions_vs_ground_truth(y_true_last, y_pred_last,
                                     title="Last Chunk: Predictions vs Truth",
                                     ax=ax4)

    if cm is not None:
        ax5 = fig.add_subplot(gs[1, 2])
        plot_confusion_matrix(cm, title="Cumulative Confusion Matrix", ax=ax5)

    fig.tight_layout()
    return _save_or_show(fig, save_path, show)
