"""Label-free PCC-versus-sample-index plots for scoring and partitioning.

The plotting functions deliberately consume only raw PCC values.  They do not
invert the score, apply smoothing, read source/class labels, or calculate
evaluation metrics.
"""

from pathlib import Path

import numpy as np


def _plot(scores, output, title, lines=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif"],
        "mathtext.fontset": "stix",
    })
    values = np.asarray(scores, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("PCC plot requires finite, non-empty scores")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        x = np.arange(values.size)
        ax.scatter(x, values, s=3, alpha=0.4, color="tab:blue", label="Raw PCC")

        for line in lines or []:
            threshold = float(line["threshold"])
            ax.axhline(
                threshold,
                color=line.get("color", "green"),
                linestyle=line.get("linestyle", ":"),
                linewidth=1.5,
                alpha=0.9,
                label=line.get("label", f"threshold={threshold:.4f}"),
            )

        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel("PCC Value (higher = more reconstruction agreement)", fontsize=12)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(output, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output


def plot_score_distribution(scores, output, round_index):
    return _plot(
        scores,
        output,
        f"Round {int(round_index)}: raw PCC by sample index",
    )


def _plot_kde_strategy(scores, output, round_index, lines, kde_data):
    """Keep the PCC-vs-index scatter as the primary view and add KDE evidence.

    The KDE panel is deliberately derived from the same raw PCC values used by
    partition.py.  It never invents a threshold: when no stable valley was
    found, the panel says so explicitly instead of drawing a misleading line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(scores, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("PCC plot requires finite, non-empty scores")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_scatter, ax_density) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.65, 1.0]}
    )
    try:
        x = np.arange(values.size)
        ax_scatter.scatter(x, values, s=3, alpha=0.4, color="tab:blue", label="Raw PCC")
        for line in lines or []:
            threshold = float(line["threshold"])
            ax_scatter.axhline(
                threshold,
                color=line.get("color", "green"),
                linestyle=line.get("linestyle", ":"),
                linewidth=1.5,
                alpha=0.9,
                label=line.get("label", f"threshold={threshold:.4f}"),
            )
        ax_scatter.set_title(
            f"Round {int(round_index)}: KDE candidate PCC scatter",
            fontsize=15,
            fontweight="bold",
        )
        ax_scatter.set_xlabel("Sample Index", fontsize=12)
        ax_scatter.set_ylabel("PCC Value (higher = more reconstruction agreement)", fontsize=12)
        ax_scatter.set_ylim(-1.0, 1.0)
        ax_scatter.grid(True, alpha=0.3, linestyle="--")
        ax_scatter.legend(loc="lower left", fontsize=10, framealpha=0.9)

        # Density is plotted against PCC, not sample index.  This is the view
        # from which peaks and valleys can actually be inspected.
        kde_data = kde_data or {}
        scales = kde_data.get("bandwidth_scales") or []
        xs = np.linspace(float(values.min()), float(values.max()), 512)
        plotted = False
        try:
            from scipy.stats import gaussian_kde
            for item in scales:
                scale = item.get("bandwidth_scale")
                if scale is None:
                    continue
                kde = gaussian_kde(values, bw_method=lambda obj, q=float(scale): obj.scotts_factor() * q)
                density = kde(xs)
                is_reference = bool(
                    kde_data.get("reference")
                    and float(kde_data["reference"].get("bandwidth_scale", -1)) == float(scale)
                )
                ax_density.plot(
                    xs,
                    density,
                    color="tab:green" if is_reference else "#777777",
                    linewidth=2.0 if is_reference else 0.9,
                    alpha=0.9 if is_reference else 0.45,
                    label=f"bw={float(scale):.2f}" if is_reference else None,
                )
                plotted = True
        except Exception:
            # The statistical tool already reports scipy availability/errors;
            # keep the scatter artifact useful even if plotting KDE is unavailable.
            plotted = False

        reference = kde_data.get("reference")
        if reference and reference.get("threshold") is not None:
            threshold = float(reference["threshold"])
            ax_density.axvline(
                threshold, color="tab:green", linestyle=":", linewidth=1.5,
                label=f"KDE valley={threshold:.4f}",
            )
        if not reference:
            status = str(kde_data.get("status") or "no_stable_valley")
            message = (
                "KDE unavailable"
                if status == "unavailable"
                else "No stable KDE valley detected"
            )
            ax_density.text(
                0.5, 0.5, message,
                transform=ax_density.transAxes, ha="center", va="center",
                fontsize=11, color="#444444",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#aaaaaa"},
            )

        ax_density.set_title("KDE density evidence", fontsize=13, fontweight="bold")
        ax_density.set_xlabel("PCC Value", fontsize=12)
        ax_density.set_ylabel("Density", fontsize=12)
        ax_density.grid(True, alpha=0.3, linestyle="--")
        if plotted or reference:
            ax_density.legend(loc="best", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(output, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output


def plot_strategy_distribution(scores, output, round_index, strategy, lines, kde_data=None):
    if strategy == "kde":
        return _plot_kde_strategy(scores, output, round_index, lines, kde_data)
    labels = {
        "mean_std": "mean-k*std candidate thresholds",
        "kmeans": "K-Means (K=2) candidate threshold",
        "kde": "KDE candidate threshold",
    }
    return _plot(
        scores,
        output,
        f"Round {int(round_index)}: {labels.get(strategy, strategy)}",
        lines=lines,
    )
