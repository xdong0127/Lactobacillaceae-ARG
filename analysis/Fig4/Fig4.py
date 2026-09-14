from __future__ import annotations

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
(REPO / "Fig4/output").mkdir(parents=True, exist_ok=True)


from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


HERE = (REPO / 'Fig4/Fig4.py')
OUT_DIR = (REPO / 'Fig4/output')
PROJECT_ROOT = REPO

SOURCE_FILE = (
    (REPO / 'Fig4/data/PTU_host_range_ARG_rate_all37_source_data.tsv')
)
ENRICHMENT_FILE = (
    (REPO / 'Fig4/data/PTU_enrichment_after_80_80_filter.tsv')
)

STEM = "Fig5A_37_PTU_ARG_distribution"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 6.4,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    }
)


INK = "#252A2D"
MUTED = "#6D7478"
GRID = "#D8DCDE"
SEPARATOR = "#AEB5B9"
LOLLIPOP = "#BBC1C4"
EDGE = "#465158"
ZERO = "#E4E7E8"

ARG_CMAP = LinearSegmentedColormap.from_list(
    "arg_rate_linear",
    [
        (0.000, "#E4E7E8"),
        (0.010, "#F4DFAC"),
        (0.100, "#EEC77E"),
        (0.250, "#E6A25E"),
        (0.500, "#D87550"),
        (0.750, "#C5534D"),
        (1.000, "#A93F47"),
    ],
)
ARG_NORM = Normalize(vmin=0, vmax=100)


def bubble_area(n: float) -> float:
    """Marker area in points squared; area is linear in plasmid count."""
    # Keep area proportional to PTU size while preventing the largest PTUs
    # from touching adjacent rows in the dense 37-row atlas.
    return 14.0 + 1.35 * float(n)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(SOURCE_FILE, sep="\t")
    enrichment = pd.read_csv(ENRICHMENT_FILE, sep="\t")
    enrichment = enrichment.loc[
        enrichment["analysis_unit"].eq("accession_used"),
        ["PTU", "BH_q", "enriched_BH_q_lt_0.05"],
    ].copy()

    data = data.merge(enrichment, on="PTU", how="left", validate="one_to_one")
    data["enriched_BH_q_lt_0.05"] = (
        data["enriched_BH_q_lt_0.05"].fillna(False).astype(bool)
    )
    data["bubble_area_points2"] = data["project_plasmid_count"].map(bubble_area)
    data["plot_x_position"] = data["host_range_grade_numeric"].astype(float)

    # Host-range grade is the primary order. Within each grade, ARG signal is
    # shown first, followed by larger PTUs, so the atlas is easy to scan.
    data = data.sort_values(
        [
            "host_range_grade_numeric",
            "ARG_positive_rate_pct_80_80",
            "project_plasmid_count",
            "PTU",
        ],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    data["display_order_top_to_bottom"] = np.arange(1, len(data) + 1)
    data["ARG_positive_over_total"] = (
        data["ARG_positive_plasmids_80_80"].astype(int).astype(str)
        + "/"
        + data["project_plasmid_count"].astype(int).astype(str)
    )
    return data


def add_grade_separators(ax: plt.Axes, data: pd.DataFrame) -> None:
    grades = data["host_range_grade_numeric"].to_numpy()
    for boundary in np.where(np.diff(grades) != 0)[0] + 0.5:
        ax.axhline(boundary, color=SEPARATOR, linewidth=0.65, zorder=0)


def plot(data: pd.DataFrame) -> None:
    # The taller but narrower canvas keeps rows distinct while reducing
    # uninformative horizontal whitespace. Legends occupy the naturally empty
    # upper-right portion of the grade matrix.
    fig = plt.figure(figsize=(4.60, 9.35), facecolor="white")
    ax = fig.add_axes([0.30, 0.075, 0.45, 0.865])

    y = np.arange(len(data))
    x = data["plot_x_position"].to_numpy()

    # Thin lollipop stems connect every PTU label/row to its host-range grade.
    # They sit below the bubbles and remain intentionally lighter than the
    # grade separators and marker outlines.
    ax.hlines(
        y=y,
        xmin=0.65,
        xmax=x,
        color=LOLLIPOP,
        linewidth=0.55,
        alpha=0.80,
        zorder=1,
    )

    scatter = ax.scatter(
        x,
        y,
        s=data["bubble_area_points2"],
        c=data["ARG_positive_rate_pct_80_80"],
        cmap=ARG_CMAP,
        norm=ARG_NORM,
        edgecolor=EDGE,
        linewidth=0.65,
        zorder=3,
    )

    add_grade_separators(ax, data)
    grade_positions = list(range(1, 7))
    for position in grade_positions:
        ax.axvline(
            position,
            color=GRID,
            linewidth=0.55,
            linestyle=(0, (2.2, 2.2)),
            zorder=0,
        )

    positive = data["ARG_positive_plasmids_80_80"].gt(0)
    for _, row in data.loc[positive].iterrows():
        ax.text(
            row["plot_x_position"] + 0.10,
            row["display_order_top_to_bottom"] - 1,
            row["ARG_positive_over_total"],
            ha="left",
            va="center",
            fontsize=5.8,
            color=INK,
            zorder=5,
        )

    labels = []
    for _, row in data.iterrows():
        label = row["PTU"]
        if row["enriched_BH_q_lt_0.05"]:
            label += "*"
        labels.append(label)

    ax.set_xlim(0.65, 6.45)
    ax.set_ylim(-0.75, len(data) - 0.25)
    ax.invert_yaxis()
    ax.set_xticks(grade_positions)
    ax.set_xticklabels(["I", "II", "III", "IV", "V", "VI"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("PTU host-range grade")
    ax.tick_params(
        axis="y",
        length=2.2,
        width=0.65,
        direction="out",
        pad=4,
        color=INK,
    )
    ax.tick_params(axis="x", length=3, width=0.7, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(INK)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_bounds(grade_positions[0], grade_positions[-1])

    for tick, enriched in zip(
        ax.get_yticklabels(), data["enriched_BH_q_lt_0.05"]
    ):
        tick.set_fontweight("bold" if enriched else "normal")
        tick.set_color(INK)

    # A separate legend column keeps the compressed categorical axis intact.
    legend_ax = fig.add_axes([0.77, 0.63, 0.21, 0.30])
    legend_ax.set_facecolor("white")
    legend_ax.patch.set_alpha(0.96)
    legend_ax.set_axis_off()

    legend_ax.text(
        0.02,
        0.98,
        "ARG-positive\nplasmids (%)",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color=INK,
    )
    cax = legend_ax.inset_axes([0.12, 0.63, 0.20, 0.25])
    cb = fig.colorbar(scatter, cax=cax)
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.ax.tick_params(labelsize=6, length=2.5, width=0.6)
    cb.outline.set_linewidth(0.6)

    legend_ax.text(
        0.02,
        0.54,
        "Project\nplasmids",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color=INK,
    )
    size_values = [1, 10, 50, 150]
    size_y = [0.42, 0.32, 0.20, 0.07]
    for value, ypos in zip(size_values, size_y):
        legend_ax.scatter(
            [0.24],
            [ypos],
            s=bubble_area(value),
            facecolor="white",
            edgecolor=EDGE,
            linewidth=0.65,
            transform=legend_ax.transAxes,
            clip_on=False,
        )
        legend_ax.text(
            0.62,
            ypos,
            str(value),
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.3,
            color=INK,
        )

    fig.text(
        0.77,
        0.595,
        "Bubble labels: ARG-positive / total\n* BH-adjusted Fisher q < 0.05",
        ha="left",
        va="top",
        fontsize=5.7,
        linespacing=1.35,
        color=MUTED,
    )

    fig.text(
        0.018,
        0.982,
        "A",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.30,
        0.982,
        "Assigned PTUs",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.30,
        0.961,
        "37 PTUs · n = 1,259 plasmids · ARG identity/coverage ≥80%",
        ha="left",
        va="top",
        fontsize=5.8,
        color=MUTED,
    )

    for extension, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(
            OUT_DIR / f"{STEM}{extension}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    fig.savefig(
        (REPO / 'Fig4/output/Fig5A_37_PTU_ARG_distribution_preview.png'),
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def write_source_and_qa(data: pd.DataFrame) -> None:
    data.to_csv((REPO / 'Fig4/output/Fig5A_37_PTU_ARG_distribution_source_data.tsv'), sep="\t", index=False)
    qa = pd.DataFrame(
        [
            ["PTU rows", len(data), 37, len(data) == 37],
            [
                "Assigned plasmids",
                int(data["project_plasmid_count"].sum()),
                1259,
                int(data["project_plasmid_count"].sum()) == 1259,
            ],
            [
                "ARG-positive assigned plasmids",
                int(data["ARG_positive_plasmids_80_80"].sum()),
                27,
                int(data["ARG_positive_plasmids_80_80"].sum()) == 27,
            ],
            [
                "ARG-bearing PTUs",
                int(data["ARG_positive_plasmids_80_80"].gt(0).sum()),
                10,
                int(data["ARG_positive_plasmids_80_80"].gt(0).sum()) == 10,
            ],
            [
                "BH-enriched PTUs",
                int(data["enriched_BH_q_lt_0.05"].sum()),
                5,
                int(data["enriched_BH_q_lt_0.05"].sum()) == 5,
            ],
            [
                "Host grades ordered top-to-bottom",
                bool(data["host_range_grade_numeric"].is_monotonic_increasing),
                True,
                bool(data["host_range_grade_numeric"].is_monotonic_increasing),
            ],
        ],
        columns=["check", "observed", "expected", "pass"],
    )
    qa.to_csv((REPO / 'Fig4/output/Fig5A_37_PTU_ARG_distribution_QA.tsv'), sep="\t", index=False)
    if not qa["pass"].all():
        raise RuntimeError(qa.to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    write_source_and_qa(data)
    plot(data)


if __name__ == "__main__":
    main()
