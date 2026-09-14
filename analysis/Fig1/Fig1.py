
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
(REPO / "Fig1/output").mkdir(parents=True, exist_ok=True)

from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np


ROOT = REPO
S3 = (REPO / 'shared/data/ARG_annotations.tsv')
EXPORTS = (REPO / 'Fig1/output/exports')
SOURCE = (REPO / 'Fig1/output')
EXPORTS.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

OUT_BASENAME = "Fig1B_ARG_carriage_class_paired_pies"


def read_rep99_hits(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader if row["is_rep99"] == "yes"]


def mid_angle(wedge):
    return math.radians((wedge.theta1 + wedge.theta2) / 2.0)


def polar_xy(center, radius, angle):
    return center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)


def arc_points(center, radius, start_deg, end_deg, n=40):
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), n)
    return [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]


def annotate_wedge(ax, wedge, text, center, radius=1.0, color="black", fontsize=9):
    angle = mid_angle(wedge)
    x, y = polar_xy(center, radius * 0.62, angle)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, linespacing=1.0)


def annotate_outside(ax, wedge, text, center, radius=1.0, color="black", fontsize=8.5):
    angle = mid_angle(wedge)
    x0, y0 = polar_xy(center, radius * 0.92, angle)
    x1, y1 = polar_xy(center, radius * 1.18, angle)
    ha = "left" if math.cos(angle) >= 0 else "right"
    ax.plot([x0, x1], [y0, y1], color="#555555", lw=0.55)
    ax.text(x1, y1, text, ha=ha, va="center", fontsize=fontsize, color=color, linespacing=1.0)


def main():
    rep_hits = read_rep99_hits(S3)
    total_representatives = 4808
    arg_positive_genomes = len({row["genome_id"] for row in rep_hits})
    no_detected = total_representatives - arg_positive_genomes

    class_counts = {}
    for row in rep_hits:
        class_counts[row["ARG_class"]] = class_counts.get(row["ARG_class"], 0) + 1

    top_order = ["MLS", "tetracycline", "phenicol", "aminoglycoside"]
    right_counts = {name: class_counts.get(name, 0) for name in top_order}
    other_count = sum(count for name, count in class_counts.items() if name not in top_order)
    right_counts["Others"] = other_count

    with ((REPO / 'Fig1/output/Fig1B_ARG_carriage_class_paired_pies_source_data.tsv')).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["panel", "category", "count", "denominator", "percentage"])
        writer.writerow(["left", "ARG-positive", arg_positive_genomes, total_representatives, arg_positive_genomes / total_representatives * 100])
        writer.writerow(["left", "No detected ARG", no_detected, total_representatives, no_detected / total_representatives * 100])
        for category, count in right_counts.items():
            writer.writerow(["right", category, count, len(rep_hits), count / len(rep_hits) * 100])

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.0, 2.45))
    ax.set_aspect("equal")
    ax.axis("off")

    left_center = (-1.72, 0.0)
    right_center = (1.72, 0.0)
    pie_radius = 0.86

    left_values = [arg_positive_genomes, no_detected]
    left_labels = ["ARG-positive", "No detected ARG"]
    left_colors = ["#C33D2C", "#D6DCE0"]
    arg_positive_angle = 360 * arg_positive_genomes / total_representatives

    right_labels = ["MLS", "Tetracycline", "Phenicol", "Aminoglycoside", "Others"]
    right_values = [right_counts["MLS"], right_counts["tetracycline"], right_counts["phenicol"], right_counts["aminoglycoside"], right_counts["Others"]]
    right_colors = ["#C33D2C", "#E88824", "#F6E2A4", "#5E8DC7", "#808080"]

    left_wedges, _ = ax.pie(
        left_values,
        colors=left_colors,
        startangle=arg_positive_angle / 2,
        counterclock=False,
        radius=pie_radius,
        center=left_center,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )

    # Fig4A-style zoom connector: the left boundary follows the ARG-positive
    # wedge, and the right boundary expands to the full class-composition pie.
    pos_wedge = left_wedges[0]
    connector_radius = pie_radius * 1.01
    left_top = polar_xy(left_center, connector_radius, math.radians(pos_wedge.theta2))
    left_bottom = polar_xy(left_center, connector_radius, math.radians(pos_wedge.theta1))
    right_top = polar_xy(right_center, connector_radius, math.radians(90))
    right_bottom = polar_xy(right_center, connector_radius, math.radians(270))
    connector_points = [left_top, right_top, right_bottom, left_bottom, left_top]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(connector_points) - 1)
    connector_path = MplPath(connector_points, codes)
    connector_patch = PathPatch(connector_path, facecolor="none", edgecolor="none", zorder=0.15)
    ax.add_patch(connector_patch)

    xmin, xmax = min(left_top[0], left_bottom[0], right_top[0], right_bottom[0]), max(left_top[0], left_bottom[0], right_top[0], right_bottom[0])
    ymin, ymax = -1.04, 1.04
    grad = np.linspace(0, 1, 900)
    rgba = np.zeros((2, grad.size, 4))
    left_gray = np.array([0.93, 0.93, 0.93])
    right_gray = np.array([0.72, 0.72, 0.72])
    rgb = left_gray[:, None] * (1 - grad) + right_gray[:, None] * grad
    rgba[:, :, 0] = rgb[0]
    rgba[:, :, 1] = rgb[1]
    rgba[:, :, 2] = rgb[2]
    rgba[:, :, 3] = 0.64
    im = ax.imshow(rgba, extent=[xmin, xmax, ymin, ymax], origin="lower", interpolation="bicubic", zorder=0.12)
    im.set_clip_path(connector_patch)

    outline = PathPatch(connector_path, facecolor="none", edgecolor="#C7C7C7", lw=0.30, alpha=0.55, zorder=0.16)
    ax.add_patch(outline)

    right_wedges, _ = ax.pie(
        right_values,
        colors=right_colors,
        startangle=90,
        counterclock=False,
        radius=pie_radius,
        center=right_center,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )

    annotate_outside(
        ax,
        left_wedges[0],
        f"ARG-positive\n{arg_positive_genomes / total_representatives * 100:.1f}%",
        left_center,
        pie_radius,
        color="#222222",
        fontsize=7.4,
    )
    annotate_wedge(
        ax,
        left_wedges[1],
        f"No detected ARG\n{no_detected / total_representatives * 100:.1f}%",
        left_center,
        pie_radius,
        color="#333333",
        fontsize=8.4,
    )

    for wedge, label, count, color in zip(right_wedges, right_labels, right_values, ["white", "white", "black", "white", "white"]):
        pct = count / len(rep_hits) * 100
        text = f"{label}\n{pct:.1f}%"
        if pct < 10.0 or label == "Aminoglycoside":
            annotate_outside(ax, wedge, text, right_center, pie_radius, color="#222222", fontsize=7.2)
        else:
            annotate_wedge(ax, wedge, text, right_center, pie_radius, color=color, fontsize=8.1)

    ax.text(left_center[0], 1.07, "ARG carriage", ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    ax.text(right_center[0], 1.07, "ARG classes", ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    ax.text(left_center[0], -1.08, f"{arg_positive_genomes:,} / {total_representatives:,} representative genomes", ha="center", va="top", fontsize=7.0, color="#555555")
    ax.text(right_center[0], -1.08, f"{len(rep_hits):,} ARG hits", ha="center", va="top", fontsize=7.0, color="#555555")

    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-1.30, 1.30)

    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(EXPORTS / f"{OUT_BASENAME}.{suffix}", bbox_inches="tight", **kwargs)

    plt.close(fig)


if __name__ == "__main__":
    main()
