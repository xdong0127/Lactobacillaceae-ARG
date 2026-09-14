from __future__ import annotations

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
(REPO / "FigS2/output").mkdir(parents=True, exist_ok=True)


from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = (REPO / 'FigS2/FigS2.py')




PROJECT_ROOT = REPO
OUT_ROOT = (REPO / 'FigS2/output')
EXPORT_DIR = (REPO / 'FigS2/output/exports')
SOURCE_DIR = (REPO / 'FigS2/output')
STATS_DIR = (REPO / 'FigS2/output')
for directory in (EXPORT_DIR, SOURCE_DIR, STATS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


SOURCE_ORDER = ["animal", "human", "other", "food"]
SOURCE_LABELS = {
    "animal": "Animal",
    "human": "Human",
    "other": "Other",
    "food": "Food",
}
SOURCE_COLORS = {
    "animal": "#B48770",
    "human": "#6F879F",
    "food": "#7BAA8E",
    "other": "#A9AFB6",
}
MINIMUM_PER_SOURCE = 5
N_ITERATIONS = 1000
RANDOM_SEED = 20260722


def find_one(base: Path, pattern: str) -> Path:
    matches = list(base.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match for {pattern!r}, found {len(matches)}.")
    return matches[0]


def clean_category(series: pd.Series, fallback: str) -> pd.Series:
    cleaned = series.astype("string").fillna(fallback).str.strip()
    return cleaned.mask(cleaned.eq(""), fallback).astype(str)


def load_genome_level_table() -> pd.DataFrame:
    s1_path = REPO / "shared/data/MQ_genomes.tsv"
    s3_path = REPO / "shared/data/ARG_annotations.tsv"

    s1 = pd.read_csv(
        s1_path,
        sep="\t",
        usecols=["genome_id", "genus", "species", "source_type"],
        dtype=str,
    )
    s3 = pd.read_csv(s3_path, sep="\t", usecols=["genome_id"], dtype=str)

    arg_positive_ids = set(s3["genome_id"].dropna().unique())
    mq = s1.assign(
        genus=clean_category(s1["genus"], "Unclassified genus"),
        species=clean_category(s1["species"], "Unclassified at species level"),
        source_type=clean_category(s1["source_type"], "other").str.lower(),
        ARG_positive=s1["genome_id"].isin(arg_positive_ids),
    )
    mq = mq[mq["source_type"].isin(SOURCE_ORDER)].copy()
    return mq


def build_resampling_units(mq: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    species_level = mq[mq["species"] != "Unclassified at species level"].copy()

    counts = (
        species_level.groupby(["genus", "species", "source_type"], observed=False)
        .agg(source_n=("genome_id", "size"), arg_positive_n=("ARG_positive", "sum"))
        .reset_index()
    )
    retained_strata = counts[counts["source_n"] >= MINIMUM_PER_SOURCE].copy()
    eligible_species = (
        retained_strata.groupby("species", observed=False)["source_type"]
        .nunique()
        .loc[lambda x: x >= 2]
        .index
    )
    retained_strata = retained_strata[retained_strata["species"].isin(eligible_species)].copy()

    eligible_keys = set(zip(retained_strata["species"], retained_strata["source_type"]))
    eligible_rows = species_level[
        species_level[["species", "source_type"]].apply(tuple, axis=1).isin(eligible_keys)
    ].copy()

    units: dict[str, dict[str, np.ndarray]] = {}
    for (species, source), frame in eligible_rows.groupby(["species", "source_type"], observed=False):
        units.setdefault(species, {})[source] = frame["ARG_positive"].to_numpy(dtype=bool)

    units = {
        species: source_arrays
        for species, source_arrays in units.items()
        if len(source_arrays) >= 2
    }

    return units, retained_strata


def run_balanced_resampling(units: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for iteration in range(1, N_ITERATIONS + 1):
        positive_by_source = defaultdict(int)
        total_by_source = defaultdict(int)

        for source_arrays in units.values():
            depth = min(len(values) for values in source_arrays.values())
            for source, values in source_arrays.items():
                sampled_indices = rng.choice(len(values), size=depth, replace=False)
                sampled = values[sampled_indices]
                positive_by_source[source] += int(sampled.sum())
                total_by_source[source] += int(depth)

        rates = {
            source: positive_by_source[source] / total_by_source[source]
            for source in SOURCE_ORDER
            if total_by_source[source] > 0
        }
        maximum = max(rates.values())
        highest_sources = sorted([source for source, rate in rates.items() if np.isclose(rate, maximum)])

        for source in SOURCE_ORDER:
            if total_by_source[source] == 0:
                continue
            rows.append(
                {
                    "iteration": iteration,
                    "source_type": source,
                    "sampled_genomes": total_by_source[source],
                    "ARG_positive_genomes": positive_by_source[source],
                    "ARG_positive_rate": rates[source],
                    "ARG_positive_pct": rates[source] * 100,
                    "highest_source_this_iteration": ";".join(highest_sources),
                    "animal_in_highest_source": "animal" in highest_sources,
                }
            )

    return pd.DataFrame(rows)


def summarize(
    mq: pd.DataFrame,
    retained_strata: pd.DataFrame,
    iterations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = (
        mq.groupby("source_type", observed=False)
        .agg(total_MQ_genomes=("genome_id", "size"), ARG_positive_MQ_genomes=("ARG_positive", "sum"))
        .reindex(SOURCE_ORDER)
        .reset_index()
    )
    raw["raw_MQ_rate"] = raw["ARG_positive_MQ_genomes"] / raw["total_MQ_genomes"]
    raw["raw_MQ_pct"] = raw["raw_MQ_rate"] * 100

    eligible_raw = (
        retained_strata.groupby("source_type", observed=False)
        .agg(eligible_MQ_genomes=("source_n", "sum"), eligible_ARG_positive_genomes=("arg_positive_n", "sum"))
        .reindex(SOURCE_ORDER)
        .reset_index()
    )
    eligible_raw["eligible_raw_rate"] = (
        eligible_raw["eligible_ARG_positive_genomes"] / eligible_raw["eligible_MQ_genomes"]
    )
    eligible_raw["eligible_raw_pct"] = eligible_raw["eligible_raw_rate"] * 100

    balanced = (
        iterations.groupby("source_type", observed=False)
        .agg(
            balanced_median_pct=("ARG_positive_pct", "median"),
            balanced_q025_pct=("ARG_positive_pct", lambda x: np.quantile(x, 0.025)),
            balanced_q975_pct=("ARG_positive_pct", lambda x: np.quantile(x, 0.975)),
            balanced_mean_sampled_genomes=("sampled_genomes", "mean"),
            balanced_min_sampled_genomes=("sampled_genomes", "min"),
            balanced_max_sampled_genomes=("sampled_genomes", "max"),
        )
        .reindex(SOURCE_ORDER)
        .reset_index()
    )
    summary = raw.merge(eligible_raw, on="source_type").merge(balanced, on="source_type")
    summary["source_label"] = summary["source_type"].map(SOURCE_LABELS)

    iteration_winners = (
        iterations[["iteration", "highest_source_this_iteration", "animal_in_highest_source"]]
        .drop_duplicates()
        .sort_values("iteration")
    )
    winner_counts = Counter(iteration_winners["highest_source_this_iteration"])
    manifest = pd.DataFrame(
        [
            ["MQ genomes", len(mq)],
            ["ARG-positive MQ genomes", int(mq["ARG_positive"].sum())],
            ["Minimum genomes per retained species-source stratum", MINIMUM_PER_SOURCE],
            ["Eligible multi-source species", len(set(retained_strata["species"]))],
            ["Retained species-source strata", len(retained_strata)],
            ["Resampling iterations", N_ITERATIONS],
            ["Random seed", RANDOM_SEED],
            ["Animal highest-source retention", iteration_winners["animal_in_highest_source"].mean()],
            ["Most frequent highest source", winner_counts.most_common(1)[0][0]],
        ],
        columns=["item", "value"],
    )
    return summary, manifest


def set_matplotlib_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#26323a",
            "axes.labelcolor": "#26323a",
            "xtick.color": "#26323a",
            "ytick.color": "#26323a",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
        }
    )


def plot_summary(summary: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    set_matplotlib_style()
    summary = summary.set_index("source_type").loc[SOURCE_ORDER].reset_index()
    y_positions = np.arange(len(summary))[::-1]

    fig, ax = plt.subplots(figsize=(4.7, 2.65), constrained_layout=False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    overall_rate = manifest.loc[manifest["item"] == "ARG-positive MQ genomes", "value"].iloc[0]
    overall_rate = int(overall_rate) / int(manifest.loc[manifest["item"] == "MQ genomes", "value"].iloc[0]) * 100
    ax.axvline(overall_rate, color="#AEB7BF", lw=0.9, linestyle=(0, (3, 3)), zorder=0)
    ax.text(
        overall_rate + 0.25,
        y_positions[0] + 0.58,
        f"overall {overall_rate:.1f}%",
        ha="left",
        va="center",
        color="#69747f",
        fontsize=7.5,
    )

    for _, row in summary.iterrows():
        y = y_positions[SOURCE_ORDER.index(row["source_type"])]
        color = SOURCE_COLORS[row["source_type"]]
        ax.hlines(
            y,
            row["balanced_q025_pct"],
            row["balanced_q975_pct"],
            color=color,
            lw=2.1,
            alpha=0.62,
            zorder=2,
        )
        ax.scatter(
            row["balanced_median_pct"],
            y,
            s=53,
            facecolor=color,
            edgecolor="#26323a",
            linewidth=0.7,
            zorder=4,
        )
        ax.text(
            35.4,
            y,
            f"{row['balanced_median_pct']:.1f}% [{row['balanced_q025_pct']:.1f}, {row['balanced_q975_pct']:.1f}]",
            ha="right",
            va="center",
            color="#26323a",
            fontsize=6.2,
        )

    animal_retention = float(
        manifest.loc[manifest["item"] == "Animal highest-source retention", "value"].iloc[0]
    )
    eligible_species = int(manifest.loc[manifest["item"] == "Eligible multi-source species", "value"].iloc[0])

    ax.set_yticks(y_positions)
    ax.set_yticklabels([SOURCE_LABELS[source] for source in SOURCE_ORDER], fontsize=9)
    ax.set_xlim(0, 36.0)
    ax.set_ylim(-0.65, len(summary) - 0.35)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("ARG-positive genomes (%)", labelpad=5)
    ax.set_ylabel("Source", labelpad=18)
    ax.set_title(
        "Balanced source-level ARG carriage",
        loc="left",
        fontsize=7.8,
        fontweight="bold",
        color="#26323a",
        pad=8,
    )
    ax.text(
        35.4,
        len(summary) - 0.03,
        "Balanced % [95% interval]",
        ha="right",
        va="bottom",
        color="#69747f",
        fontsize=5.8,
    )
    ax.text(
        18.7,
        2.62,
        f"animal highest in {animal_retention * 100:.0f}% of resamplings",
        ha="left",
        va="center",
        color="#69747f",
        fontsize=5.8,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("#59636C")
    ax.spines["bottom"].set_color("#59636C")
    ax.tick_params(axis="y", length=3, width=0.7, pad=4)
    ax.tick_params(axis="x", length=3, width=0.7)
    ax.grid(axis="x", color="#ECEFF2", lw=0.7)
    ax.grid(axis="y", visible=False)

    balanced_handle = ax.scatter([], [], s=53, facecolor="#B48770", edgecolor="#26323a", linewidth=0.7)
    interval_handle = mpl.lines.Line2D([0], [0], color="#B48770", lw=2.1, alpha=0.62)
    ax.legend(
        [balanced_handle, interval_handle],
        ["Balanced median", "95% interval"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.29),
        ncol=2,
        fontsize=6.1,
        handlelength=1.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.32)

    out_base = (REPO / 'FigS2/output/FigS3a_MQ_balanced_within_species_resampling')
    for ext in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_base.with_suffix(".png")


def main() -> None:
    mq = load_genome_level_table()
    units, retained_strata = build_resampling_units(mq)
    iterations = run_balanced_resampling(units)
    summary, manifest = summarize(mq, retained_strata, iterations)

    iterations.to_csv(
        (REPO / 'FigS2/output/FigS3a_balanced_source_resampling_iterations.tsv'),
        sep="\t",
        index=False,
    )
    summary.to_csv(
        (REPO / 'FigS2/output/FigS3a_balanced_source_resampling_summary.tsv'),
        sep="\t",
        index=False,
    )
    retained_strata.to_csv(
        (REPO / 'FigS2/output/FigS3a_retained_species_source_strata.tsv'),
        sep="\t",
        index=False,
    )
    manifest.to_csv(
        (REPO / 'FigS2/output/FigS3a_balanced_source_resampling_manifest.tsv'),
        sep="\t",
        index=False,
    )

    preview_path = plot_summary(summary, manifest)

    print("Fig. S3a balanced resampling completed.")
    print(f"Eligible multi-source species: {len(set(retained_strata['species']))}")
    print(f"Retained species-source strata: {len(retained_strata)}")
    animal_retention = float(
        manifest.loc[manifest["item"] == "Animal highest-source retention", "value"].iloc[0]
    )
    print(f"Animal highest-source retention: {animal_retention * 100:.1f}%")
    print(summary[[
        "source_type",
        "raw_MQ_pct",
        "eligible_raw_pct",
        "balanced_median_pct",
        "balanced_q025_pct",
        "balanced_q975_pct",
    ]].to_string(index=False))
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
