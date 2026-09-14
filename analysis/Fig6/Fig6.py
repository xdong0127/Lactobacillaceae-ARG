#!/usr/bin/env python3
"""Plot all 3,295 tested clusters with the 689 robust subset highlighted.

The y axis is the prevalence difference between lineages 8/10 and other
lineages. Horizontal position is deterministic jitter only. The 2,606 tested
clusters that did not enter the final robust set form the grey background.
"""

from __future__ import annotations

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
(REPO / "Fig6/output").mkdir(parents=True, exist_ok=True)


import os as _lab_os
from pathlib import Path as _LabPath


import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


ROOT = (REPO / 'Fig6/output')
FIGROOT = (REPO / 'Fig6/output')
ALL_INPUT = (
    (REPO / 'Fig6/data/Fig6C_lineage_association_volcano_source_data.tsv')
)
ROBUST_INPUT = (
    (REPO / 'Fig6/data/Fig6D_robust_effect_filtered_candidate_source_data.tsv')
)
OUTDIR = (REPO / 'Fig6/output')
OUTPUT_STEM = "Fig6C_all3295_robust689_effect_size_strip_refined"
OUTBASE = (REPO / 'Fig6/output/Fig6C_all3295_robust689_effect_size_strip_refined')

COLOR_OTHER = "#2166AC"
COLOR_L8L10 = "#D6601D"
COLOR_BACKGROUND = "#AFAFAF"
COLOR_ZERO = "#4F4F4F"
JITTER_SEED = 20260817
JITTER_SIGMA = 0.056
JITTER_CLIP = 0.152


def load_data() -> pd.DataFrame:
    all_df = pd.read_csv(ALL_INPUT, sep="\t")
    robust = pd.read_csv(ROBUST_INPUT, sep="\t")
    if len(all_df) != 3295 or all_df["panaroo_cluster"].duplicated().any():
        raise ValueError("Expected 3,295 unique tested Panaroo clusters")
    if len(robust) != 689 or robust["panaroo_cluster"].duplicated().any():
        raise ValueError("Expected 689 unique robust Panaroo clusters")

    robust_keep = robust[
        [
            "panaroo_cluster",
            "all_L8L10_minus_other",
            "multivariable_lineage_LRT_BH_FDR",
            "CMH_BH_FDR",
            "isolate_fisher_BH_FDR",
            "direction_consistent_all_isolate",
        ]
    ].copy()
    robust_keep["robust_candidate"] = True
    df = all_df.merge(
        robust_keep,
        on="panaroo_cluster",
        how="left",
        validate="one_to_one",
    )
    df["robust_candidate"] = df["robust_candidate"].fillna(False).astype(bool)
    if int(df["robust_candidate"].sum()) != 689:
        raise ValueError("The robust 689-cluster set is not an exact subset of the 3,295 tested clusters")

    both = df["robust_candidate"]
    if not np.allclose(
        df.loc[both, "L8L10_minus_other"],
        df.loc[both, "all_L8L10_minus_other"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Effect sizes disagree between the all-tested and robust source tables")

    locked = (
        (df.loc[both, "multivariable_lineage_LRT_BH_FDR"] < 0.05)
        & (df.loc[both, "CMH_BH_FDR"] < 0.05)
        & (df.loc[both, "isolate_fisher_BH_FDR"] < 0.05)
        & df.loc[both, "direction_consistent_all_isolate"].astype(str).str.lower().eq("true")
        & (df.loc[both, "L8L10_minus_other"].abs() >= 0.20 - 1e-12)
    )
    if not locked.all():
        raise ValueError("One or more highlighted rows no longer satisfy the locked robust definition")

    df = df.sort_values("panaroo_cluster", kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(JITTER_SEED)
    df["jitter_x"] = np.clip(
        rng.normal(0.0, JITTER_SIGMA, len(df)), -JITTER_CLIP, JITTER_CLIP
    )
    df["display_class"] = np.select(
        [
            df["robust_candidate"] & (df["L8L10_minus_other"] > 0),
            df["robust_candidate"] & (df["L8L10_minus_other"] < 0),
        ],
        ["Robust L8/L10 enriched", "Robust Other enriched"],
        default="Tested background",
    )
    return df


def make_figure(df: pd.DataFrame) -> None:
    width_mm = height_mm = 90
    fig, ax = plt.subplots(
        figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white"
    )

    background = df["display_class"].eq("Tested background")
    ax.scatter(
        df.loc[background, "jitter_x"],
        df.loc[background, "L8L10_minus_other"],
        s=5.5,
        c=COLOR_BACKGROUND,
        alpha=0.26,
        linewidths=0,
        zorder=1,
    )
    for group, color in [
        ("Robust Other enriched", COLOR_OTHER),
        ("Robust L8/L10 enriched", COLOR_L8L10),
    ]:
        mask = df["display_class"].eq(group)
        ax.scatter(
            df.loc[mask, "jitter_x"],
            df.loc[mask, "L8L10_minus_other"],
            s=9.5,
            c=color,
            alpha=0.68,
            linewidths=0,
            zorder=3,
        )

    for threshold in (-0.20, 0.20):
        ax.axhline(
            threshold,
            color="#A4A4A4",
            lw=0.55,
            ls=(0, (3, 3)),
            zorder=2,
        )
    ax.axhline(0, color="#5F5F5F", lw=0.65, zorder=2)
    ax.set_xlim(-0.27, 0.27)
    if float(df["L8L10_minus_other"].abs().max()) >= 0.90:
        raise ValueError("A fixed ±0.90 y range would clip one or more observations")
    ax.set_ylim(-0.90, 0.90)
    ax.set_yticks(np.arange(-0.8, 0.81, 0.2))
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel(
        "Prevalence difference\n(lineages 8/10 − other lineages)",
        fontsize=7.4,
        labelpad=5,
    )
    ax.tick_params(axis="y", labelsize=6.5, width=0.8, length=3)

    ax.text(
        0.5,
        1.005,
        "Robust L8/L10-enriched clusters (n = 352)",
        transform=ax.transAxes,
        color=COLOR_L8L10,
        fontsize=6.1,
        fontweight="bold",
        ha="center",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        0.5,
        -0.012,
        "Robust other-lineage-enriched clusters (n = 337)",
        transform=ax.transAxes,
        color=COLOR_OTHER,
        fontsize=6.1,
        fontweight="bold",
        ha="center",
        va="top",
        clip_on=False,
    )
    ax.text(
        0.98,
        0.525,
        "Tested background (n = 2,606)",
        transform=ax.transAxes,
        color="#6E6E6E",
        fontsize=5.2,
        ha="right",
        va="bottom",
    )
    threshold_transform = ax.get_yaxis_transform()
    ax.text(
        0.985,
        0.205,
        "+0.20",
        transform=threshold_transform,
        color="#808080",
        fontsize=5.0,
        ha="right",
        va="bottom",
    )
    ax.text(
        0.985,
        -0.205,
        "−0.20",
        transform=threshold_transform,
        color="#808080",
        fontsize=5.0,
        ha="right",
        va="top",
    )
    ax.text(
        0.5,
        -0.070,
        "Horizontal position represents jitter only",
        transform=ax.transAxes,
        color="#707070",
        fontsize=5.5,
        fontstyle="italic",
        ha="center",
        va="top",
        clip_on=False,
    )
    ax.text(
        -0.19,
        1.000,
        "c",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.subplots_adjust(left=0.22, right=0.97, top=0.94, bottom=0.13)
    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"facecolor": "white"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUTBASE.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


def write_outputs(df: pd.DataFrame) -> None:
    df.to_csv(
        (REPO / 'Fig6/output/Fig6C_all3295_robust689_effect_size_strip_refined_source_data.tsv'),
        sep="\t",
        index=False,
    )
    stats = {
        "status": "PASS",
        "figure_role": "all-tested one-dimensional effect-size strip with the final robust subset highlighted",
        "backend": "Python/matplotlib",
        "dimensions_mm": [90, 90],
        "canvas_aspect_ratio": "1:1",
        "n_tested_clusters": int(len(df)),
        "n_robust_clusters": int(df["robust_candidate"].sum()),
        "n_robust_L8L10_enriched": int(
            (df["display_class"] == "Robust L8/L10 enriched").sum()
        ),
        "n_robust_other_enriched": int(
            (df["display_class"] == "Robust Other enriched").sum()
        ),
        "n_tested_background": int((~df["robust_candidate"]).sum()),
        "n_direct_ARG_clusters_retained_as_ordinary_points": int(
            df["direct_ARG_feature"].astype(str).str.lower().eq("true").sum()
        ),
        "effect_variable": "L8L10_minus_other",
        "x_axis_meaning": "none; deterministic jitter only",
        "jitter_seed": JITTER_SEED,
        "jitter_sigma": JITTER_SIGMA,
        "jitter_clip": JITTER_CLIP,
        "y_axis_limits": [-0.90, 0.90],
        "effect_reference_lines": [-0.20, 0.20],
        "candidate_definition": (
            "Lineage LRT BH-FDR < 0.05 in presence ~ lineage_group + genome_type; "
            "genome-type-stratified CMH BH-FDR < 0.05; isolate-only Fisher BH-FDR < 0.05; "
            "same effect direction in all genomes and isolates; absolute prevalence difference >= 0.20."
        ),
        "interpretation_boundary": (
            "Coloured points are the subset passing every locked robustness criterion; grey points are tested "
            "clusters that did not enter that final set. Horizontal displacement has no biological meaning, "
            "and association does not establish a causal role in ARG acquisition or maintenance."
        ),
    }
    ((REPO / 'Fig6/output/Fig6C_all3295_robust689_effect_size_strip_refined_statistics.json')).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    caption = (
        "Fig. 6c | Robust bidirectional gene-cluster prevalence differences within the complete tested "
        "accessory-genome background. Each point represents one of 3,295 tested Panaroo clusters. The y axis "
        "shows the prevalence difference between lineages 8/10 (n = 157 genomes) and the remaining lineages "
        "(n = 261); positive values indicate higher prevalence in lineages 8/10 and negative values indicate "
        "higher prevalence in other lineages. Orange and blue points denote the 352 lineage-8/10-enriched and "
        "337 other-lineage-enriched clusters, respectively, that passed the lineage model, genome-type-stratified "
        "CMH, isolate-only sensitivity, direction-consistency and absolute-effect filters. The remaining 2,606 "
        "tested clusters are shown in grey. Dashed horizontal lines mark prevalence differences of ±0.20, "
        "the effect-size component of the final robustness filter. Horizontal positions are deterministic jitter only and have no "
        "quantitative meaning. Direct ARG-defining clusters are retained as ordinary observations. These "
        "associations describe lineage-linked genomic backgrounds and do not establish causal roles in ARG "
        "acquisition or maintenance."
    )
    ((REPO / 'Fig6/output/Fig6C_all3295_robust689_effect_size_strip_refined_caption.txt')).write_text(
        caption + "\n", encoding="utf-8"
    )


def main() -> None:
    df = load_data()
    write_outputs(df)
    make_figure(df)
    print(json.dumps({"status": "PASS", "output_base": str(OUTBASE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
