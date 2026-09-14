# Lactobacillaceae ARG: selected figure code

Selected analyses and plotting examples, organised as **one code file per figure**. This release contains 7 figure files and 14 processed input tables under `analysis/`.

| Code file | Included analyses and plots |
|---|---|
| [Fig1.py](analysis/Fig1/Fig1.py) | Fig1C: ARG carriage and resistance-class pies |
| [Fig2.Rmd](analysis/Fig2/Fig2.Rmd) | Taxonomic statistics and Fig2A genus prevalence |
| [Fig3.Rmd](analysis/Fig3/Fig3.Rmd) | Fig3C network tables, Fig3E cluster MGE comparison and Fig3F species statistics |
| [Fig4.py](analysis/Fig4/Fig4.py) | Fig4B: PTU ARG distribution from summary inputs |
| [Fig5.Rmd](analysis/Fig5/Fig5.Rmd) | Source enrichment, within-species comparisons and Fig5D heatmap |
| [Fig6.py](analysis/Fig6/Fig6.py) | Fig6C: effect-size strip from precomputed results |
| [FigS2.py](analysis/FigS2/FigS2.py) | FigS2B: source-balanced resampling |

Figure-specific inputs are in that figure's `data/` folder. Three shared inputs are in `analysis/shared/data/`. Historical input/output basenames may retain earlier figure numbers; code filenames use the final numbering.

## Running

Use an existing environment with the packages needed by the chosen section:

- **Python:** matplotlib, numpy, pandas, scipy, statsmodels.
- **R:** broom, dplyr, emmeans, forcats, ggplot2, knitr, openxlsx, ragg, readr, rmarkdown, scales, svglite, tibble, tidyr, vegan.

These are dependency names, not a historical version lockfile. No installation or environment-creation steps are included.

```bash
python analysis/Fig1/Fig1.py
python analysis/Fig4/Fig4.py
```

Open an `.Rmd` in RStudio to run its sections, or render it from the repository root:

```r
rmarkdown::render("analysis/Fig3/Fig3.Rmd")
```

Mixed-language notebooks use knitr's external Python engine. Python is found on PATH, or selected with the `PYTHON` environment variable. Input paths resolve from the code file's location. Generated files go to figure-local `output/` folders and are excluded from Git. Fig2's permutation analysis can take substantially longer than its plotting section. Fig6 does not refit the underlying models.

## Scope and checks

This is a selective release, not a complete reproduction package for all manuscript results or final layouts. Trees, iTOL annotations, phylogenetic distances, full gene presence/absence matrices, genomic sequence windows, upstream histories and omitted plotting/model-fitting workflows are not included. Country, continent and related geographic summaries have been removed. Source categories remain as analytical variables. Reduced schemas retain fields needed by the included sections; they do not establish complete anonymisation.

All Python code and R chunks passed syntax checks. The four Python files ran in a relocated copy. Fig3 and Fig5 executed all included R/Python sections. Fig2's plotting section ran; its expensive taxonomic-statistics section was parsed but not rerun. Retained input cells and row order were checked against the original tables. Final HTML styling was not reviewed.

Numbering follows the final Fig3 PDF (E: cluster MGE; F: species prevalence), which differs from its supplied captions. Fig5 distinguishes all 62 significant associations from the 57 shared-species-only associations. Fig4 uses the supplied one-sided Fisher enrichment table; the manuscript caption states two-sided. These documented differences were not silently changed during code curation.
