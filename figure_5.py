from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import omicverse as ov
from plot_style import (
    clean_axis,
    embedding_dataframe,
    ensure_outdir,
    first_obsm_key,
    first_present,
    get_gene_vector,
    panel_label,
    read_table,
    save_panel,
    set_nature_style,
)

def spatial_xy(adata) -> pd.DataFrame:
    xy = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"], index=adata.obs_names)
    return xy.join(adata.obs)

# ---- Fig. 5c | SLPI ligand interactions from NMF4 cells to macrophages ----
cci = read_table(CELLCOM_DIR / "cci_table_kidney.csv")
ligand_col = first_present(cci.columns, ["ligand", "ligand_complex", "gene"])
source_col = first_present(cci.columns, ["source", "sender", "source_cell"])
target_col = first_present(cci.columns, ["target", "receiver", "target_cell"])
score_col = first_present(cci.columns, ["score", "magnitude_rank", "interaction_score", "weight"])
slpi_cci = cci.loc[cci[ligand_col].astype(str).str.contains("SLPI", case=False, regex=False)]
slpi_cci = slpi_cci.loc[slpi_cci[source_col].astype(str).str.contains("NMF4", case=False, regex=False)]
fig, ax = plt.subplots(figsize=(3.0, 2.2))
sns.barplot(data=slpi_cci.sort_values(score_col, ascending=False).head(12), x=score_col, y=target_col, color="#D64A3A", ax=ax)
ax.set(xlabel="SLPI interaction score", ylabel="")
clean_axis(ax)
panel_label(ax, "h")
save_panel(fig, OUTDIR, "fig5h_slpi_nmf4_to_macrophage_interactions")
plt.close(fig)

# ---- Fig. 5g | Macrophage subtypes resolved by single-cell RNA-seq ----
macro = sc.read_h5ad(MACRO_DIR / "Macro_annonated.h5ad")
ov.pl.embedding(macro,basis=' MDE',color='sub_celltype')

# ---- Fig. 5k | TREM2 expression identifies the macrophage state of interest ----
ov.pl.barplot(macro,keys='TREM2',groupby='group')

# ---- Fig. 5i | Spatial correlation between NMF4 and TREM2 macrophage score ----
corr = read_table(ST_DIR / "NMF_VS_TREM2.csv")
nmf_col = first_present(corr.columns, ["NMF_state",])
r_col = first_present(corr.columns, ["correlation", "r", "R", "pearson_r"])
fig, ax = plt.subplots(figsize=(2.4, 2.0))
sns.barplot(data=corr, x=nmf_col, y=r_col, color="#D64A3A", ax=ax)
ax.axhline(0, color="black", linewidth=0.5)
ax.set(xlabel="", ylabel="Correlation with TREM2+ macrophage score")
ax.tick_params(axis="x", rotation=45)
clean_axis(ax)
panel_label(ax, "e")
save_panel(fig, OUTDIR, "fig5e_st_nmf_trem2_correlation")
plt.close(fig)

st = sc.read_h5ad(ST_DIR / "New" / "Data2" / "ST_ALL.h5ad")
df_sp = spatial_xy(st)
score_cols = [c for c in ["NMF4_score", "TREM2+ Macro_score", "TREM2_score", "TREM2"] if c in df_sp.columns]
fig, axes = plt.subplots(1, len(score_cols), figsize=(2.4 * len(score_cols), 2.3))
if len(score_cols) == 1:
    axes = [axes]
for ax, score in zip(axes, score_cols):
    sns.scatterplot(data=df_sp, x="x", y="y", hue=score, palette="magma", s=4, linewidth=0, ax=ax)
    ax.set(xticks=[], yticks=[], xlabel="", ylabel="", title=score)
    ax.set_aspect("equal")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
panel_label(axes[0], "f")
save_panel(fig, OUTDIR, "fig5f_spatial_nmf4_trem2_maps")
plt.close(fig)