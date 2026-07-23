from __future__ import annotations

import os
from pathlib import Path
import omicverse as ov
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from plot_style import (
    MAJOR_CELL_COLORS,
    NMF_COLORS,
    clean_axis,
    embedding_dataframe,
    ensure_outdir,
    first_obsm_key,
    first_present,
    panel_label,
    read_table,
    save_panel,
    set_nature_style,
    zscore_frame,
)


MARKER_GENES = {
    "Cancer cell": ["EPCAM", "KRT8", "KRT18", "CA9"],
    "T cell": ["CD3D", "CD3E", "CD8A", "IL7R"],
    "Myeloid": ["LYZ", "C1QA", "CD68", "FCGR3A"],
    "B cell": ["MS4A1", "CD79A"],
    "Fibroblast": ["COL1A1", "DCN"],
    "Endothelial": ["PECAM1", "VWF"],
}


def spatial_xy(adata) -> pd.DataFrame:
    xy = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"], index=adata.obs_names)
    return xy.join(adata.obs)


# ---- Fig. 1c | Integrated RCC single-cell atlas by major compartment ----
# Input object created from Data_after_QC.h5ad after concatenating tumor, normal,
# control and treatment samples.
adata = sc.read_h5ad(SC_DIR / "1.QC" / "Data_after_QC.h5ad")
cell_col = first_present(adata.obs.columns, ["major_celltype", "Major_celltype", "cell_type", "Celltype", "annotation","disease"])
ov.pl.embedding(adata,color=cell_col)


# ---- Fig. 1d | Canonical marker genes used for major-cell annotation ----
dot = sc.pl.dotplot(
    adata,
    MARKER_GENES,
    groupby=cell_col,
    standard_scale="var",
    dendrogram=False,
    show=False,
    return_fig=True,
)
dot.savefig(OUTDIR / "fig1c_major_celltype_marker_dotplot.pdf")


# ---- Fig. 1f | Cell-type proportions by patient sample ----
sample_prop = (
    adata.obs.groupby([sample_col, cell_col], observed=True)
    .size()
    .rename("n_cells")
    .reset_index()
)
sample_prop["fraction"] = sample_prop["n_cells"] / sample_prop.groupby(sample_col)["n_cells"].transform("sum")
fig, ax = plt.subplots(figsize=(5.0, 2.0))
sns.barplot(data=sample_prop, x=sample_col, y="fraction", hue=cell_col, palette=MAJOR_CELL_COLORS, ax=ax)
ax.set(xlabel="", ylabel="Cell fraction")
ax.tick_params(axis="x", rotation=90)
ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
clean_axis(ax)
panel_label(ax, "d")
save_panel(fig, OUTDIR, "fig1d_sample_level_cell_proportions")
plt.close(fig)


# ---- Fig. 1g | Cell-type composition by ICB response group ----
response_col = first_present(adata.obs.columns, ["Response", "response", "ICB_response", "treatment_response"])
response_prop = (
    adata.obs.groupby([response_col, sample_col, cell_col], observed=True)
    .size()
    .rename("n_cells")
    .reset_index()
)
response_prop["fraction"] = response_prop["n_cells"] / response_prop.groupby([response_col, sample_col])["n_cells"].transform("sum")
fig, ax = plt.subplots(figsize=(3.0, 2.2))
sns.boxplot(data=response_prop, x=response_col, y="fraction", hue=cell_col, fliersize=0, linewidth=0.5, ax=ax)
sns.stripplot(data=response_prop, x=response_col, y="fraction", hue=cell_col, dodge=True, size=1.2, linewidth=0, alpha=0.45, ax=ax)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[: len(set(labels))], labels[: len(set(labels))], frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set(xlabel="", ylabel="Cell fraction")
clean_axis(ax)
panel_label(ax, "e")
save_panel(fig, OUTDIR, "fig1g_cell_proportions_by_icb_response")
plt.close(fig)


# ---- Fig. 1l | Spatial immune-hot and immune-cold regions ----
st_files = sorted((ST_DIR / "New" / "Data2").glob("*.h5ad"))
st = sc.read_h5ad(st_files[0])
region_col = first_present(st.obs.columns, ["Region", "immune_region", "region"])
df_sp = spatial_xy(st)
fig, ax = plt.subplots(figsize=(2.4, 2.4))
sns.scatterplot(data=df_sp, x="x", y="y", hue=region_col, s=4, linewidth=0, ax=ax)
ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
ax.set_aspect("equal")
ax.legend(frameon=False, markerscale=2.5, bbox_to_anchor=(1.02, 1), loc="upper left")
panel_label(ax, "f")
save_panel(fig, OUTDIR, "fig1l_spatial_immune_regions")
plt.close(fig)


# ---- Fig. 1m | Immune-region pathway scores in spatial transcriptomics ----
region_summary = read_table(ST_DIR / "ST_region_mean_summary.csv")
score_cols = [c for c in region_summary.columns if c.endswith("_score") or c in ["M2_score", "CD4 Treg_score"]]
plot_df = region_summary.melt(id_vars=[region_col], value_vars=score_cols, var_name="signature", value_name="score")
fig, ax = plt.subplots(figsize=(3.1, 2.2))
sns.boxplot(data=plot_df, x=region_col, y="score", hue="signature", fliersize=0, linewidth=0.5, ax=ax)
ax.set(xlabel="", ylabel="Scaled score")
ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
clean_axis(ax)
panel_label(ax, "g")
save_panel(fig, OUTDIR, "fig1m_region_pathway_scores")
plt.close(fig)


# ---- Fig. 1n | Enrichment analysis of immune-hot regions ----
enrich = read_table(ST_DIR / "GO_depathway.csv")
term_col = first_present(enrich.columns, ["Term", "Description", "pathway", "Name"])
fdr_col = first_present(enrich.columns, ["Adjusted P-value", "adjusted_pvalue", "p.adjust", "FDR"])
enrich = enrich.sort_values(fdr_col).head(15)
enrich["neg_log10_fdr"] = -np.log10(pd.Series(enrich[fdr_col]).astype(float).clip(lower=1e-300))
fig, ax = plt.subplots(figsize=(3.2, 2.6))
sns.barplot(data=enrich, x="neg_log10_fdr", y=term_col, color="#4E79A7", ax=ax)
ax.set(xlabel="-log10(FDR)", ylabel="")
clean_axis(ax)
panel_label(ax, "h")
save_panel(fig, OUTDIR, "fig1n_immune_region_enrichment")
plt.close(fig)


# ---- Fig. 1o | Transcriptional variability across major cell compartments ----
gini_path = SC_DIR / "8.Cellcom" / "present" / "gini_df.csv"
if gini_path.exists():
    variability = read_table(gini_path)
else:
    variability = adata.obs[[cell_col]].copy()
    cell_mean = adata.X.mean(axis=1)
    variability["variability"] = cell_mean.A1 if hasattr(cell_mean, "A1") else cell_mean
fig, ax = plt.subplots(figsize=(2.8, 2.0))
x_col = first_present(variability.columns, [cell_col, "cell_type", "Celltype", "group"])
y_col = first_present(variability.columns, ["gini", "Gini", "ROGUE", "variability", "variance"])
sns.boxplot(data=variability, x=x_col, y=y_col, fliersize=0, linewidth=0.5, ax=ax)
ax.tick_params(axis="x", rotation=45)
ax.set(xlabel="", ylabel="Variability score")
clean_axis(ax)
panel_label(ax, "i")
save_panel(fig, OUTDIR, "fig1o_transcriptional_variability")
plt.close(fig)


# ---- Fig. 1q | cNMF rank selection for malignant-cell programmes ----
rank_table = read_table(SC_DIR / "4.Comparison" / "Treatment_divide" / "res.csv")
rank_col = first_present(rank_table.columns, ["k", "rank", "n_components"])
stability_col = first_present(rank_table.columns, ["stability", "silhouette", "score", "prediction_error"])
fig, ax = plt.subplots(figsize=(2.2, 1.8))
sns.lineplot(data=rank_table, x=rank_col, y=stability_col, marker="o", color="#333333", ax=ax)
ax.axvline(8, color="#D64A3A", linewidth=0.8, linestyle="--")
ax.set(xlabel="cNMF rank", ylabel=stability_col)
clean_axis(ax)
panel_label(ax, "j")
save_panel(fig, OUTDIR, "fig1q_cnmf_rank_selection")
plt.close(fig)


# ---- Fig. 1r | Spatial projection of malignant NMF niches ----
nmf_score_cols = [c for c in [f"NMF{i}_score" for i in range(1, 7)] if c in st.obs.columns]
st.obs["NMF_state"] = zscore_frame(st.obs[nmf_score_cols]).idxmax(axis=1).str.replace("_score", "", regex=False)
df_sp = spatial_xy(st)
fig, ax = plt.subplots(figsize=(2.5, 2.4))
sns.scatterplot(data=df_sp, x="x", y="y", hue="NMF_state", palette=NMF_COLORS, s=4, linewidth=0, ax=ax)
ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
ax.set_aspect("equal")
ax.legend(frameon=False, markerscale=2.5, bbox_to_anchor=(1.02, 1), loc="upper left")
panel_label(ax, "m")
save_panel(fig, OUTDIR, "fig1r_spatial_nmf_niches")
plt.close(fig)



# ---- Fig. 1t | Immunosuppressive functional scores across malignant niches ----
immune_score_cols = [c for c in ["CD4 Treg_score", "M2_score", "CD8 Cytotoxic T_score", "T cell_score"] if c in st.obs.columns]
immune_scores = st.obs[["NMF_state"] + immune_score_cols].melt("NMF_state", var_name="signature", value_name="score")
fig, ax = plt.subplots(figsize=(3.1, 2.2))
sns.boxplot(data=immune_scores, x="NMF_state", y="score", hue="signature", fliersize=0, linewidth=0.5, ax=ax)
ax.set(xlabel="", ylabel="Scaled score")
ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
clean_axis(ax)
panel_label(ax, "o")
save_panel(fig, OUTDIR, "fig1t_niche_immune_scores")
plt.close(fig)
