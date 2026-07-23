from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import omicverse as ov
from plot_style import (
    NMF_COLORS,
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


# ---- Fig. 3a | CytoTRACE2 score across malignant NMF programmes ----
tumor = sc.read_h5ad(SC_DIR / "4.Comparison" / "Treatment_divide" / "CNMF_GOOD_COLOR.h5ad")
cyto_col = first_present(tumor.obs.columns, ["CytoTRACE2_Score", "CytoTRACE2", "cytotrace2_score"])
ov.pl.embedding(adata,color=cyto_col,groupby='NMF_cluster')


# ---- Fig. 3b | CytoTRACE2 score by malignant programme ----
fig, ax = plt.subplots(figsize=(2.8, 2.0))
sns.boxplot(data=tumor.obs, x=nmf_col, y=cyto_col, palette=NMF_COLORS, fliersize=0, linewidth=0.5, ax=ax)
ax.set(xlabel="", ylabel="CytoTRACE2 score")
ax.tick_params(axis="x", rotation=45)
clean_axis(ax)
panel_label(ax, "b")
save_panel(fig, OUTDIR, "fig3b_cytotrace2_by_nmf")
plt.close(fig)


# ---- Fig. 3c | Stemness score by malignant programme ----
if "Stemness_score" not in tumor.obs:
    stemness_genes = [g for g in ["CD44", "ALDH1A1", "PROM1", "SOX2", "MYC", "KLF4"] if g in tumor.var_names]
    sc.tl.score_genes(tumor, stemness_genes, score_name="Stemness_score")
fig, ax = plt.subplots(figsize=(2.8, 2.0))
sns.boxplot(data=tumor.obs, x=nmf_col, y="Stemness_score", palette=NMF_COLORS, fliersize=0, linewidth=0.5, ax=ax)
ax.set(xlabel="", ylabel="Stemness score")
ax.tick_params(axis="x", rotation=45)
clean_axis(ax)
panel_label(ax, "c")
save_panel(fig, OUTDIR, "fig3c_stemness_score_by_nmf")
plt.close(fig)


# ---- Fig. 3d | Stemness-associated genes in malignant programmes ----
stemness_panel = [g for g in ["CD44", "ALX1", "FOXM1", "HMGA2", "SOX2",] if g in tumor.var_names]
dot = sc.pl.dotplot(tumor, stemness_panel, groupby=nmf_col, standard_scale="var", show=False, return_fig=True)
dot.savefig(OUTDIR / "fig3d_stemness_gene_dotplot.pdf")


# ---- Fig. 3f | Pathway enrichment in NMF4 ----
go = read_table(SC_DIR / "4.Comparison" / "Treatment_divide" / "GO_depathway.csv")
term_col = first_present(go.columns, ["Term", "Description", "pathway", "Name"])
fdr_col = first_present(go.columns, ["Adjusted P-value", "p.adjust", "FDR", "adjusted_pvalue"])
go_nmf4 = go.loc[go[term_col].astype(str).str.contains("senescence|cell cycle|stem", case=False, regex=True)].copy()
go_nmf4 = go_nmf4.sort_values(fdr_col).head(12)
go_nmf4["neg_log10_fdr"] = -pd.Series(go_nmf4[fdr_col]).astype(float).clip(lower=1e-300).map(lambda x: __import__("math").log10(x))
fig, ax = plt.subplots(figsize=(3.0, 2.3))
sns.barplot(data=go_nmf4, x="neg_log10_fdr", y=term_col, color="#D64A3A", ax=ax)
ax.set(xlabel="-log10(FDR)", ylabel="")
clean_axis(ax)
panel_label(ax, "e")
save_panel(fig, OUTDIR, "fig3e_nmf4_senescence_enrichment")
plt.close(fig)


# ---- Fig. 3h-k | Senepy senescence score by malignant programme ----
sen_col = first_present(tumor.obs.columns, ["Senescence", "Senepy_score", "Cellular_senescence_score", "senescence_score"])
fig, ax = plt.subplots(figsize=(2.8, 2.0))
sns.boxplot(data=tumor.obs, x=nmf_col, y=sen_col, palette=NMF_COLORS, fliersize=0, linewidth=0.5, ax=ax)
ax.set(xlabel="", ylabel="Senescence-like score")
ax.tick_params(axis="x", rotation=45)
clean_axis(ax)
panel_label(ax, "f")
save_panel(fig, OUTDIR, "fig3f_senescence_score_by_nmf")
plt.close(fig)


# ---- Fig. 3h-k | Canonical senescence genes across malignant programmes ----
senescence_genes = [g for g in ["CDKN2A", "CDKN2B", "GLB1", "TP53"] if g in tumor.var_names]
dot = sc.pl.dotplot(tumor, senescence_genes, groupby=nmf_col, standard_scale="var", show=False, return_fig=True)
dot.savefig(OUTDIR / "fig3g_senescence_gene_dotplot.pdf")


# ---- Fig. 3n | Bulk-cohort senescence marker expression by NMF4 score group ----
bulk_marker = read_table(SC_DIR / "8.Cellcom" / "cohort" / "UN_Cohort_group_median_scaled.csv")
group_col = first_present(bulk_marker.columns, ["group", "NMF4_group"])
marker_cols = [c for c in ["CDKN2A", "CDKN2B", "GLB1", "TP53", "SERPINE1", "IL6", "CXCL8"] if c in bulk_marker.columns]
heat = bulk_marker.groupby(group_col)[marker_cols].mean()
fig, ax = plt.subplots(figsize=(2.8, 1.8))
sns.heatmap(heat, cmap="Reds", center=0, linewidths=0.2, linecolor="white", cbar_kws={"label": "Scaled expression"}, ax=ax)
ax.set(xlabel="", ylabel="")
panel_label(ax, "h")
save_panel(fig, OUTDIR, "fig3h_bulk_senescence_markers_by_nmf4")
plt.close(fig)


# ---- Fig. 3l | Spatial colocalization of NMF4 score and senescence score ----
nmf_sen = read_table(ST_DIR / "1.NMF4_senescence" / "NMF4_vs_senescence.csv")
x_col = first_present(nmf_sen.columns, ["NMF4_score",])
y_col = first_present(nmf_sen.columns, ["Cellular_senescence_score",])
fig, ax = plt.subplots(figsize=(2.2, 2.0))
sns.regplot(data=nmf_sen, x=x_col, y=y_col, scatter_kws={"s": 5, "linewidth": 0, "alpha": 0.55}, line_kws={"linewidth": 0.8}, ax=ax)
ax.set(xlabel="NMF4 score", ylabel="Senescence-like score")
clean_axis(ax)
panel_label(ax, "i")
save_panel(fig, OUTDIR, "fig3i_st_nmf4_senescence_correlation")
plt.close(fig)


# ---- Fig. 3m | Spatial maps of NMF4 and senescence-like scores ----
st = sc.read_h5ad(ST_DIR / "New" / "Data2" / "GSM5924037_ffpe_c_34_filtered_feature_bc_matrix_processed_annotated.h5ad")
df_sp = spatial_xy(st)
score_cols = [c for c in ["NMF4_score", "Cellular_senescence_score"] if c in df_sp.columns]
fig, axes = plt.subplots(1, len(score_cols), figsize=(2.4 * len(score_cols), 2.3))
if len(score_cols) == 1:
    axes = [axes]
for ax, score in zip(axes, score_cols):
    sns.scatterplot(data=df_sp, x="x", y="y", hue=score, palette="Reds", s=4, linewidth=0, ax=ax)
    ax.set(xticks=[], yticks=[], xlabel="", ylabel="", title=score)
    ax.set_aspect("equal")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
panel_label(axes[0], "j")
save_panel(fig, OUTDIR, "fig3j_spatial_nmf4_senescence_maps")
plt.close(fig)



