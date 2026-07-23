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
    ensure_outdir,
    first_present,
    get_gene_vector,
    panel_label,
    read_table,
    save_panel,
    set_nature_style,
)


LIPID_TRANSPORT_GENES = [
    "APOE",
    "LPL",
    "APOC1",
    "APOC2",
    "ABCA1",
    "ABCG1",
    "CD36",
    "FABP4",
    "FABP5",
]

NFKB_GENES = [
    "NFKB1",
    "NFKB2",
    "RELA",
    "RELB",
    "TNFAIP3",
    "NFKBIA",
    "CXCL10",
    "IL1B",
]

# ---- Fig. 6b | Differential expression after recombinant SLPI exposure ----
deg = read_table(TRANSCRIPTOME_DIR / "slpi_vs_control_deg.csv")
gene_col = first_present(deg.columns, ["gene", "Gene", "symbol"])
lfc_col = first_present(deg.columns, ["log2FoldChange", "logFC", "avg_log2FC"])
padj_col = first_present(deg.columns, ["padj", "FDR", "p_val_adj", "adjusted_pvalue"])
deg["neg_log10_fdr"] = -pd.Series(deg[padj_col]).astype(float).clip(lower=1e-300).map(lambda x: __import__("math").log10(x))
fig, ax = plt.subplots(figsize=(2.5, 2.2))
sns.scatterplot(data=deg, x=lfc_col, y="neg_log10_fdr", hue="highlight", palette={False: "#BBBBBB", True: "#D64A3A"}, s=8, linewidth=0, ax=ax)
ax.axvline(0, color="black", linewidth=0.5)
ax.set(xlabel="log2 fold change", ylabel="-log10(FDR)")
ax.legend(frameon=False, title="")
clean_axis(ax)
panel_label(ax, "f")
save_panel(fig, OUTDIR, "fig6f_slpi_vs_control_deg")
plt.close(fig)

# ---- Fig. 6c | GOBP enrichment of TREM2 macrophages compared with other macrophages ----
go = read_table(MACRO_DIR / "GO_depathway.csv")
term_col = first_present(go.columns, ["Term", "Description", "pathway", "Name"])
fdr_col = first_present(go.columns, ["Adjusted P-value", "p.adjust", "FDR", "adjusted_pvalue"])
go = go.sort_values(fdr_col).head(15)
go["neg_log10_fdr"] = -pd.Series(go[fdr_col]).astype(float).clip(lower=1e-300).map(lambda x: __import__("math").log10(x))
fig, ax = plt.subplots(figsize=(3.2, 2.5))
sns.barplot(data=go, x="neg_log10_fdr", y=term_col, color="#59A14F", ax=ax)
ax.set(xlabel="-log10(FDR)", ylabel="")
clean_axis(ax)
panel_label(ax, "a")
save_panel(fig, OUTDIR, "fig6a_trem2_macrophage_gobp_enrichment")
plt.close(fig)

# ---- Fig. 6d-fe| SLPI silencing reduces macrophage lipid-transport programme ----
score_table = read_table(TRANSCRIPTOME_DIR / "macrophage_pathway_scores.csv")
condition_col = first_present(score_table.columns, ["condition", "group", "treatment"])
score_col = first_present(score_table.columns, ["Lipid_transport_score", "lipid_transport", "score"])
fig, ax = plt.subplots(figsize=(2.4, 2.0))
sns.boxplot(data=score_table, x=condition_col, y=score_col, color="#59A14F", fliersize=0, linewidth=0.5, ax=ax)
sns.stripplot(data=score_table, x=condition_col, y=score_col, color="black", size=1.4, linewidth=0, alpha=0.45, ax=ax)
ax.set(xlabel="", ylabel="Lipid-transport score")
ax.tick_params(axis="x", rotation=45)
clean_axis(ax)
panel_label(ax, "c")
save_panel(fig, OUTDIR, "fig6c_lipid_transport_score_slpi_perturbation")
plt.close(fig)

# ---- Fig. 6f | Lipid-transport score across macrophage subtypes ----
macro = sc.read_h5ad(MACRO_DIR / "Macro_annonated.h5ad")
macro_col =  "sub_celltype"
lipid_genes = [g for g in LIPID_TRANSPORT_GENES if g in macro.var_names]
if "Lipid_transport_score" not in macro.obs:
    sc.tl.score_genes(macro, lipid_genes, score_name="Lipid_transport_score")
ov.pl.violion(macro,keys='Lipid_transport_score',groupby='sub_celltype')

# ---- Fig. 6h | APOE expression in TREM2 macrophages and SLPI perturbation ----
macro.obs["APOE"] = get_gene_vector(macro, "APOE")
fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0))
sns.violinplot(data=macro.obs, x=macro_col, y="APOE", inner=None, linewidth=0.4, color="#59A14F", ax=axes[0])
axes[0].set(xlabel="", ylabel="APOE expression")
axes[0].tick_params(axis="x", rotation=45)
apoe = read_table(TRANSCRIPTOME_DIR / "apoe_expression_by_condition.csv")
condition_col = first_present(apoe.columns, ["condition", "group", "treatment"])
expr_col = first_present(apoe.columns, ["APOE", "expression", "normalized_expression"])
sns.boxplot(data=apoe, x=condition_col, y=expr_col, color="#D64A3A", fliersize=0, linewidth=0.5, ax=axes[1])
sns.stripplot(data=apoe, x=condition_col, y=expr_col, color="black", size=1.4, linewidth=0, alpha=0.45, ax=axes[1])
axes[1].set(xlabel="", ylabel="APOE expression")
axes[1].tick_params(axis="x", rotation=45)
for ax in axes:
    clean_axis(ax)
panel_label(axes[0], "e")
save_panel(fig, OUTDIR, "fig6e_apoe_expression")
plt.close(fig)

# ---- Fig. 6l | KEGG enrichment among genes downregulated by SLPI ----
kegg_down = read_table(TRANSCRIPTOME_DIR / "slpi_downregulated_kegg.csv")
term_col = first_present(kegg_down.columns, ["Term", "Description", "pathway", "Name"])
fdr_col = first_present(kegg_down.columns, ["Adjusted P-value", "p.adjust", "FDR", "adjusted_pvalue"])
kegg_down = kegg_down.sort_values(fdr_col).head(12)
kegg_down["neg_log10_fdr"] = -pd.Series(kegg_down[fdr_col]).astype(float).clip(lower=1e-300).map(lambda x: __import__("math").log10(x))
fig, ax = plt.subplots(figsize=(3.0, 2.4))
sns.barplot(data=kegg_down, x="neg_log10_fdr", y=term_col, color="#4E79A7", ax=ax)
ax.set(xlabel="-log10(FDR)", ylabel="")
clean_axis(ax)
panel_label(ax, "g")
save_panel(fig, OUTDIR, "fig6g_slpi_downregulated_kegg")
plt.close(fig)

# ---- Fig. 6m | GSEA of NF-kB signalling after SLPI exposure ----
gsea = read_table(TRANSCRIPTOME_DIR / "slpi_vs_control_gsea.csv")
term_col = first_present(gsea.columns, ["Term", "Name", "pathway", "gene_set"])
nes_col = first_present(gsea.columns, ["NES", "nes"])
fdr_col = first_present(gsea.columns, ["FDR", "qval", "padj", "adjusted_pvalue"])
nfkb = gsea.loc[gsea[term_col].astype(str).str.contains("NF", case=False, regex=False)].sort_values(fdr_col).head(8)
fig, ax = plt.subplots(figsize=(3.0, 2.0))
sns.barplot(data=nfkb, x=nes_col, y=term_col, color="#4E79A7", ax=ax)
ax.axvline(0, color="black", linewidth=0.5)
ax.set(xlabel="Normalized enrichment score", ylabel="")
clean_axis(ax)
panel_label(ax, "h")
save_panel(fig, OUTDIR, "fig6h_nfkb_gsea")
plt.close(fig)


# ---- Fig. 6n| shSLPI enrichment analysis supports reciprocal NF-kB activation ----
kegg_sh = read_table(TRANSCRIPTOME_DIR / "shslpi_vs_control_kegg.csv")
term_col = first_present(kegg_sh.columns, ["Term", "Description", "pathway", "Name"])
fdr_col = first_present(kegg_sh.columns, ["Adjusted P-value", "p.adjust", "FDR", "adjusted_pvalue"])
kegg_sh = kegg_sh.sort_values(fdr_col).head(12)
kegg_sh["neg_log10_fdr"] = -pd.Series(kegg_sh[fdr_col]).astype(float).clip(lower=1e-300).map(lambda x: __import__("math").log10(x))
fig, ax = plt.subplots(figsize=(3.0, 2.4))
sns.barplot(data=kegg_sh, x="neg_log10_fdr", y=term_col, color="#D64A3A", ax=ax)
ax.set(xlabel="-log10(FDR)", ylabel="")
clean_axis(ax)
panel_label(ax, "i")
save_panel(fig, OUTDIR, "fig6i_shslpi_kegg")
plt.close(fig)


# ---- Fig. 6o | NF-kB pathway score after SLPI gain and loss ----
if "NFkB_score" not in score_table.columns:
    score_table["NFkB_score"] = score_table[[c for c in NFKB_GENES if c in score_table.columns]].mean(axis=1)
fig, ax = plt.subplots(figsize=(2.4, 2.0))
sns.boxplot(data=score_table, x=condition_col, y="NFkB_score", color="#4E79A7", fliersize=0, linewidth=0.5, ax=ax)
sns.stripplot(data=score_table, x=condition_col, y="NFkB_score", color="black", size=1.4, linewidth=0, alpha=0.45, ax=ax)
ax.set(xlabel="", ylabel="NF-kB pathway score")
ax.tick_params(axis="x", rotation=45)
clean_axis(ax)
panel_label(ax, "j")
save_panel(fig, OUTDIR, "fig6j_nfkb_score_slpi_perturbation")
plt.close(fig)
