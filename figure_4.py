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
    km_plot,
    panel_label,
    read_table,
    save_panel,
    set_nature_style,
)


def spatial_xy(adata) -> pd.DataFrame:
    xy = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"], index=adata.obs_names)
    return xy.join(adata.obs)


# ---- Fig. 4c | Cell-communication strength between malignant programmes and immune compartments ----
comm = read_table(CELLCOM_DIR / "summary.csv")
source_col = first_present(comm.columns, ["source", "source_cell", "sender", "cell_source"])
target_col = first_present(comm.columns, ["target", "target_cell", "receiver", "cell_target"])
score_col = first_present(comm.columns, ["score", "magnitude_rank", "interaction_score", "weight"])
comm_heat = comm.pivot_table(index=source_col, columns=target_col, values=score_col, aggfunc="mean")
fig, ax = plt.subplots(figsize=(3.2, 2.6))
sns.heatmap(comm_heat, cmap="Reds", linewidths=0.2, linecolor="white", cbar_kws={"label": "Interaction score"}, ax=ax)
ax.set(xlabel="", ylabel="")
panel_label(ax, "a")
save_panel(fig, OUTDIR, "fig4a_liana_cell_communication_heatmap")
plt.close(fig)


# ---- Fig. 4d | Ranking of secreted ligands from NMF4 malignant cells ----
rank_file = CELLCOM_DIR /  "\u6392\u5e8f.csv"
ligands = read_table(rank_file)
ligand_col = first_present(ligands.columns, ["ligand", "gene", "ligand_complex", "molecule"])
score_col = first_present(ligands.columns, ["score", "rank_score", "specificity", "weight"])
top = ligands.sort_values(score_col, ascending=False).head(15)
fig, ax = plt.subplots(figsize=(2.8, 2.4))
sns.barplot(data=top, x=score_col, y=ligand_col, color="#D64A3A", ax=ax)
ax.set(xlabel="Secreted-factor score", ylabel="")
clean_axis(ax)
panel_label(ax, "b")
save_panel(fig, OUTDIR, "fig4b_nmf4_secreted_ligand_ranking")
plt.close(fig)




# ---- Fig. 4e | SLPI survival association in bulk RCC cohorts ----
for cohort, filename in {
    "TCGA": "tcga_slpi_group.csv",
    "UN-Cohort": "UN_Cohort_slpi_group.csv",
    "CheckMate": "checkmate_slpi_group.csv",
    "JAVELIN": "javelin_slpi_group.csv",
}.items():
    data = read_table(CELLCOM_DIR / "cohort" / filename)
    time_col = first_present(data.columns, ["OS.time", "PFS", "time", "days"])
    event_col = first_present(data.columns, ["OS", "PFS_event", "event", "status"])
    group_col = first_present(data.columns, ["SLPI_group", "group", "score_group"])
    fig, ax = plt.subplots(figsize=(2.3, 2.0))
    km_plot(ax, data, time=time_col, event=event_col, group=group_col)
    ax.set_title(cohort)
    panel_label(ax, "e")
    save_panel(fig, OUTDIR, f"fig4e_slpi_km_{cohort.lower().replace('-', '_')}")
    plt.close(fig)


# ---- Fig. 4g | SLPI expression in spatial transcriptomics ----
st = sc.read_h5ad(ST_DIR / "New" / "Data2" / "ST_ALL.h5ad")
if "SLPI" not in st.obs:
    st.obs["SLPI"] = get_gene_vector(st, "SLPI")
df_sp = spatial_xy(st)
fig, ax = plt.subplots(figsize=(2.4, 2.3))
sns.scatterplot(data=df_sp, x="x", y="y", hue="SLPI", palette="magma", s=4, linewidth=0, ax=ax)
ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
ax.set_aspect("equal")
ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
panel_label(ax, "f")
save_panel(fig, OUTDIR, "fig4f_spatial_slpi_expression")
plt.close(fig)



# ---- Fig. 4o | Immune-cell single-cell atlas used for SLPI-associated remodeling ----
immune = sc.read_h5ad(CELLCOM_DIR / "mouse.h5ad")
immune_col = first_present(immune.obs.columns, ["sub_celltype", "cell_type", "Celltype", "annotation"])
emb_immune = embedding_dataframe(immune, immune_col, basis=first_obsm_key(immune))
fig, ax = plt.subplots(figsize=(3.0, 2.6))
ov.pl.embedding(adata,color='sub_celltype')
immune_T = sc.read_h5ad(CELLCOM_DIR / "mouse_T.h5ad")
ov.pl.cellproportion(adata,keys='sub_celltype',groupby=''group")

