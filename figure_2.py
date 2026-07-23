from __future__ import annotations

import os
from pathlib import Path
import omicverse as ov
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

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



# ---- Fig. 2g | TCGA survival stratified by NMF4 score ----
tcga = read_table(BULK_DIR / "TCGA" / "sample_info.csv")
time_col = first_present(tcga.columns, ["OS.time", "time", "days_to_death"])
event_col = first_present(tcga.columns, ["OS", "event", "vital_status"])
group_col = first_present(tcga.columns, ["cNMF_4_score_group", "NMF4_group", "score_group"])
fig, ax = plt.subplots(figsize=(2.5, 2.1))
km_plot(ax, tcga, time=time_col, event=event_col, group=group_col)
panel_label(ax, "c")
save_panel(fig, OUTDIR, "fig2c_tcga_nmf4_km")
plt.close(fig)


# ---- Fig. 2i-j | CheckMate NMF4 score and ICB benefit ----
checkmate = read_table(BULK_DIR / "checkmate" / "clinical.xlsx")
score_col = first_present(checkmate.columns, ["cNMF_4_score", "NMF4_score", "NMF4"])
benefit_col = first_present(checkmate.columns, ["Benefit", "Response", "BOR", "clinical_benefit"])
fig, ax = plt.subplots(figsize=(2.4, 2.0))
sns.boxplot(data=checkmate, x=benefit_col, y=score_col, color="#D64A3A", fliersize=0, linewidth=0.5, ax=ax)
sns.stripplot(data=checkmate, x=benefit_col, y=score_col, color="black", size=1.2, linewidth=0, alpha=0.45, ax=ax)
ax.set(xlabel="", ylabel="NMF4 score")
ax.tick_params(axis="x", rotation=45)
clean_axis(ax)
panel_label(ax, "d")
save_panel(fig, OUTDIR, "fig2d_checkmate_nmf4_icb_benefit")
plt.close(fig)


# ---- Fig. 2k | JAVELIN NMF4 score and treatment outcome ----
javelin = read_table(BULK_DIR / "JAVLIN" / "Clinical.xlsx")
score_col = first_present(javelin.columns, ["cNMF_4_score", "NMF4_score", "NMF4"])
outcome_col = first_present(javelin.columns, ["Best_response", "BOR", "Response", "PFS_status"])
fig, ax = plt.subplots(figsize=(2.4, 2.0))
sns.boxplot(data=javelin, x=outcome_col, y=score_col, color="#D64A3A", fliersize=0, linewidth=0.5, ax=ax)
sns.stripplot(data=javelin, x=outcome_col, y=score_col, color="black", size=1.2, linewidth=0, alpha=0.45, ax=ax)
ax.set(xlabel="", ylabel="NMF4 score")
ax.tick_params(axis="x", rotation=45)
clean_axis(ax)
panel_label(ax, "e")
save_panel(fig, OUTDIR, "fig2e_javelin_nmf4_outcome")
plt.close(fig)





# ---- Fig. 2u | Spatial enrichment of NMF4 at tumor-interface regions ----
tif = sc.read_h5ad(ST_DIR / "New" / "Data2" / "interface_st.h5ad")
score_col = first_present(tif.obs.columns, ["NMF4_score"])
sc.pl.spatial(adata,color='NMF4_score"')



# ---- Fig. 2v | Single-cell validation of NMF4 score in interface versus core samples ----
interface = sc.read_h5ad(SC_DIR / "4.Comparison" / "Treatment_divide" / "present" / "validation" / "interface.h5ad")
score_col = first_present(interface.obs.columns, ["NMF4_score", "NMF4"])
region_col = first_present(interface.obs.columns, ["Region", "region", "site", "sample_region"])
ov.pl.violin(adata,keys='NMF4_score',groupby='Region')
