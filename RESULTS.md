# Deep Multi-Graph Clustering

## Molecular and cellular interactions in disease

### 1. Validation across five heart scRNA-seq studies

DMGC recovers aligned cross-study cluster blocks that correspond to lymphoid, myeloid, mesenchymal, and endothelial identities.

<p align="center">
  <img src="results/figures/heart_validation_cell_type_alignment.png" alt="Cross-study cell-type alignment heatmap" width="560" />
</p>

### 2. Shared gene modules across organs

Mesenchymal-cell modules are separated in lung and show associations across lung, kidney, heart, and liver.

<p align="center">
  <img src="results/figures/cross_organ_module_alignment.png" alt="Separated lung gene modules" width="720" />
</p>

Shared modules show enrichment for extracellular-region programs, angiogenesis, and IL-10/IL-17 signaling.

<p align="center">
  <img src="results/figures/cross_organ_pathway_enrichment.png" alt="Shared pathway enrichment across organs" width="600" />
</p>

### 3. Spatial tonsil interaction

Cell-type GSEA identifies significant enrichment for both NBC_MBC (NES 2.84, FDR 0.00) and CD4_T (NES 1.51, FDR 0.012).

<p align="center">
  <img src="results/figures/spatial_gsea_cell_type_enrichment_table.png" alt="GSEA cell-type enrichment table showing NBC MBC and CD4 T results" width="680" />
</p>

The NBC_MBC–CD4 T-cell association is enriched for MHC-II antigen presentation, T-cell receptor signaling, and T-follicular-helper differentiation.

<p align="center">
  <img src="results/figures/spatial_cellular_interaction_enrichment.png" alt="NBC MBC and CD4 T-cell enrichment results" width="760" />
</p>

Candidate co-occurring regions are visible in the spatial transcriptomics tissue map.

<p align="center">
  <img src="results/figures/spatial_interaction_visualization.png" alt="Spatial visualization of NBC MBC and CD4 T-cell modules" width="760" />
</p>

## Analysis workflows

<p align="center">
  <img src="results/figures/flow_heart_validation.png" alt="Heart validation workflow" width="760" />
</p>

<p align="center">
  <img src="results/figures/flow_cross_organ.png" alt="Cross-organ analysis workflow" width="760" />
</p>

<p align="center">
  <img src="results/figures/flow_spatial_integration.png" alt="Spatial integration workflow" width="760" />
</p>

[Complete presentation](results/presentation_final.pptx)
