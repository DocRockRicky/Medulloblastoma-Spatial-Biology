## ----setup, include=FALSE-----------------------------
library(Seurat)
library(ggplot2)
library(patchwork)
library(dplyr)
library(ape)
library(SeuratWrappers)
library(Banksy)
library(harmony)
library(hdf5r)
library(MAST)
library(clustree)
library(cluster)
library(sf)
library(Matrix)
library(clusterCrit) # for memory-efficient silhouette calculation


## -----------------------------------------------------

data_dir <- "/mount/mdtaylor2/ricardo/10XVisiumHD/MDT_AP_0102_spaceranger/outs"
image_dir <- file.path(data_dir, "segmented_outputs/spatial")
out_dir <- "/mount/mdtaylor2/ricardo/10XVisiumHD/Analysis/MDTAP0102"

# to load raw feature matrix
object  <- Load10X_Spatial(
  data.dir = data_dir,
  slice = "slice1",
  bin.size = c("polygons",8,16)
)
# Subset Seurat object to exclude cells
object@meta.data <- object@meta.data[, !grepl("Spatial.008um|Spatial.016um", colnames(object@meta.data))]
object <- subset(object, subset = !grepl("s", orig.ident))
DefaultAssay(object) <- "Spatial.Polygons"
head(object@meta.data)


## -----------------------------------------------------
# ===Quality Control (QC)===# =========================================================================
# remove bins with low Unique Molecular Identifier (UMI) counts.
pre.vln.plot <- VlnPlot(object, features = "nCount_Spatial.Polygons", pt.size = 0) + theme(axis.text = element_text(size = 4)) + NoLegend()
pre.count.plot <- SpatialFeaturePlot(object, features = "nCount_Spatial.Polygons") + theme(legend.position = "right")
pre.vln.plot 
pre.count.plot

# Calculate mitochondrial percent
object[["percent.mt"]] <- PercentageFeatureSet(object, pattern = "^MT-")

# UMI filtering with lower and upper cutoffs
thres <- quantile(object$nCount_Spatial.Polygons, 0.98, na.rm = TRUE)
object <- subset(object, subset = nCount_Spatial.Polygons > 10 & nCount_Spatial.Polygons < thres)

# Gene filtering: keep genes detected in >= 3 bins
detected_counts <- Matrix::rowSums(GetAssayData(object, slot = "counts") > 0)
genes_to_keep <- names(detected_counts[detected_counts >= 3])
object <- subset(object, features = genes_to_keep)

# Bin filtering: keep bins with >= 100 detected genes
genes_per_bin <- Matrix::colSums(GetAssayData(object, slot = "counts") > 0)
object$nFeature_Spatial.Polygons <- genes_per_bin
object <- subset(object, subset = nFeature_Spatial.Polygons >= 0)

# Mitochondrial percent filtering (e.g., max 10%)
max.mito.percent <- 10
object <- subset(object, subset = percent.mt < max.mito.percent)

hist(object$nFeature_Spatial.Polygons, main = "Detected Genes per Bin", xlab = "Number of Genes", breaks = 5000)
abline(v = 10, col = "red", lwd = 2)

post.vln.plot <- VlnPlot(object, features = "nCount_Spatial.Polygons", pt.size = 0) + theme(axis.text = element_text(size = 4)) + NoLegend()
post.count.plot <- SpatialFeaturePlot(object, features = "nCount_Spatial.Polygons") + theme(legend.position = "right")
post.vln.plot 
post.count.plot

#Post QC Histogram and Bins counts* 
hist(object$nCount_Spatial.Polygons, xlim = c(0, 3000), breaks = 5000)
abline(v = 00)
n_bins <- ncol(object)
print(n_bins)

#Save
saveRDS(object, file = file.path(out_dir, "Objects","QC_filtered_object_MDTAP0102.rds"))
ggsave(file = file.path(out_dir, "Images", "0102_pre_countplot.tiff"),
       plot = pre.count.plot, device = "tiff", type = "cairo",
       width = 8, height = 10, units = "in",
       dpi = 300, compression = "lzw")
ggsave(file = file.path(out_dir, "Images", "0102_pre_vlnplot.tiff"),
       plot = pre.vln.plot, device = "tiff", type = "cairo",
       width = 8, height = 10, units = "in",
       dpi = 300, compression = "lzw")
ggsave(file = file.path(out_dir, "Images", "0102_post_countplot.tiff"),
       plot = post.count.plot, device = "tiff", type = "cairo",
       width = 8, height = 10, units = "in",
       dpi = 300, compression = "lzw")
ggsave(file = file.path(out_dir, "Images", "0102_post_vlnplot.tiff"),
       plot = post.vln.plot, device = "tiff", type = "cairo",
       width = 8, height = 10, units = "in",
       dpi = 300, compression = "lzw")



## -----------------------------------------------------
# normalize our data using median transcript count as the scale factor to normalize based on library size.
 object <- NormalizeData(object, assay = "Spatial.Polygons", normalization.method = "LogNormalize", scale.factor = 1e6)
## Find top 2k variable genes
object <- FindVariableFeatures(object)
  print(paste("Number of Variable Features:", length(VariableFeatures(object))))


## -----------------------------------------------------
# Inputs (assign your actual values)
nPCs <- 50                  # Number of PCs to calculate, e.g., 100
ndims <- 50                   # Number of dimensions for JackStraw, e.g., 100
nrep <- 100                   # Number of JackStraw replicates, e.g., 100
ifelse(nPCs < ndims, ndims <- nPCs, ndims <- ndims)# Adjust ndims based on nPCs


## -----------------------------------------------------
object <- ScaleData(object)# Scale the data
object <- RunPCA(object = object, npcs = nPCs, verbose = FALSE)


## -----------------------------------------------------
ElbowPlot(object, ndims = 50)
# Look for the “elbow” where additional PCs contribute little extra variance; often coincides with the tail-off of JackStraw significance
pdf(file = file.path(out_dir,"Images", "ElbowPlot.pdf"), width = 10, height = 10, useDingbats = FALSE)
ElbowPlot(object, ndims = 50)# shows the percent variance explained per PC—use in combination with JackStraw:
dev.off()


## -----------------------------------------------------
# Run JackStraw analysis if not already done
if (is.null(object@misc$JSsigPC)) {
  object <- JackStraw(object = object, num.replicate = nrep, dims = ndims)
  object <- ScoreJackStraw(object = object, dims = 1:ndims)
  
  tbl_jack <- object[['pca']]@jackstraw$overall.p.values %>% as.data.frame()
  sig_PC <- tbl_jack$PC[tbl_jack$Score > 0.05][1] - 1
  object@misc$JSsigPC <- sig_PC
} else {
  sig_PC <- object@misc$JSsigPC
}
# Print number of significant PCs
print(paste0("Number of PCs that were significant from JackStraw analysis : ", sig_PC))

# Save CSVs
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
write.csv(tbl_jack, file = file.path(out_dir, "Values", "JackStraw_pvalues.csv"), row.names = FALSE)
write.csv(data.frame(sig_PC = sig_PC), file = file.path(out_dir,"Values", "Significant_PCs.csv"), row.names = FALSE)

# Save plots to PDF# The plot shows, for each PC, a QQ-plot of observed vs. null p-values. PCs with a significant "tail" of low p-values are likely to capture significant structure.
# Retain those with strong evidence of significance (see where the plot transitions from significant to background-like).
pdf(file = file.path(out_dir, "Images", "JackStrawPlot.pdf"), width = 15, height = 10, useDingbats = FALSE)
JackStrawPlot(object, dims = 1:nrow(tbl_jack))
dev.off()


## -----------------------------------------------------
object <- RunUMAP(object, dims = 1:sig_PC)
object <- FindNeighbors(object=object, dims=1:sig_PC)


## -----------------------------------------------------
#' Finds the optimal cluster resolution for a Seurat object using Silhouette method (clusterCrit).
find_optimal_resolution <- function(seurat_obj, resolutions_to_test, dims_to_use, reduction_type = "pca") {
  # Ensure the specified reduction exists in the Seurat object
  if (!reduction_type %in% Reductions(seurat_obj)) {
    stop(paste("The specified reduction type", reduction_type, "is not found in the Seurat object."))
  }
  # Extract embeddings
  embeddings <- Embeddings(seurat_obj, reduction = reduction_type)
  # Create empty results data frame
  results_df <- data.frame(
    Resolution = numeric(),
    AvgSilhouetteScore = numeric(),
    stringsAsFactors = FALSE
  )
  message("Starting Silhouette score calculation for resolutions: ", paste(resolutions_to_test, collapse = ", "))
  for (res in resolutions_to_test) {
    message("Clustering with resolution: ", res)
    # Perform clustering
    seurat_obj <- FindNeighbors(seurat_obj, reduction = reduction_type, dims = dims_to_use, verbose = FALSE)
    seurat_obj <- FindClusters(seurat_obj, resolution = res, verbose = FALSE)
    clusters <- seurat_obj@meta.data$seurat_clusters

    if (length(unique(clusters)) > 1) {
      # Use clusterCrit for silhouette calculation (memory efficient)
      crit <- intCriteria(as.matrix(embeddings[, dims_to_use]), as.integer(clusters), "Silhouette")
      avg_sil_score <- crit$silhouette
      results_df <- rbind(results_df, data.frame(Resolution = res, AvgSilhouetteScore = avg_sil_score))
    } else {
      message("Warning: Only one cluster found at resolution ", res, ". Skipping Silhouette score calculation.")
      results_df <- rbind(results_df, data.frame(Resolution = res, AvgSilhouetteScore = NA))
    }
  }
  message("Calculation complete.")
  return(results_df)
}


# Define resolutions to test
resolutions_to_test <- seq(0.1, 2.0, by = 0.1)

# Run optimal resolution finder
silhouette_results <- find_optimal_resolution(object, resolutions_to_test, dims_to_use = 1:sig_PC)

# Print results
print(silhouette_results)

# Find best resolution
best_resolution <- silhouette_results %>%
  filter(AvgSilhouetteScore == max(AvgSilhouetteScore, na.rm = TRUE)) %>%
  pull(Resolution)
message("\nBest resolution based on Silhouette score: ", best_resolution)

# Save results as CSV
write.csv(silhouette_results,
          file = file.path(out_dir, "Values", "silhouette_scores.csv"),
          row.names = FALSE)

# Create and save silhouette plot
silhouette_plot <- ggplot(silhouette_results, aes(x = Resolution, y = AvgSilhouetteScore)) +
  geom_line(color = "blue") +
  geom_point(color = "red", size = 3) +
  labs(
    title = "Average Silhouette Score vs. Cluster Resolution",
    subtitle = "Higher score indicates better cluster separation",
    x = "Resolution",
    y = "Average Silhouette Score"
  ) +
  theme_bw()

ggsave(
  filename = file.path(out_dir, "Images", "silhouette_plot.pdf"),
  plot = silhouette_plot,
  device = "pdf",
  width = 8,
  height = 6
)
saveRDS(object, file = file.path(out_dir,"Objects", "optimal_res_MDTAP0102.rds"))



## -----------------------------------------------------
# --- USER-DEFINED INPUTS ---
# Load query and reference objects
object <- readRDS("/mount/mdtaylor2/ricardo/10XVisiumHD/Analysis/MDTAP0102/Objects/QC_filtered_object_MDTAP0102.rds")
ref1 <- readRDS("/mount/mdtaylor2/ricardo/10XVisiumHD/Annotations/G3_MB.rds")
ref2 <- readRDS("/mount/mdtaylor2/ricardo/10XVisiumHD/Annotations/Aldinger_NatNeurosci_2021.rds")

# --- DATA PREPARATION ---
# Prepare your query object
object <- NormalizeData(object)
object <- FindVariableFeatures(object)
object <- ScaleData(object)
object <- RunPCA(object)

# Prepare reference objects
# For ref1 (Visvanathan/MB)
ref1[["RNA"]] <- CreateAssayObject(counts = GetAssayData(ref1, assay = "SCT", slot = "counts"))
DefaultAssay(ref1) <- "RNA"
ref1 <- FindVariableFeatures(ref1)
ref1 <- ScaleData(ref1)
ref1 <- RunPCA(ref1)

# For ref2 (Aldinger)
ref2 <- CreateSeuratObject(counts = ref2@assays$RNA@counts, meta.data = ref2@meta.data)
DefaultAssay(ref2) <- "RNA"
ref2 <- NormalizeData(ref2)
ref2 <- FindVariableFeatures(ref2)
ref2 <- ScaleData(ref2)
ref2 <- RunPCA(ref2)

# Reference list and annotation column names
refs <- list(
  list(seu = ref1, column = "annotated.clusters", name = "Visvanathan"),
  list(seu = ref2, column = "fig_cell_type", name = "Aldinger")
)

# --- LABEL TRANSFER ---
# Annotate query object with both references
for (r in refs) {
  message("Finding anchors for reference: ", r$name)
  anchors <- FindTransferAnchors(
    reference = r$seu,
    query = object,
    dims = 1:sig_PC,
    reference.reduction = "pca"
  )
  
  message("Transferring data from reference: ", r$name)
  # FIX: The `drop = TRUE` argument is essential here to return a vector
  # of cell identities instead of a data frame, which is what `TransferData` expects.
  predictions <- TransferData(
    anchorset = anchors,
    refdata = r$seu[[r$column, drop = TRUE]], # CORRECTED LINE
    dims = 1:sig_PC
  )
  
  # Store the predicted cell type labels in the query object's metadata.
  object[[paste0("predicted.id_", r$name)]] <- predictions$predicted.id
}

message("Successfully transferred data and annotated the Seurat object.")

# --- INSPECTION AND EXPORT ---
# Optionally inspect annotations
message("Visvanathan Annotation Summary:")
print(table(object$predicted.id_Visvanathan))

message("\nAldinger Annotation Summary:")
print(table(object$predicted.id_Aldinger))

# Export combined annotations for Loupe Browser
anno_cols <- c("predicted.id_Visvanathan", "predicted.id_Aldinger")
loupe_df <- data.frame(
  Barcode = rownames(object@meta.data),
  object@meta.data[, anno_cols, drop = FALSE]
)
# Ensure the output directory exists
if (!dir.exists(file.path(out_dir, "Values"))) {
  dir.create(file.path(out_dir, "Values"), recursive = TRUE)
}
write.csv(loupe_df, file = file.path(out_dir,"Values", "predicted_loupe_annotations.csv"), row.names = FALSE)

message("\nLoupe browser annotation file saved to: ", file.path(out_dir,"Values", "predicted_loupe_annotations.csv"))



## -----------------------------------------------------
# --- SAVE ANNOTATED OBJECT ---
message("\nSaving the annotated Seurat object...")
# It's good practice to save the final object for future use, including all
# the new metadata columns.
saveRDS(object, file = file.path(out_dir, "Objects", "Annotated_Visvanathan_Aldinger_MDTAP1166.rds"))
message("Annotated object saved to: ", file.path(out_dir, "Objects", "Annotated_MDTAP1166.rds"))

