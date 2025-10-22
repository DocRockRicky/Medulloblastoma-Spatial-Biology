#%% md
# # Visium HD sample Analysis in Python:
# Python Environment Setup and Library Installation
#%%
import warnings
warnings.filterwarnings("ignore")
import spatialdata as spd
import spatialdata_plot as splt
import spatialdata_io as so
import geosketch as sketch
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce

import json
import gc
import os
import geopandas as gpd
from spatialdata.models import Image2DModel, TableModel, ShapesModel
import matplotlib.pyplot as plt

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from PIL import Image
from spatialdata.transformations import Identity, Scale
from shapely.geometry import Polygon
#%% md
# # Helper Functions
#%% md
# #create_zarr
# 
# five inputs:
# 
# image_path: The path to the image file.
# 
# count_matrix_path: The path to the cell segmentation filtered feature-barcode count matrix file.
# 
# scale_factors_path: The path to the scale factors JSON file.
# 
# geojson_path: The path to the cell segmentation GeoJSON file.
# 
# sample_name: A name for the Zarr output file.
#%%
def create_zarr(count_matrix_path,
                image_path,
                scale_factors_path,
                geojson_path,
                sample_name
):
    print(sample_name)

    # Load and Prepare Raw Data
    # Define file paths
    COUNT_MATRIX_PATH = count_matrix_path
    IMAGE_PATH = image_path
    SCALE_FACTORS_PATH = scale_factors_path
    GEOJSON_PATH = geojson_path

    # Load AnnData
    adata = sc.read_10x_h5(COUNT_MATRIX_PATH)
    adata.var_names_make_unique()
    adata.obs['sample'] = sample_name
    adata.obs.index = sample_name +"_" + adata.obs.index.astype(str)

    # Load and preprocess image data
    image_data = np.array(Image.open(IMAGE_PATH))
    if image_data.ndim == 2:
        image_data = image_data[np.newaxis, :, :] # Add channel dimension for grayscale
    elif image_data.ndim == 3:
        image_data = np.transpose(image_data, (2, 0, 1)) # (H, W, C) -> (C, H, W) for spatialdata

    # Load scale factors
    with open(SCALE_FACTORS_PATH, 'r') as f:
        scale_data = json.load(f)

    # Load GeoJSON data
    with open(GEOJSON_PATH, 'r') as f:
        geojson_data = json.load(f)

    # Define coordinate systems:
    # `downscale_to_hires`: The coordinate system where shapes are located, scaled relative to the hires resolution.

    hires_scale = scale_data['tissue_hires_scalef']

    # Transformation for shapes (from pixel to downscale_to_hires)
    shapes_transformations = {
       "downscale_to_hires": Scale(np.array([hires_scale, hires_scale]), axes=("x", "y")) # if the high-resolution microscope image is being used and Identity() transform would be performed.
    }

    # Transformation for the 'hires_tissue_image' (it's already in the 'downscale_to_hires' space visually)
    image_transformations = {
        "downscale_to_hires": Identity()
    }

    # Process Cell Segmentation (GeoJSON) and Integrate with AnnData

    # Create a mapping from adata.obs.index to geojson features
    geojson_features_map = {
        f"{sample_name}_cellid_{feature['properties']['cell_id']:09d}-1": feature
        for feature in geojson_data['features']
    }

    # Prepare data for GeoDataFrame and update adata.obs
    geometries = []
    cell_ids_ordered = []

    for obs_index_str in adata.obs.index:
        feature = geojson_features_map.get(obs_index_str)
        if feature:
            # Create shapely Polygon from coordinates
            polygon_coords = np.array(feature['geometry']['coordinates'][0])
            geometries.append(Polygon(polygon_coords))
            cell_ids_ordered.append(obs_index_str)
        else:
            geometries.append(None) # Or a suitable placeholder
            cell_ids_ordered.append(obs_index_str)

    # Remove None entries if any (or handle them upstream)
    valid_indices = [i for i, geom in enumerate(geometries) if geom is not None]
    geometries = [geometries[i] for i in valid_indices]
    cell_ids_ordered = [cell_ids_ordered[i] for i in valid_indices]


    # Create GeoDataFrame for shapes
    shapes_gdf = gpd.GeoDataFrame({
        'cell_id': cell_ids_ordered,
        'geometry': geometries
    }, index=cell_ids_ordered)
    # Update adata.obs with cluster information and spatial identifiers
    adata.obs['cell_id'] = adata.obs.index
    adata.obs['region'] = sample_name + '_cell_boundaries'
    adata.obs['region'] = adata.obs['region'].astype('category')
    adata = adata[shapes_gdf.index].copy() # Filter adata to match shapes_gdf

    # Define names for SpatialData elements
    IMAGE_KEY =  sample_name + '_hires_tissue_image'
    TABLE_KEY =  'segmentation_counts'
    SHAPES_KEY = sample_name + '_cell_boundaries'

    # Create SpatialData elements directly
    sdata = spd.SpatialData(
        images={
            IMAGE_KEY: Image2DModel.parse(image_data, transformations=image_transformations)
        },
        tables={
            TABLE_KEY: TableModel.parse(
                adata,
                region=SHAPES_KEY, # Link table to shapes element
                region_key='region', # Column in adata.obs indicating region name
                instance_key='cell_id' # Column in adata.obs with instance IDs (cell_id)
            )
        },
        shapes={
            SHAPES_KEY: ShapesModel.parse(shapes_gdf, transformations=shapes_transformations)
        }
    )

    sdata.write(sample_name, overwrite=True)
    del sdata
    gc.collect()

#%% md
# If I want to use square-bin outputs instead of the segmentation-based bin outputs,use the so.visium_hd function from spatialdata version 0.4.0. This simplifies the create_zarr function, as seen in this code:
# 
# def create_zarr(
#                 path_to_outputs,
#                 zarr_name,
#                 bin_size
# ):
#     print(zarr_name)
#     sdata = so.visium_hd(path=path_to_outputs,
#                          load_all_images=True, bin_size=bin_size)
#     sdata.write(zarr_name, overwrite=True)
#     del sdata
#%% md
# #crop0
#%%
def crop0(x,crs,bbox):
    return spd.bounding_box_query(
        x,
        min_coordinate=[bbox['x'][0], bbox['y'][0]],
        max_coordinate=[bbox['x'][1], bbox['y'][1]],
        axes=("x", "y"),
        target_coordinate_system=crs,
    )
#%% md
# # Create and save Zarr files for the cell segmentation outputs.
#%%
# Define the base paths for the two samples
samples_base_paths = {
    "MDT_AP_1166": "/Users/ricardodgonzalez/Desktop/10X Genomics/April_14_2025/Outputs/MDT_AP_1166_spaceranger/outs/segmented_outputs",
    "MDT_AP_0102": "/Users/ricardodgonzalez/Desktop/10X Genomics/April_14_2025/Outputs/MDT_AP_0102_spaceranger/outs/segmented_outputs"
}

# Prepare the samples dictionary using the base paths and relative file locations
samples = {}
for sample_name, base_path in samples_base_paths.items():
    samples[sample_name] = {
        "count_matrix_path": os.path.join(base_path, "filtered_feature_cell_matrix.h5"),
        "image_path": os.path.join(base_path, "spatial", "tissue_hires_image.png"),
        "scale_factors_path": os.path.join(base_path, "spatial", "scalefactors_json.json"),
        "geojson_path": os.path.join(base_path, "cell_segmentations.geojson"),
        "sample_name": sample_name
    }

print("Saving zarr files")
for sample_key, paths in samples.items():
    create_zarr(count_matrix_path=paths["count_matrix_path"],
                image_path=paths["image_path"],
                scale_factors_path=paths["scale_factors_path"],
                geojson_path=paths["geojson_path"],
                sample_name=paths["sample_name"])

# Cleanup
del samples, samples_base_paths, sample_key, paths
gc.collect()

print(f"The files are being saved to: {os.getcwd()}")
#%% md
# # Load The Spatial Data Object and Its Components
# 
#%%
%%time
# Loading the zarr files
visium_hd_zarr_paths = {
    "MDT_AP_0102": "./MDT_AP_0102",
    "MDT_AP_1166": "./MDT_AP_1166"
}

# Loading samples into a dictionary
sdatas = []
for key, path in visium_hd_zarr_paths.items():
    sdata = spd.read_zarr(path)

    for table in sdata.tables.values():
        table.var_names_make_unique()
        table.obs["sample"] = key

    sdatas.append(sdata)
    del sdata, table
    gc.collect()

concatenated_sdata = spd.concatenate(sdatas, concatenate_tables=True)

concatenated_sdata.write("concatenated_sdata", overwrite=True)

del concatenated_sdata, sdatas, visium_hd_zarr_paths, key, path
gc.collect()

concatenated_sdata = spd.read_zarr("concatenated_sdata")

print("---------------------------------")
print(concatenated_sdata)


#%% md
# # Quality Control and Filtering
#%%
adata = concatenated_sdata["segmentation_counts"] # we link the AnnData Table in the SpatialData object to the variable adata to make the code easier to read
adata
#%%
# Add mitochondrial gene calculation for QC
adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None)


# Visualization for QC
sc.pl.violin(adata=adata, keys=["log1p_total_counts"], stripplot=False, inner="box",show=False, groupby="sample",)
plt.title("Total UMI by Sample")
plt.axhline(y=4, color='r', linestyle='-')
plt.axhline(y=8.5, color='r', linestyle='-')
plt.title("Total UMI by Sample")
plt.show()

sc.pl.violin(adata=adata, keys=["log1p_n_genes_by_counts"], groupby="sample", stripplot=False, inner="box",show=False)
plt.title("Total Genes by Sample")
plt.show()

sc.pl.violin(adata=adata, keys=["log1p_total_counts_mt"], groupby="sample", stripplot=False, inner="box",show=False)
plt.title("Mitochondrial Genes by Sample")
plt.show()

plt.close('all')
#%%
# Estimating the cut off
min_counts = np.expm1(4).astype("int")
max_counts = np.expm1(8.5).astype("int")

# Filtering genes and cells
sc.pp.filter_genes(adata, min_cells=25)
sc.pp.filter_cells(adata, min_counts=min_counts)
sc.pp.filter_cells(adata, max_counts=max_counts)

# Visualization for QC
sc.pl.violin(adata=adata, keys=["log1p_total_counts"], stripplot=False, inner="box",show=False, groupby="sample",)
plt.title("Total UMI by Sample")
plt.axhline(y=4, color='r', linestyle='-')
plt.axhline(y=8.5, color='r', linestyle='-')
plt.title("Total UMI by Sample")
plt.show()

sc.pl.violin(adata=adata, keys=["log1p_n_genes_by_counts"], groupby="sample", stripplot=False, inner="box",show=False)
plt.title("Total Genes by Sample")
plt.show()

sc.pl.violin(adata=adata, keys=["log1p_total_counts_mt"], groupby="sample", stripplot=False, inner="box",show=False)
plt.title("Mitochondrial Genes by Sample")
plt.show()

plt.close('all')

# storing filtered counts
adata.layers["filtered_counts"] = adata.X.copy()


del max_counts, min_counts
gc.collect()

#%% md
# # Data Normalization and Dimensionality Reduction
#%%
sc.pp.normalize_total(adata, target_sum = None)
sc.pp.log1p(adata)
sc.tl.pca(adata)

adata.write("preprocessed_adata.h5ad")

# Elbow plot
sc.pl.pca_variance_ratio(adata, log=True,n_pcs=50)


#%% md
# # Clustering and UMAP Visualization
#%%
# neighborhood and clustering resolution
RES = 1 # clustering resolution
NEIGHBORS = 50  # number of neighbors

MIN_DIST=0.5 #default 0.5
SPREAD=2 #default 1

sc.pp.neighbors(adata, n_neighbors=NEIGHBORS, use_rep="X_pca", metric="manhattan")
                                                                        #"manhattan": Sum of absolute differences of coordinates.
                                                                        #"euclidean" : Straight-line distance.
                                                                        #"cosine": Cosine distance/similarity.
                                                                        #"correlation": 1 - correlation coefficient (Pearson).
sc.tl.leiden(adata, flavor="igraph", key_added="clusters", resolution=RES,random_state=0)

# To ensure that the results are reproducible we are going to reorder the clusters by size.
adata.obs['orig_clusters'] = adata.obs['clusters']
clusters = adata.obs['clusters'].astype(int)

# Count cells per cluster
cluster_sizes = clusters.value_counts().sort_values(ascending=False)

# Create mapping: old cluster ID → new ordered ID
cluster_order = {old: new for new, old in enumerate(cluster_sizes.index)}

# Relabel clusters in adata
adata.obs['clusters'] = clusters.map(cluster_order).astype(str)

# Set random_state for reproducible UMAP
sc.tl.umap(adata,min_dist=MIN_DIST, spread=SPREAD, random_state=0)

# Plot UMAP
sc.pl.umap(adata, color=["clusters"], title="UMAP by Clusters")
sc.pl.umap(adata, color=["sample"], title="UMAP by Sample")

# Cell distribution across clusters
sample_names = adata.obs["sample"].unique()
plt.imshow(pd.crosstab(adata.obs["sample"], adata.obs["clusters"]), cmap='hot', interpolation='nearest')
plt.title("Cell Distribution Across Clusters")
plt.xlabel("Cluster")
plt.yticks(range(len(sample_names)), sample_names)
plt.show()
plt.close('all')

del RES, NEIGHBORS, MIN_DIST, SPREAD
gc.collect()
#%% md
# # Batch Correction
#%%
# neighborhood and clustering resolution
RES = 1 # clustering resolution
NEIGHBORS = 50  # number of neighbors

MIN_DIST=0.5 #default 0.5
SPREAD=2 #default 1

# Performing batch correction
adata_harmony = concatenated_sdata["segmentation_counts"].copy()
sce.pp.harmony_integrate(adata_harmony, key="sample", basis="X_pca",max_iter_harmony=20)

# Copying the harmony PCA embedding results
adata_harmony.obsm["X_pca_orig"] = adata_harmony.obsm["X_pca"]
adata_harmony.obsm["X_pca"] = adata_harmony.obsm["X_pca_harmony"]
adata_harmony.obs["cluster_orig"] =  adata_harmony.obs["clusters"]

sc.pp.neighbors(adata_harmony, n_neighbors=NEIGHBORS, use_rep="X_pca",metric="manhattan")
sc.tl.leiden(adata_harmony, flavor="igraph",key_added="harmony_clusters", resolution=RES,random_state=0)
adata_harmony.obs["clusters"] =  adata_harmony.obs["harmony_clusters"]

# Set random_state for reproducible UMAP
sc.tl.umap(adata_harmony,min_dist=MIN_DIST, spread=SPREAD, random_state=0)

# Plot UMAP
sc.pl.umap(adata_harmony, color=["clusters"], title="Harmony Corrected UMAP by Clusters")
sc.pl.umap(adata_harmony, color=["sample"], title="Harmony Corrected UMAP by Sample")

# Cell distribution across clusters
sample_names = adata_harmony.obs["sample"].unique()
plt.imshow(pd.crosstab(adata_harmony.obs["sample"], adata_harmony.obs["clusters"]), cmap='hot', interpolation='nearest')
plt.title("Cell Distribution Across Clusters")
plt.xlabel("Cluster")
plt.yticks(range(len(sample_names)), sample_names)
plt.show()
plt.close('all')


# adds the harmony results in the analysis by overwriting the AnnData table in the SpatialData object below.
#concatenated_sdata["segmentation_counts"] = adata_harmony

del adata_harmony, RES, NEIGHBORS, MIN_DIST, SPREAD
gc.collect()
#%% md
# # Spatial Visualization of Clusters
#%%
image_elements = list(concatenated_sdata.images.keys())
shape_elements = list(concatenated_sdata.shapes.keys())

# We are going to create a bounding box to crop the data to the capture area.
extents = []

for i in range(len(image_elements)):
    extent =  spd.get_extent(concatenated_sdata,elements=[shape_elements[i]],coordinate_system='downscale_to_hires')
    extents.append(extent)

# Plotting
if len(image_elements) != len(shape_elements):
    print("Check the spatial data to make sure that for every image there is a shape")
else:
    for i in range(len(image_elements)):
        print("Plotting: "+ image_elements[i])
        title=image_elements[i].replace("_hires_tissue_image","")
        crop0(concatenated_sdata,crs="downscale_to_hires",bbox=extents[i]).pl.render_images(image_elements[i]).pl.render_shapes(shape_elements[i],color="clusters").pl.show(coordinate_systems="downscale_to_hires", title=title)
#%% md
# # *Marker Gene Identification and Cluster Annotation
#%%
marker_genes_g3mb = {
    # Tumor compartments
    "Tumor_StemLike": [
        "SOX2", "HES1",
        "OLIG2", "ASCL1",
        "PROM1", "NES"
    ],
    "Tumor_Proliferating_MYC": [
        "MYC", "MYCN",
        "MKI67", "TOP2A", "PCNA",
        "CCNB1", "CDK1"
    ],
    "Tumor_Differentiated_GNlike": [
        "NEUROD1", "NEUROD2",
        "PAX6", "ZIC1", "ZIC2",
        "MAP2", "DCX"
    ],

    # Tumor microenvironment (TME)
    "Endothelial": [
        "PECAM1", "VWF", "CLDN5", "CD34","ACE", "CLDN5","EMCN","PECAM1","EGFL7","ABCB1"
    ],
    "Pericytes": [
        "PDGFRB", "RGS5", "CSPG4", "ACTA2","COL4A1","DES","MGP","MYL9"
    ],
    "Microglia_Macrophages": [
      "SLC1A3","SPP1","PLXDC2","SLCO2B1","C1QA"
    ],
    "Astrocytes": [
        "GFAP", "AQP4", "S100B", "ALDH1L1"
    ],
    "Oligodendrocytes": [
        "QKI","ST18","PCDH9","PDE4B","ERBB4","ENPP2"
    ],
    "T_Cells": [
        "ANXA2","CD4", "CD8A","AQP3",
    ],
}


# Dotplot for endothelial, pericytes, microglial, T cells, glial progenitors, dying cells, malignant cells
sc.pl.dotplot(
    adata=concatenated_sdata["segmentation_counts"],
    var_names=marker_genes_g3mb,
    groupby="clusters",
    standard_scale="var",
    title="Markers for G3 MB malignant cells"
)




#%%
marker_genes_rl_cells = {

    # RL_VZ (Rhombic Lip - Ventricular Zone)
    "RL_VZ": [
        "SOX6",
        "TCF7L2",
        "OTX2",
        "SUZ12",
        "EZH2",
        "NEUROD1",
        "PAX6",
        "BARHL1",
        "MEIS2",
        "WLS",
    ],

    # RL_SVZ (Rhombic Lip - Subventricular Zone)
    "RL_SVZ": [
        "SOX6",
        "TCF7L2",
        "SUZ12",
        "EZH2",
        "BARHL1",
        "CTCF",
        "EOMES",
        "UNCX",
        "OTX2",
     "NR2F1","HES6","ID1","BRCA1"
    ],

    # Early_UBCs (Early Unipolar Brush Cell)
    "Early_UBCs": [
        "EOMES",
        "UNCX",
        "LMX1A",

    ],

    # Late_UBCs (Late Unipolar Brush Cell)
    "Late_UBCs": [
        "PPARA",
        "NRF1",
        "ZBTB37",
        "RELN"
    ],

    # GCP (Granule Cell Precursor)
    "GCP": [

        "PPARA",
        "NRF1",
        "MSI2",
        "LUZP2",
        "ZCCHC14",
        "DPYD",
        "DCC",
        "RBFOX3"
    ],

    # Early_GN (Early Granule Neuron)
    "Early_GN": [
        "ZEB1",
        "TAF1",
        "MEIS1",
        "MSI2",
        "FOXO3",
        "TBL1XR1",
        "LUZP2",
        "ZCCHC14",
        "ZMAT4",
        "DCC",
        "RBFOX3",
        "PARM1"
    ],

    # GN (Granule Neuron)
    "GN": [
        "ZEB1",
        "TAF1",
        "MEIS1",
        "MSI2",
        "FOXO3",
        "TBL1XR1",
        "ZCCHC14",
        "ZMAT4"
    ]
}

# Dotplot for RL VZ and RL SVZ cells specifically
sc.pl.dotplot(
    adata=concatenated_sdata["segmentation_counts"],
    var_names=marker_genes_rl_cells,
    groupby="clusters",
    standard_scale="var",
    title="Markers for RL-VZ and RL-SVZ cells"
)
#%%
adata.var_names[adata.var_names.str.contains("CLDN5", case=False)]
vmarker_genes_rl_cells = {
    "RL_VZ": [ ],  # Rhombic Lip - Ventricular Zone markers
    "RL_SVZ": []     # Rhombic Lip - Subventricular Zone markers
}
#%%

# Obtain cluster-specific marker genes
sc.tl.rank_genes_groups(adata = concatenated_sdata["segmentation_counts"], groupby="clusters", method="wilcoxon")

# Explicitly compute the dendrogram before plotting
sc.tl.dendrogram(concatenated_sdata["segmentation_counts"], groupby="clusters")

# Now the dotplot will use the precomputed dendrogram
sc.pl.rank_genes_groups_dotplot(adata = concatenated_sdata["segmentation_counts"], groupby="clusters", standard_scale="var", n_genes=5)

# Save marker genes
df_marker_genes = sc.get.rank_genes_groups_df(adata = concatenated_sdata["segmentation_counts"], group = None, pval_cutoff=0.05)
df_marker_genes.to_csv("marker_genes_pval.csv")


# Initialize a dictionary to store top 25 genes per cluster
top25_genes_per_cluster = {}

# Iterate over each cluster/group in the ranked results
groups = concatenated_sdata["segmentation_counts"].uns['rank_genes_groups']['names'].dtype.names
for group in groups:
    # Changed the slice from :10 to :25
    top25_genes = concatenated_sdata["segmentation_counts"].uns['rank_genes_groups']['names'][group][:25]
    top25_genes_per_cluster[group] = top25_genes

# Convert to a DataFrame for saving (clusters as columns, genes as rows)
# Changed the DataFrame variable name for clarity
df_top25 = pd.DataFrame(top25_genes_per_cluster)

# Save to CSV
# Changed the output file name for clarity
df_top25.to_csv("top25_marker_genes_per_cluster.csv")
#%%
# Cluster annotation mapping
original_clusters = concatenated_sdata["segmentation_counts"].obs['clusters']

# Updated Annotations based on predicted cell types for Group 3 Medulloblastoma
cell_annotation = {
    '0': 'Neuronal Progenitor-like Cells',
    '1': 'Photoreceptor-like Tumor Cells',
    '2': 'Mature Neurons',
    '3': 'Mesenchymal / Stromal Cells',
    '4': 'Developing Neurons / Neural Precursors',
    '5': 'Vascular / Endothelial Cells',
    '6': 'Photoreceptor-like Tumor Cells',
    '7': 'Activated Macrophages / Immune Cells',
    '8': 'Fibroblasts / Vascular Mural Cells'
}

# Apply the mapping. This new_categories Series should have the same index as original_clusters.
new_categories = original_clusters.astype('string').map(cell_annotation)

# Assign to sdata_concatenate.
concatenated_sdata["segmentation_counts"].obs["grouped_clusters"] = new_categories.astype('category')

# Plotting with new grouped clusters
for i in range(len(image_elements)):
    print("Plotting: "+ image_elements[i])
    title = image_elements[i].replace("_hires_tissue_image","")
    crop0(concatenated_sdata, crs="downscale_to_hires", bbox=extents[i]).pl.render_images(image_elements[i]).pl.render_shapes(shape_elements[i], color="grouped_clusters").pl.show(coordinate_systems="downscale_to_hires", title=title)

#%% md
# # Annotation
#%%
import scanorama

# Assuming 'adata' is your segmentation counts AnnData with spatial spots
# Load your reference single-cell dataset with annotated cell types
ref_adata = sc.read_h5ad('./G3.h5ad')

adata.var_names_make_unique()
ref_adata.var_names_make_unique()

# Normalize and log-transform as needed here on both datasets...

# Integrate in place (modifies .obsm['X_scanorama'] on each AnnData)
scanorama.integrate_scanpy([ref_adata, adata], dimred=50)

# Now access integrated embeddings directly
print(ref_adata.obsm['X_scanorama'].shape)
print(adata.obsm['X_scanorama'].shape)

# Further analysis with integrated embeddings like neighbors and label transfer
sc.pp.neighbors(adata, use_rep='X_scanorama')
sc.tl.umap(adata)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ref_adata.obsm['X_scanorama'], ref_adata.obs['annotated.clusters'])
predicted_labels = knn.predict(adata.obsm['X_scanorama'])
adata.obs['predicted_cell_type'] = predicted_labels

adata.write('annotated_spatial_adata.h5ad')



#%%
# Assuming your annotated file is saved as 'annotated_spatial_adata.h5ad'

import spatialdata as sd
# Load your computationally annotated data
annotated_adata = sc.read_h5ad('./annotated_spatial_adata.h5ad')

concatenated_sdata["segmentation_counts"].obs['predicted_cell_type'] = annotated_adata.obs['predicted_cell_type']
# Cluster annotation mapping
# use the new 'predicted_cell_type' column for plotting
# The column 'predicted_cell_type' already exists and is a category, so no extra steps are needed.
# Assign the new column to a variable for plotting, or use it directly
concatenated_sdata["segmentation_counts"].obs["grouped_clusters"] = concatenated_sdata["segmentation_counts"].obs['predicted_cell_type']



# Plotting with new grouped clusters
for i in range(len(image_elements)):
  print("Plotting: "+ image_elements[i])
  title=image_elements[i].replace("_hires_tissue_image","")
  crop0(concatenated_sdata,crs="downscale_to_hires",bbox=extents[i]).pl.render_images(image_elements[i]).pl.render_shapes(shape_elements[i],color="grouped_clusters").pl.show(coordinate_systems="downscale_to_hires", title=title)
#%% md
# adjust for my samples
#%%
# For first sample MDT_AP_1166
spd.bounding_box_query(
        concatenated_sdata,
        min_coordinate=[2500, 1000],
        max_coordinate=[4500, 3000],
        axes=("x", "y"),
        target_coordinate_system='downscale_to_hires').pl.render_images("MDT_AP_1166_hires_tissue_image").pl.show(coordinate_systems='downscale_to_hires', title="MDT_AP_1166")

spd.bounding_box_query(
        concatenated_sdata,
        min_coordinate=[3000, 1500],
        max_coordinate=[4000, 2500],
        axes=("x", "y"),
        target_coordinate_system='downscale_to_hires').pl.render_images("MDT_AP_1166_hires_tissue_image").pl.render_shapes(shape_elements[1],color="grouped_clusters").pl.show(coordinate_systems='downscale_to_hires',title="MDT_AP_1166")
# For second sample MDT_AP_0102 (update coordinates and image key as needed)
spd.bounding_box_query(
        concatenated_sdata,
        min_coordinate=[2500, 1000],
        max_coordinate=[4500, 3000],
        axes=("x", "y"),
        target_coordinate_system='downscale_to_hires').pl.render_images("MDT_AP_0102_hires_tissue_image").pl.show(coordinate_systems='downscale_to_hires', title="MDT_AP_0102")

spd.bounding_box_query(
        concatenated_sdata,
        min_coordinate=[3000, 1500],
        max_coordinate=[4000, 2500],
        axes=("x", "y"),
        target_coordinate_system='downscale_to_hires').pl.render_images("MDT_AP_0102_hires_tissue_image").pl.render_shapes(shape_elements[1],color="grouped_clusters").pl.show(coordinate_systems='downscale_to_hires',title="MDT_AP_0102")

#%% md
#  # save
#  The clustering results can also be written to a `CSV` file for import into Loupe Browser (v9.0.0 or later).
#%%

for sample_name in concatenated_sdata["segmentation_counts"].obs['sample'].unique():
    # Filter the table for the current sample
    adata_sample = concatenated_sdata["segmentation_counts"][concatenated_sdata["segmentation_counts"].obs['sample'] == sample_name].copy()

    # Create a DataFrame
    df_output = pd.DataFrame({
        'Barcode': 'cellid_' + adata_sample.obs.index.str.split('cellid_').str[1],
        'Grouped_Annotation': adata_sample.obs['grouped_clusters']
    })

    # Save the results
    output_filename = f"{sample_name}_cell_clusters.csv"
    df_output.to_csv(output_filename, index=False)

    print(f"Saved {output_filename}")

del adata_sample, df_output, sample_name, output_filename
gc.collect()
#%% md
# # Differential Gene Expression Analysis
#%%
adata.obs.head()
concatenated_sdata
adata
concatenated_sdata
print("Head of the AnnData object's .obs DataFrame:")
print(adata.obs.head())
#%% md
# #Peri_Vs_Malignant
# 
#%%
import squidpy as sq


# --- USER INPUT SECTION ---
malignant_stem_cell_type = "Malignant_cells"
pericyte_cell_type = "Pericytes"
output_prefix = "Pericyte_Malignant_differential_expression"

try:
    # Check segmentation counts table present
    if "segmentation_counts" not in concatenated_sdata.tables:
        raise KeyError("The key 'segmentation_counts' was not found in your concatenated_sdata object.")

    adata = concatenated_sdata["segmentation_counts"]

    # Check cell type annotation present
    if 'predicted_cell_type' not in annotated_adata.obs:
        raise KeyError("The annotated AnnData object does not contain a 'predicted_cell_type' column.")

    # Assign predicted cell types
    adata.obs['predicted_cell_type'] = annotated_adata.obs['predicted_cell_type']

    # Fix sample names for merging with shapes keys
    adata.obs['sample_with_boundaries'] = adata.obs['sample'].astype(str) + '_cell_boundaries'

    print("AnnData object loaded and cell types merged successfully.")
    print(adata)

    print("Extracting spatial coordinates from `Shapes` element...")

    updated_adata_list = []

    for sample_name, shapes_gdf in concatenated_sdata.shapes.items():
        sample_adata = adata[adata.obs['sample_with_boundaries'] == sample_name].copy()

        shapes_gdf = shapes_gdf.copy()
        shapes_gdf.index = shapes_gdf.index.str.strip()
        sample_adata.obs.index = sample_adata.obs.index.str.strip()

        intersect_ids = sample_adata.obs.index.intersection(shapes_gdf.index)
        if len(intersect_ids) == 0:
            raise ValueError(f"No overlapping cell IDs found between AnnData and Shapes for sample {sample_name}")

        sample_adata = sample_adata[intersect_ids]
        shapes_gdf = shapes_gdf.loc[intersect_ids]

        centroid_coords = shapes_gdf['geometry'].apply(lambda geom: (geom.centroid.x, geom.centroid.y))
        coords_df = pd.DataFrame(centroid_coords.tolist(), columns=['x', 'y'], index=centroid_coords.index)

        sample_adata.obsm['spatial'] = coords_df.loc[sample_adata.obs.index].values

        updated_adata_list.append(sample_adata)

    adata_with_coords = sc.concat(updated_adata_list, join='inner', merge='same')

    if 'spatial' not in adata_with_coords.obsm:
        raise KeyError("Spatial coordinates could not be added to the AnnData object.")

    print("Spatial coordinates successfully added to adata.obsm['spatial'].")

    print("Building spatial graph...")
    sq.gr.spatial_neighbors(adata_with_coords, n_rings=1)
    print("Spatial graph built successfully.")

    adata_with_graph = adata_with_coords

    print("Constructing neighbor types list...")
    adata_with_graph.obs['neighbor_types'] = [
        adata_with_graph.obs.iloc[adata_with_graph.obsp['spatial_connectivities'][i].indices]['predicted_cell_type'].tolist()
        for i in range(adata_with_graph.n_obs)
    ]

    print("Classifying malignant stem cells into groups...")

    adata_with_graph.obs['group'] = 'Other'
    mask_near_pericytes = adata_with_graph.obs['neighbor_types'].apply(
        lambda neighbors: any(cell_type == pericyte_cell_type for cell_type in neighbors)
    ) & (adata_with_graph.obs['predicted_cell_type'] == malignant_stem_cell_type)
    adata_with_graph.obs.loc[mask_near_pericytes, 'group'] = 'Near_Pericytes'

    mask_only_near_mal_stem = adata_with_graph.obs['neighbor_types'].apply(
        lambda neighbors: all(cell_type == malignant_stem_cell_type for cell_type in neighbors)
    ) & (adata_with_graph.obs['predicted_cell_type'] == malignant_stem_cell_type)
    adata_with_graph.obs.loc[mask_only_near_mal_stem, 'group'] = 'Only_Near_Malignant_Stem_Cells'

    final_adata = adata_with_graph[adata_with_graph.obs['group'].isin(['Near_Pericytes', 'Only_Near_Malignant_Stem_Cells'])].copy()

    if final_adata.n_obs == 0:
        print("No cells found that match the criteria. Please check your cell type names.")
    else:
        print(f"Found {final_adata.n_obs} malignant stem cells for analysis.")
        print("Performing differential expression analysis...")
        sc.tl.rank_genes_groups(final_adata, groupby='group', method='wilcoxon')

        print("Saving results to separate CSV files based on groups...")
        df_near = sc.get.rank_genes_groups_df(final_adata, group='Near_Pericytes')
        df_near['direction'] = 'Upregulated in Near_Pericytes'
        df_only = sc.get.rank_genes_groups_df(final_adata, group='Only_Near_Malignant_Stem_Cells')
        df_only['direction'] = 'Upregulated in Only_Near_Malignant_Stem_Cells'

        df_near.to_csv(f"{output_prefix}_Near_Pericytes.csv", index=False)
        df_only.to_csv(f"{output_prefix}_Only_Near_Malignant_Stem_Cells.csv", index=False)

        print(f"Differential expression results saved to {output_prefix}_Near_Pericytes.csv and {output_prefix}_Only_Near_Malignant_Stem_Cells.csv.")

except KeyError as e:
    print(f"Error: A required key was not found. Details: {e}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


#%%
# Visualization

# Add spatial coordinates as a 2D embedding for Scanpy plotting
adata_with_graph.obsm['X_spatial'] = adata_with_graph.obsm['spatial']

# Heatmap of top differentially expressed genes
sc.pl.rank_genes_groups_heatmap(final_adata, n_genes=25, groupby='group')

# Dotplot of top DE genes
sc.pl.rank_genes_groups_dotplot(final_adata, n_genes=25, groupby='group')

# Violin plot for a few top genes
top_genes = final_adata.uns['rank_genes_groups']['names']['Near_Pericytes'][:1]
sc.pl.violin(final_adata, keys=top_genes, groupby='group')
sc.pl.spatial(adata_with_graph, color='group')
#%% md
# #Endo_Vs_Malignant
#%% md
# #EndoPeri_Vs_Malignant
#%% md
# # Network analysis
# 
#%%
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

# --- UPDATED USER INPUT SECTION ---
# These variables now reflect the new 'grouped_clusters' annotation names.
tumor_cell_type = "Photoreceptor-like Tumor Cells"
# Combining the vascular and mural cells into a single list for neighbor check
perivascular_cell_types = ["Vascular / Endothelial Cells", "Fibroblasts / Vascular Mural Cells"]
output_prefix = "Tumor_Perivascular_differential_expression"

try:
    # Check segmentation counts table present (Assuming concatenated_sdata is available)
    if "segmentation_counts" not in concatenated_sdata.tables:
        raise KeyError("The key 'segmentation_counts' was not found in your concatenated_sdata object.")

    adata = concatenated_sdata["segmentation_counts"]

    # Check the new cluster annotation key provided by the user
    if 'grouped_clusters' not in adata.obs:
         # This assumes the user ran the mapping logic *before* this script.
         # If not, attempt to create it using the provided logic (if possible, but safer to assume pre-run)
         # For robustness, we will assume the key exists, but provide a custom error message if not.
         raise KeyError("The 'grouped_clusters' column was not found in adata.obs. Please ensure you run the cell_annotation mapping logic before this script.")

    # --- UPDATED: Assign the new 'grouped_clusters' as the primary cell type annotation ---
    adata.obs['cell_annotation'] = adata.obs['grouped_clusters'].copy()


    # Fix sample names for merging with shapes keys
    adata.obs['sample_with_boundaries'] = adata.obs['sample'].astype(str) + '_cell_boundaries'

    print("AnnData object loaded and cell annotations merged successfully.")
    print(adata)

    print("Extracting spatial coordinates from `Shapes` element...")

    updated_adata_list = []

    for sample_name, shapes_gdf in concatenated_sdata.shapes.items():
        sample_adata = adata[adata.obs['sample_with_boundaries'] == sample_name].copy()

        shapes_gdf = shapes_gdf.copy()
        shapes_gdf.index = shapes_gdf.index.str.strip()
        sample_adata.obs.index = sample_adata.obs.index.str.strip()

        intersect_ids = sample_adata.obs.index.intersection(shapes_gdf.index)
        if len(intersect_ids) == 0:
            raise ValueError(f"No overlapping cell IDs found between AnnData and Shapes for sample {sample_name}")

        sample_adata = sample_adata[intersect_ids]
        shapes_gdf = shapes_gdf.loc[intersect_ids]

        centroid_coords = shapes_gdf['geometry'].apply(lambda geom: (geom.centroid.x, geom.centroid.y))
        coords_df = pd.DataFrame(centroid_coords.tolist(), columns=['x', 'y'], index=centroid_coords.index)

        # Ensure that the coordinates are correctly aligned with the AnnData index
        sample_adata.obsm['spatial'] = coords_df.loc[sample_adata.obs.index].values

        updated_adata_list.append(sample_adata)

    # Using 'inner' join ensures only cells present in all inputs are kept, 'merge='same'' keeps existing obs/var unchanged.
    adata_with_coords = sc.concat(updated_adata_list, join='inner', merge='same')

    if 'spatial' not in adata_with_coords.obsm:
        raise KeyError("Spatial coordinates could not be added to the AnnData object.")

    print("Spatial coordinates successfully added to adata.obsm['spatial'].")

    print("Building spatial graph...")
    # Using n_rings=1 ensures only immediate physical neighbors are connected
    sq.gr.spatial_neighbors(adata_with_coords, n_rings=1)
    print("Spatial graph built successfully.")

    adata_with_graph = adata_with_coords

    # --- ADDED: Full Cell-Cell Interaction Network Analysis ---
    print("\n--- Performing Cell-Cell Interaction Network Analysis (All Cell Types) ---")

    # Calculate the interaction matrix using the new 'cell_annotation' column
    sq.gr.interaction_matrix(
        adata_with_graph,
        cluster_key='cell_annotation',
    )

    # Visualization of the network interaction matrix
    print("Visualizing interaction matrix with hierarchical clustering...")
    # Create two side-by-side plots: one for frequency and one for significance (p-values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. Cell-Cell Interaction Frequency (Normalized) - CLUSTERED
    sq.pl.interaction_matrix(
        adata_with_graph,
        cluster_key='cell_annotation', # Use new key
        p_val_threshold=0.05,
        title='Cell-Cell Interaction Frequency (Clustered)',
        ax=ax1,
        show=False,
        cluster=True,
        method='ward'
    )

    # 2. Cell-Cell Interaction P-Values (FDR adjusted) - CLUSTERED
    try:
        sq.pl.interaction_matrix(
            adata_with_graph,
            cluster_key='cell_annotation', # Use new key
            p_val_threshold=0.05,
            title='Cell-Cell Interaction P-Values (Clustered)',
            val_to_plot='pvalues',
            ax=ax2,
            show=False,
            cluster=True,
            method='ward'
        )
    except Exception as e:
        print(f"Warning: Could not plot p-values, likely because they were not computed in this version of squidpy: {e}")
        ax2.set_title("P-Value Plot Failed (Missing P-Values)")


    plt.tight_layout()
    plt.show()

    print("Network analysis results saved to adata_with_graph.uns['interaction_matrix'] and related keys.")
    print("--------------------------------------------------------------------------------\n")
    # --- END Network Analysis Section ---


    print("Constructing neighbor types list...")
    # NOTE: The neighbor list construction here might be slow for very large datasets
    # but is necessary for the subsequent classification logic.
    adata_with_graph.obs['neighbor_types'] = [
        adata_with_graph.obs.iloc[adata_with_graph.obsp['spatial_connectivities'][i].indices]['cell_annotation'].tolist()
        for i in range(adata_with_graph.n_obs)
    ]

    print("Classifying tumor cells into groups based on their new annotations...")

    # Continue with the user's classification and DE logic
    adata_with_graph.obs['group'] = 'Other'

    # --- UPDATED: Mask for tumor cells near ANY perivascular cell type ---
    mask_near_perivascular = adata_with_graph.obs['neighbor_types'].apply(
        lambda neighbors: any(cell_type in perivascular_cell_types for cell_type in neighbors)
    ) & (adata_with_graph.obs['cell_annotation'] == tumor_cell_type)
    adata_with_graph.obs.loc[mask_near_perivascular, 'group'] = 'Near_Perivascular_Niche'

    # --- UPDATED: Mask for tumor cells only near other tumor cells ---
    mask_only_near_tumor = adata_with_graph.obs['neighbor_types'].apply(
        lambda neighbors: all(cell_type == tumor_cell_type for cell_type in neighbors)
    ) & (adata_with_graph.obs['cell_annotation'] == tumor_cell_type)
    adata_with_graph.obs.loc[mask_only_near_tumor, 'group'] = 'Only_Near_Tumor_Cells'

    # --- UPDATED: Spatial Visualization of the Groups (Robust Matplotlib Plot) ---
    print("\n--- Visualizing Spatial Distribution of Malignant Tumor Cell Groups ---")

    target_groups = ['Near_Perivascular_Niche', 'Only_Near_Tumor_Cells']
    plot_data = adata_with_graph[adata_with_graph.obs['group'].isin(target_groups)].copy()

    if plot_data.n_obs == 0:
        print("Warning: Cannot visualize, no target cells found in the defined groups. Check your cell names.")
    else:
        # Define colors manually for the two groups
        group_colors = {
            'Near_Perivascular_Niche': 'darkorange', # New color for clarity
            'Only_Near_Tumor_Cells': 'mediumblue'   # New color for clarity
        }

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Spatial Distribution of Tumor Cells: Near Niche vs. Tumor-Only')
        ax.set_xlabel('Spatial X Coordinate')
        ax.set_ylabel('Spatial Y Coordinate')

        for group, color in group_colors.items():
            subset = plot_data[plot_data.obs['group'] == group]
            if subset.n_obs > 0:
                # Use coordinates directly from obsm['spatial']
                ax.scatter(
                    subset.obsm['spatial'][:, 0],
                    subset.obsm['spatial'][:, 1],
                    label=group,
                    s=35, # Spot size
                    c=color
                )

        # Optional: Plot 'Other' cells in a light gray for context
        other_cells = adata_with_graph[adata_with_graph.obs['group'] == 'Other']
        if other_cells.n_obs > 0:
             ax.scatter(
                other_cells.obsm['spatial'][:, 0],
                other_cells.obsm['spatial'][:, 1],
                label='Other Cells',
                s=10,
                c='lightgray',
                alpha=0.3,
                zorder=0 # Ensures they are plotted behind the target groups
            )

        ax.legend(loc='best', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis() # Often required for spatial data to match image orientation
        plt.show()

    print("Spatial visualization complete. Check the plot for the distribution of the two tumor cell groups.")
    print("--------------------------------------------------------------------------------\n")
    # --- END Visualization Section ---

    final_adata = adata_with_graph[adata_with_graph.obs['group'].isin(['Near_Perivascular_Niche', 'Only_Near_Tumor_Cells'])].copy()

    if final_adata.n_obs == 0:
        print("No cells found that match the criteria. Please check your cell type names.")
    else:
        print(f"Found {final_adata.n_obs} tumor cells for differential expression analysis.")
        print("Performing differential expression analysis...")
        sc.tl.rank_genes_groups(final_adata, groupby='group', method='wilcoxon')

        print("Saving results to separate CSV files based on groups...")
        df_near = sc.get.rank_genes_groups_df(final_adata, group='Near_Perivascular_Niche')
        df_near['direction'] = 'Upregulated in Near_Perivascular_Niche'
        df_only = sc.get.rank_genes_groups_df(final_adata, group='Only_Near_Tumor_Cells')
        df_only['direction'] = 'Upregulated in Only_Near_Tumor_Cells'

        df_near.to_csv(f"{output_prefix}_Near_Perivascular_Niche.csv", index=False)
        df_only.to_csv(f"{output_prefix}_Only_Near_Tumor_Cells.csv", index=False)

        print(f"Differential expression results saved to {output_prefix}_Near_Perivascular_Niche.csv and {output_prefix}_Only_Near_Tumor_Cells.csv.")

except KeyError as e:
    print(f"Error: A required key was not found. Details: {e}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration (MUST match the prefix used in the analysis script) ---
# UPDATED: Using the new output prefix
output_prefix = "Tumor_Perivascular_differential_expression"

def plot_volcano(df_near, df_only, near_label, only_label):
    """
    Creates a comparative Volcano Plot from two differential expression DataFrames.

    Parameters:
    - df_near (pd.DataFrame): DE results for genes upregulated in the 'Near' group.
    - df_only (pd.DataFrame): DE results for genes upregulated in the 'Only' group.
    - near_label (str): Label for the 'Near' group (e.g., 'Near_Perivascular_Niche').
    - only_label (str): Label for the 'Only' group (e.g., 'Only_Near_Tumor_Cells').
    """

    # Use adjusted p-values if available, falling back to raw p-values
    p_val_col = 'pvals_adj' if 'pvals_adj' in df_near.columns else 'pvals'

    # 1. Prepare the 'Only' group data for the left side of the plot
    # Flip the logFC for the 'Only' group to represent upregulation in 'Only' relative to 'Near'.
    df_only_plot = df_only.copy()
    df_only_plot['logfoldchanges'] = -df_only_plot['logfoldchanges']
    df_only_plot['Group'] = only_label

    # 2. Prepare the 'Near' group data for the right side of the plot
    df_near_plot = df_near.copy()
    df_near_plot['Group'] = near_label

    # 3. Combine DataFrames
    combined_df = pd.concat([df_near_plot, df_only_plot], ignore_index=True)

    # Calculate key metrics using the chosen p-value column
    combined_df['neg_log10_pvals'] = -np.log10(combined_df[p_val_col])

    # 4. Define thresholds and significance
    PVAL_THRESHOLD = 0.05
    LOGFC_THRESHOLD = 0.5

    # Determine significance for coloring
    is_up_near = (combined_df['logfoldchanges'] > LOGFC_THRESHOLD) & (combined_df[p_val_col] < PVAL_THRESHOLD) & (combined_df['Group'] == near_label)
    is_up_only = (combined_df['logfoldchanges'] < -LOGFC_THRESHOLD) & (combined_df[p_val_col] < PVAL_THRESHOLD) & (combined_df['Group'] == only_label)

    # Initialize all points as 'Not Significant'
    combined_df['Significance'] = 'Not Significant'
    combined_df.loc[is_up_near, 'Significance'] = f'Upregulated in {near_label}'
    combined_df.loc[is_up_only, 'Significance'] = f'Upregulated in {only_label}'

    # Define the improved palette
    improved_palette = {
        f'Upregulated in {near_label}': '#FF7F00', # Vibrant Orange
        f'Upregulated in {only_label}': '#00A3A3', # Strong Teal
        'Not Significant': '#CCCCCC' # Light Gray
    }

    # 5. Create the Volcano Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort by significance so significant points plot on top of gray points
    combined_df.sort_values(by='Significance', key=lambda x: x != 'Not Significant', inplace=True)

    sns.scatterplot(
        data=combined_df,
        x='logfoldchanges',
        y='neg_log10_pvals',
        hue='Significance',
        palette=improved_palette,
        size='neg_log10_pvals', # Size by significance
        sizes=(15, 150),
        alpha=0.7,
        linewidth=0,
        legend='full',
        ax=ax
    )

    # Add reference lines
    ax.axvline(x=LOGFC_THRESHOLD, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=-LOGFC_THRESHOLD, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=-np.log10(PVAL_THRESHOLD), color='black', linestyle=':', alpha=0.7)

    # --- Title and Axis Labels ---
    plt.title(f'Volcano Plot: DE in Tumor Cells ({near_label} vs. {only_label})', fontsize=16, pad=20)
    plt.xlabel('$\log_2 \text{{Fold Change}}$ (Positive $\\to$ Near Perivascular Niche)', fontsize=14)
    plt.ylabel(f'$-\log_{{10}}$ ({p_val_col.replace("pvals", "p-value")})', fontsize=14)

    # --- Label the top 5 most significant genes from each side ---
    top_near = combined_df[is_up_near].sort_values(by='neg_log10_pvals', ascending=False).head(5)
    top_only = combined_df[is_up_only].sort_values(by='neg_log10_pvals', ascending=False).head(5)

    for i in range(len(top_near)):
        row = top_near.iloc[i]
        plt.annotate(row['names'], (row['logfoldchanges'] + 0.05, row['neg_log10_pvals']), # Offset slightly right
                     fontsize=9, color=improved_palette[f'Upregulated in {near_label}'],
                     alpha=1.0, ha='left')

    for i in range(len(top_only)):
        row = top_only.iloc[i]
        plt.annotate(row['names'], (row['logfoldchanges'] - 0.05, row['neg_log10_pvals']), # Offset slightly left
                     fontsize=9, color=improved_palette[f'Upregulated in {only_label}'],
                     alpha=1.0, ha='right')

    # Update legend to include line descriptions
    handles, labels = ax.get_legend_handles_labels()

    # Add manual descriptions for the lines
    line_handles = [plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.5),
                    plt.Line2D([0], [0], color='black', linestyle=':', alpha=0.7)]

    # FIX: Replacing \le with <= string to avoid ParseFatalException
    line_labels = [f'$|\log_2 \text{{FC}}| >= {LOGFC_THRESHOLD}$ Threshold',
                   f'{p_val_col.replace("pvals", "p-value")} <= {PVAL_THRESHOLD}$ Threshold']

    plt.legend(
        handles=handles + line_handles,
        labels=labels + line_labels,
        title='Enrichment', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10
    )

    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to fit legend
    plt.show()

# --- Main Execution ---
try:
    # UPDATED: Use the new file names based on the latest analysis script
    file_near = f"{output_prefix}_Near_Perivascular_Niche.csv"
    file_only = f"{output_prefix}_Only_Near_Tumor_Cells.csv"

    # Load the DE results
    df_near_results = pd.read_csv(file_near)
    df_only_results = pd.read_csv(file_only)

    print(f"Loaded {len(df_near_results)} genes upregulated in Near_Perivascular_Niche.")
    print(f"Loaded {len(df_only_results)} genes upregulated in Only_Near_Tumor_Cells.")

    # Call the plotting function with the NEW labels
    plot_volcano(
        df_near_results,
        df_only_results,
        near_label='Near_Perivascular_Niche',
        only_label='Only_Near_Tumor_Cells'
    )

except FileNotFoundError:
    print(f"Error: One or both output files were not found. Please ensure the files are in the current directory and match the prefix '{output_prefix}'.")
except Exception as e:
    print(f"An unexpected error occurred during visualization: {e}")

#%%

import anndata as ad # Adding this for type hint context, though not strictly used in the logic below

# --- USER INPUT SECTION ---
# NOTE: concatenated_sdata and annotated_adata must be defined and available
# in your environment for this script to run.
malignant_stem_cell_type = "Malignant_cells"
pericyte_cell_type = "Pericytes"
output_prefix = "Pericyte_Malignant_differential_expression"

try:
    # Check segmentation counts table present (Assuming concatenated_sdata is available)
    if "segmentation_counts" not in concatenated_sdata.tables:
        raise KeyError("The key 'segmentation_counts' was not found in your concatenated_sdata object.")

    adata = concatenated_sdata["segmentation_counts"]

    # Check cell type annotation present (Assuming annotated_adata is available)
    if 'predicted_cell_type' not in annotated_adata.obs:
        raise KeyError("The annotated AnnData object does not contain a 'predicted_cell_type' column.")

    # Assign predicted cell types
    adata.obs['predicted_cell_type'] = annotated_adata.obs['predicted_cell_type']

    # Fix sample names for merging with shapes keys
    adata.obs['sample_with_boundaries'] = adata.obs['sample'].astype(str) + '_cell_boundaries'

    print("AnnData object loaded and cell types merged successfully.")
    print(adata)

    print("Extracting spatial coordinates from `Shapes` element...")

    updated_adata_list = []

    for sample_name, shapes_gdf in concatenated_sdata.shapes.items():
        sample_adata = adata[adata.obs['sample_with_boundaries'] == sample_name].copy()

        shapes_gdf = shapes_gdf.copy()
        shapes_gdf.index = shapes_gdf.index.str.strip()
        sample_adata.obs.index = sample_adata.obs.index.str.strip()

        intersect_ids = sample_adata.obs.index.intersection(shapes_gdf.index)
        if len(intersect_ids) == 0:
            raise ValueError(f"No overlapping cell IDs found between AnnData and Shapes for sample {sample_name}")

        sample_adata = sample_adata[intersect_ids]
        shapes_gdf = shapes_gdf.loc[intersect_ids]

        centroid_coords = shapes_gdf['geometry'].apply(lambda geom: (geom.centroid.x, geom.centroid.y))
        coords_df = pd.DataFrame(centroid_coords.tolist(), columns=['x', 'y'], index=centroid_coords.index)

        # Ensure that the coordinates are correctly aligned with the AnnData index
        sample_adata.obsm['spatial'] = coords_df.loc[sample_adata.obs.index].values

        updated_adata_list.append(sample_adata)

    # Using 'inner' join ensures only cells present in all inputs are kept, 'merge='same'' keeps existing obs/var unchanged.
    adata_with_coords = sc.concat(updated_adata_list, join='inner', merge='same')

    if 'spatial' not in adata_with_coords.obsm:
        raise KeyError("Spatial coordinates could not be added to the AnnData object.")

    print("Spatial coordinates successfully added to adata.obsm['spatial'].")

    print("Building spatial graph...")
    # Using n_rings=1 ensures only immediate physical neighbors are connected
    sq.gr.spatial_neighbors(adata_with_coords, n_rings=1)
    print("Spatial graph built successfully.")

    adata_with_graph = adata_with_coords

    # --- Full Cell-Cell Interaction Network Analysis (for overall context) ---
    print("\n--- Performing Cell-Cell Interaction Network Analysis (All Cell Types) ---")

    # Calculate the interaction matrix. Removed 'n_perms' to maintain compatibility.
    sq.gr.interaction_matrix(
        adata_with_graph,
        cluster_key='predicted_cell_type', # Use the main cell type annotation column
    )

    # Visualization of the network interaction matrix
    print("Visualizing interaction matrix...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    sq.pl.interaction_matrix(
        adata_with_graph,
        cluster_key='predicted_cell_type',
        p_val_threshold=0.05,
        title='Cell-Cell Interaction Frequency (Normalized)',
        ax=ax1,
        show=False
    )

    try:
        sq.pl.interaction_matrix(
            adata_with_graph,
            cluster_key='predicted_cell_type',
            p_val_threshold=0.05,
            title='Cell-Cell Interaction P-Values (FDR adjusted)',
            val_to_plot='pvalues',
            ax=ax2,
            show=False
        )
    except Exception as e:
        # Catch error if p-values were not computed (common when n_perms is missing)
        print(f"Warning: Could not plot p-values: {e}")
        ax2.set_title("P-Value Plot Failed (Missing P-Values)")


    plt.tight_layout()
    plt.show()

    print("Network analysis results saved to adata_with_graph.uns['interaction_matrix'] and related keys.")
    print("--------------------------------------------------------------------------------\n")
    # --- END Network Analysis Section ---


    print("Constructing neighbor types list...")
    adata_with_graph.obs['neighbor_types'] = [
        adata_with_graph.obs.iloc[adata_with_graph.obsp['spatial_connectivities'][i].indices]['predicted_cell_type'].tolist()
        for i in range(adata_with_graph.n_obs)
    ]

    print("Classifying malignant stem cells into groups...")

    # Classification Logic
    adata_with_graph.obs['group'] = 'Other'
    mask_near_pericytes = adata_with_graph.obs['neighbor_types'].apply(
        lambda neighbors: any(cell_type == pericyte_cell_type for cell_type in neighbors)
    ) & (adata_with_graph.obs['predicted_cell_type'] == malignant_stem_cell_type)
    adata_with_graph.obs.loc[mask_near_pericytes, 'group'] = 'Near_Pericytes'

    mask_only_near_mal_stem = adata_with_graph.obs['neighbor_types'].apply(
        lambda neighbors: all(cell_type == malignant_stem_cell_type for cell_type in neighbors)
    ) & (adata_with_graph.obs['predicted_cell_type'] == malignant_stem_cell_type)
    adata_with_graph.obs.loc[mask_only_near_mal_stem, 'group'] = 'Only_Near_Malignant_Stem_Cells'

    # --- UPDATED: Spatial Visualization of the Groups (Robust Matplotlib Plot) ---
    print("\n--- Visualizing Spatial Distribution of Malignant Stem Cell Groups ---")

    # Subset data to only include the cells we want to highlight
    target_groups = ['Near_Pericytes', 'Only_Near_Malignant_Stem_Cells']
    plot_data = adata_with_graph[adata_with_graph.obs['group'].isin(target_groups)].copy()

    if plot_data.n_obs == 0:
        print("Warning: Cannot visualize, no target cells found in the defined groups.")
    else:
        # Define colors manually for the two groups
        group_colors = {
            'Near_Pericytes': 'red',
            'Only_Near_Malignant_Stem_Cells': 'blue'
        }

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Spatial Distribution of Near_Pericytes vs. Only_Near_Malignant_Stem_Cells')
        ax.set_xlabel('Spatial X Coordinate')
        ax.set_ylabel('Spatial Y Coordinate')

        for group, color in group_colors.items():
            subset = plot_data[plot_data.obs['group'] == group]
            if subset.n_obs > 0:
                # Use coordinates directly from obsm['spatial']
                ax.scatter(
                    subset.obsm['spatial'][:, 0],
                    subset.obsm['spatial'][:, 1],
                    label=group,
                    s=35, # Spot size
                    c=color
                )

        # Optional: Plot 'Other' cells in a light gray for context
        other_cells = adata_with_graph[adata_with_graph.obs['group'] == 'Other']
        if other_cells.n_obs > 0:
             ax.scatter(
                other_cells.obsm['spatial'][:, 0],
                other_cells.obsm['spatial'][:, 1],
                label='Other Cells',
                s=10,
                c='lightgray',
                alpha=0.5,
                zorder=0 # Ensures they are plotted behind the target groups
            )

        ax.legend(loc='best', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis() # Often required for spatial data to match image orientation
        plt.show()

    print("Spatial visualization complete. Check the plot for the distribution of the two malignant stem cell groups.")
    print("--------------------------------------------------------------------------------\n")
    # --- END Visualization Section ---

    final_adata = adata_with_graph[adata_with_graph.obs['group'].isin(['Near_Pericytes', 'Only_Near_Malignant_Stem_Cells'])].copy()

    if final_adata.n_obs == 0:
        print("No cells found that match the criteria. Please check your cell type names.")
    else:
        print(f"Found {final_adata.n_obs} malignant stem cells for differential expression analysis.")
        print("Performing differential expression analysis...")
        sc.tl.rank_genes_groups(final_adata, groupby='group', method='wilcoxon')

        print("Saving results to separate CSV files based on groups...")
        df_near = sc.get.rank_genes_groups_df(final_adata, group='Near_Pericytes')
        df_near['direction'] = 'Upregulated in Near_Pericytes'
        df_only = sc.get.rank_genes_groups_df(final_adata, group='Only_Near_Malignant_Stem_Cells')
        df_only['direction'] = 'Upregulated in Only_Near_Malignant_Stem_Cells'

        df_near.to_csv(f"{output_prefix}_Near_Pericytes.csv", index=False)
        df_only.to_csv(f"{output_prefix}_Only_Near_Malignant_Stem_Cells.csv", index=False)

        print(f"Differential expression results saved to {output_prefix}_Near_Pericytes.csv and {output_prefix}_Only_Near_Malignant_Stem_Cells.csv.")

except KeyError as e:
    print(f"Error: A required key was not found. Details: {e}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration (MUST match the prefix used in the analysis script) ---
output_prefix = "Pericyte_Malignant_differential_expression"

def plot_volcano(df_near, df_only, near_label, only_label):
    """
    Creates a comparative Volcano Plot from two differential expression DataFrames.

    Parameters:
    - df_near (pd.DataFrame): DE results for genes upregulated in the 'Near' group.
    - df_only (pd.DataFrame): DE results for genes upregulated in the 'Only' group.
    - near_label (str): Label for the 'Near' group (e.g., 'Near_Pericytes').
    - only_label (str): Label for the 'Only' group (e.g., 'Only_Near_Malignant_Stem_Cells').
    """

    # 1. Prepare the 'Only' group data for the left side of the plot
    # Since the comparison was done relative to the 'rest' group, we combine
    # the two lists and flip the logFC for the 'Only' group to represent
    # upregulation in 'Only' relative to 'Near'.
    df_only_plot = df_only.copy()
    df_only_plot['logfoldchanges'] = -df_only_plot['logfoldchanges']
    df_only_plot['Group'] = only_label

    # 2. Prepare the 'Near' group data for the right side of the plot
    df_near_plot = df_near.copy()
    df_near_plot['Group'] = near_label

    # 3. Combine DataFrames
    combined_df = pd.concat([df_near_plot, df_only_plot], ignore_index=True)
    combined_df['neg_log10_pvals'] = -np.log10(combined_df['pvals'])

    # 4. Filter for plotting: only keep genes with significant p-values and decent fold change
    # Define thresholds (adjust these based on desired stringency)
    PVAL_THRESHOLD = 0.05
    LOGFC_THRESHOLD = 0.5

    # Determine significance for coloring
    is_up_near = (combined_df['logfoldchanges'] > LOGFC_THRESHOLD) & (combined_df['pvals'] < PVAL_THRESHOLD) & (combined_df['Group'] == near_label)
    is_up_only = (combined_df['logfoldchanges'] < -LOGFC_THRESHOLD) & (combined_df['pvals'] < PVAL_THRESHOLD) & (combined_df['Group'] == only_label)

    combined_df['Significance'] = 'Not Significant'
    combined_df.loc[is_up_near, 'Significance'] = f'Upregulated in {near_label}'
    combined_df.loc[is_up_only, 'Significance'] = f'Upregulated in {only_label}'

    # 5. Create the Volcano Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=combined_df,
        x='logfoldchanges',
        y='neg_log10_pvals',
        hue='Significance',
        palette={
            f'Upregulated in {near_label}': 'darkred',
            f'Upregulated in {only_label}': 'darkblue',
            'Not Significant': 'lightgray'
        },
        size='neg_log10_pvals',
        sizes=(10, 100),
        alpha=0.6,
        linewidth=0,
        legend='full'
    )

    # Add reference lines
    plt.axvline(x=LOGFC_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-LOGFC_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=-np.log10(PVAL_THRESHOLD), color='gray', linestyle='--', alpha=0.5)

    plt.title(f'Volcano Plot: {near_label} vs. {only_label}', fontsize=16)
    plt.xlabel('$\log_2$ Fold Change (Positive = Near Pericytes)', fontsize=14)
    plt.ylabel('$-\log_{10}$ (p-value)', fontsize=14)

    # Label the top 5 most significant genes from each side
    top_near = combined_df[is_up_near].sort_values(by='neg_log10_pvals', ascending=False).head(5)
    top_only = combined_df[is_up_only].sort_values(by='neg_log10_pvals', ascending=False).head(5)

    for i in range(len(top_near)):
        row = top_near.iloc[i]
        plt.annotate(row['names'], (row['logfoldchanges'], row['neg_log10_pvals']),
                     fontsize=9, alpha=0.8, color='darkred', ha='left')

    for i in range(len(top_only)):
        row = top_only.iloc[i]
        plt.annotate(row['names'], (row['logfoldchanges'], row['neg_log10_pvals']),
                     fontsize=9, alpha=0.8, color='darkblue', ha='right')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='Enrichment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# --- Main Execution ---
try:
    file_near = f"{output_prefix}_Near_Pericytes.csv"
    file_only = f"{output_prefix}_Only_Near_Malignant_Stem_Cells.csv"

    # Load the DE results
    df_near_results = pd.read_csv(file_near)
    df_only_results = pd.read_csv(file_only)

    print(f"Loaded {len(df_near_results)} genes upregulated in Near_Pericytes.")
    print(f"Loaded {len(df_only_results)} genes upregulated in Only_Near_Malignant_Stem_Cells.")

    # Call the plotting function
    plot_volcano(
        df_near_results,
        df_only_results,
        near_label='Near_Pericytes',
        only_label='Only_Near_Malignant_Stem_Cells'
    )

except FileNotFoundError:
    print(f"Error: One or both output files were not found. Please ensure the files are in the current directory and match the prefix '{output_prefix}'.")
except Exception as e:
    print(f"An unexpected error occurred during visualization: {e}")
