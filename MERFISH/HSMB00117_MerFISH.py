import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

import dask
import numpy as np
dask.config.set({"dataframe.query-planning": True})


# Load the datasets
hsmb = sc.read_h5ad("/Users/ricardodgonzalez/Desktop/Python/HSMB0117_MERFISH.h5ad")
g3_mb = sc.read_h5ad("/Users/ricardodgonzalez/Desktop/Python/G3_MB.h5ad")
hsmb = hsmb.copy()  # Convert to a full AnnData object
g3_mb = g3_mb.copy()
# Read the metadata CSV
g3metadata = pd.read_csv("/Users/ricardodgonzalez/Desktop/MERFISH/G3_MB_metadata.csv", index_col=0)
#keep metadata columns and overwrite existing ones:
g3_mb.obs = g3metadata.join(g3_mb.obs.drop(columns=g3metadata.columns), how='right')
# Read the predicted CSV
predicted = pd.read_csv("/Users/ricardodgonzalez/Desktop/MERFISH/query.csv", index_col=0)
hsmbmeta = pd.read_csv("/Users/ricardodgonzalez/PycharmProjects/PythonProject/hsmb_metadata.csv", index_col=0)
# Ensure the index of both DataFrames are strings to avoid type mismatches
predicted.index = predicted.index.astype(str)
hsmbmeta.index = hsmbmeta.index.astype(str)
hsmb.obs.index = hsmb.obs.index.astype(str)
# Combine the DataFrames based on the shared index (EntityID)
combined_data = hsmbmeta.join(predicted, how="left")  # Use 'left' to keep all rows from hsmbmeta
# Extract the 'predicted.id' column from the combined data
predicted_id_series = combined_data['predicted.id']
# Add the 'predicted.id' column to hsmb.obs, matching on the index
hsmb.obs['predicted.id'] = predicted_id_series
# Optional: Print the first few rows of hsmb.obs to verify
print(hsmb.obs.head())
# Optional: Save the updated AnnData object
hsmb.write("/Users/ricardodgonzalez/Desktop/Python/HSMB0117_MERFISH_predicted.h5ad")



'''g3_mb      '''
sc.pp.pca(g3_mb)# Compute PCA if not done #print(g3_mb.obsm.keys())
sc.pp.neighbors(g3_mb,use_rep='X')# Compute the KNN graph #print(g3_mb.uns.keys())
sc.tl.leiden(g3_mb, flavor="igraph", n_iterations=-1, directed=False)#Perform Leiden clustering #sc.tl.leiden(g3_mb)
sc.tl.umap(g3_mb) #Use UMAP or t-SNE for dimensionality reduction (if not already done): #sc.tl.tsne(adata)
sc.pl.umap(g3_mb, color='annotated.clusters', legend_loc='on data')#Plot the Leiden clusters # sc.pl.tsne(adata, color='leiden', legend_loc='on data')
sc.pl.umap(g3_mb, color=['annotated.clusters'], size=30, legend_fontsize=10, show=False)#Plot the Leiden clusters
plt.gcf().set_size_inches(18, 10)
# Save manually
plt.savefig("/Users/ricardodgonzalez/Desktop/G3Maria_Leiden_clusters.png", dpi=300, bbox_inches="tight")
plt.show()



'''  hsmb       '''
print(hsmb.obsm.keys())#Check if PCA Exists
sc.pp.pca(hsmb)# Compute PCA if not done
print(hsmb.uns.keys())  # Should contain 'neighbors' if it's already computed #Check if Neighbors Already Exist
sc.pp.neighbors(hsmb,use_rep='X')# Compute the KNN graph
sc.tl.leiden(hsmb, flavor="igraph", n_iterations=2, directed=False)#Perform Leiden clustering #sc.tl.leiden(g3_mb)
sc.tl.umap(hsmb) #sc.tl.tsne(adata)#Use UMAP or t-SNE for dimensionality reduction (if not already done):
sc.pl.umap(hsmb, color=['leiden'], legend_loc='on data', size=5, legend_fontsize=10, show=False,save="ChenLab_Leiden_clusters.png")#Plot the Leiden clusters
plt.gcf().set_size_inches(18, 10)
plt.show()

#Plot the annotated Leiden clusters
sc.pl.umap(hsmb, color=['celltype'], size=5, legend_fontsize=10, show=False,save="ChenLab_celltype_clusters.png")
plt.gcf().set_size_inches(18, 10)
plt.show()
#Plot the annotated Leiden clusters
sc.pl.umap(hsmb, color=['predicted.id'], size=5, legend_fontsize=10, show=False,save="predictedID_Leiden_clusters.tiff")
plt.gcf().set_size_inches(18, 10)
plt.show()


#Plot the Leiden clusters in spatial coordinates
sc.pl.spatial(hsmb, color="predicted.id", spot_size=20,save="predictedID_clusters.tiff")

# Subset to exclude "Malignant_cells"
subset_no_malignant = hsmb[hsmb.obs['predicted.id'] != 'Malignant_cells', :]
print(subset_no_malignant) #Verify the subset
# Plot the Leiden clusters in spatial coordinates without malignant cells
sc.pl.spatial(subset_no_malignant, color="predicted.id", spot_size=25, save="no_malignant_clusters.tiff")



# Subset for endothelial cells and malignant cells
subset_hsmb = hsmb[hsmb.obs['predicted.id'].isin(['Endothelial_cells', 'Malignant_cells']), :]
print(subset_hsmb)  # Verify the subset
# Custom color mapping and transparency
color_map = {'Endothelial_cells': 'blue', 'Malignant_cells': 'salmon','Microglia': 'green'}  # Define your preferred colors
alpha_values = [1 if cell_type == 'Endothelial_cells' else 0.25 for cell_type in subset_hsmb.obs['predicted.id']] #make malignant cells more transparent
# Plot with custom colors and transparency
sc.pl.spatial(
    subset_hsmb,
    color="predicted.id",
    spot_size=30,
    save="endo_clusters_custom.tiff",
    palette=color_map,
    alpha=alpha_values)
# Optional: Further highlight endothelial cells by plotting them on top
endothelial_subset = subset_hsmb[subset_hsmb.obs['predicted.id'] == 'Endothelial_cells']
sc.pl.spatial(
    endothelial_subset,
    color="predicted.id",
    spot_size=35,  # Slightly larger for emphasis
    save="endo_highlight.png",
    palette={'Endothelial_cells': 'blue'},
    alpha=1,
    show=False, #prevent from showing on its own
    ax = plt.gca() #plot on top of the existing plot
)

# Subset for microglia cells and stem cells
Micro_subset_hsmb = hsmb[hsmb.obs['predicted.id'].isin(['Microglia', 'Malignant_cells']), :]
print(Micro_subset_hsmb)# Verify the subset
sc.pl.spatial(Micro_subset_hsmb, color="predicted.id", spot_size=20,save="microsubset_clusters.png")#Plot the Leiden clusters in spatial coordinates
alpha_values = [1 if cell_type == 'Microglia' else 0.25 for cell_type in Micro_subset_hsmb.obs['predicted.id']] #make malignant cells more transparent
# Plot with custom colors and transparency
sc.pl.spatial(
    Micro_subset_hsmb,
    color="predicted.id",
    spot_size=30,
    save="micro_clusters_custom.tiff",
    palette=color_map,
    alpha=alpha_values)
# Optional: Further highlight endothelial cells by plotting them on top
micro_subset = subset_hsmb[subset_hsmb.obs['predicted.id'] == 'Microglia']
sc.pl.spatial(
    micro_subset,
    color="predicted.id",
    spot_size=35,  # Slightly larger for emphasis
    save="micro_highlight.png",
    palette={'Microglia': 'purple'},
    alpha=1,
    show=False, #prevent from showing on its own
    ax = plt.gca() )#plot on top of the existing plot



# Subset for endothelial, malignant, and microglia cells
subset_hsmb = hsmb[hsmb.obs['predicted.id'].isin(['Endothelial_cells', 'Malignant_cells', 'Microglia']), :]
print(subset_hsmb)  # Verify the subset
# Custom color mapping and transparency
color_map = {'Endothelial_cells': 'blue', 'Malignant_cells': 'salmon', 'Microglia': 'green'}  # Define your preferred colors
alpha_values = [
    1 if cell_type == 'Endothelial_cells'
    else 0.25 if cell_type == 'Malignant_cells'
    else 0.7  # Microglia transparency
    for cell_type in subset_hsmb.obs['predicted.id']]
# Plot with custom colors and transparency
sc.pl.spatial(
    subset_hsmb,
    color="predicted.id",
    spot_size=30,
    save="endo_malignant_microglia_clusters.tiff",
    palette=color_map,
    alpha=alpha_values)
plt.show()  # Show the plot.




# Check if the PRTG gene exists in the AnnData object
if 'PRTG' in hsmb.var_names:
    print("PRTG gene is present in hsmb.var_names")
    # Get the index of the PRTG gene
    prtg_index = np.where(hsmb.var_names == 'PRTG')[0][0]
    # Get the PRTG expression values from hsmb.X, handling both sparse and dense cases
    if hasattr(hsmb.X, 'toarray'):  # Check if it's a sparse matrix
        prtg_expression = hsmb.X[:, prtg_index].toarray().flatten()
    else:  # It's a dense NumPy array
        prtg_expression = hsmb.X[:, prtg_index]

    # Add the PRTG expression values to hsmb.obs, renaming the column
    hsmb.obs['PRTG_expression'] = prtg_expression

    # Find cells with PRTG gene expression (PRTG > 0)
    prtg_expressing_cells = hsmb[hsmb.obs['PRTG_expression'] > 0]
    print(f"Number of cells with PRTG expression: {prtg_expressing_cells.n_obs}")

    # Plot the spatial locations of PRTG-expressing cells
    sc.pl.spatial(
        prtg_expressing_cells,
        color="PRTG_expression",  # Use the renamed column
        spot_size=30,
        save="PRTG_expressing_cells_spatial.tiff",
    )
    # Optionally, you can also color by 'predicted.id' to see the cell types expressing PRTG
    sc.pl.spatial(
        prtg_expressing_cells,
        color="predicted.id",
        spot_size=30,
        save="PRTG_expressing_cells_predicted_id_spatial.tiff",
    )
else:
    print("PRTG gene is NOT present in hsmb.var_names")
plt.show()  # Show the plots.




# Check if the CD34 gene exists in the AnnData object
if 'CD34' in hsmb.var_names:
    print("CD34 gene is present in hsmb.var_names")
    # Get the index of the CD34 gene
    cd34_index = np.where(hsmb.var_names == 'CD34')[0][0]

    # Get the CD34 expression values from hsmb.X, handling both sparse and dense cases
    if hasattr(hsmb.X, 'toarray'):
        cd34_expression = hsmb.X[:, cd34_index].toarray().flatten()
    else:
        cd34_expression = hsmb.X[:, cd34_index]

    # Add the CD34 expression values to hsmb.obs, renaming the column
    hsmb.obs['CD34_expression'] = cd34_expression

    # Find cells with CD34 gene expression (CD34 > 0)
    cd34_expressing_cells = hsmb[hsmb.obs['CD34_expression'] > 0]

    print(f"Number of cells with CD34 expression: {cd34_expressing_cells.n_obs}")

    # Plot the spatial locations of CD34-expressing cells, colored by expression
    sc.pl.spatial(
        cd34_expressing_cells,
        spot_size=30,
        save="CD34_expression_spatial.tiff",
        color="CD34_expression",
    )

    # Optionally, color by 'predicted.id' to see cell types expressing CD34
    sc.pl.spatial(
        cd34_expressing_cells,
        color="predicted.id",
        spot_size=30,
        save="CD34_expressing_cells_predicted_id_spatial.tiff",
    )

else: print("CD34 gene is NOT present in hsmb.var_names")

plt.show() #show plots.

# Function to plot expression for a given gene
def plot_gene_expression(gene_name):
    if gene_name in hsmb.var_names:
        print(f"{gene_name} gene is present in hsmb.var_names")
        # Get the index of the gene
        gene_index = np.where(hsmb.var_names == gene_name)[0][0]

        # Get the gene expression values from hsmb.X, handling both sparse and dense cases
        if hasattr(hsmb.X, 'toarray'):
            gene_expression = hsmb.X[:, gene_index].toarray().flatten()
        else:
            gene_expression = hsmb.X[:, gene_index]

        # Add the gene expression values to hsmb.obs, renaming the column
        hsmb.obs[f'{gene_name}_expression'] = gene_expression

        # Find cells with gene expression (gene > 0)
        gene_expressing_cells = hsmb[hsmb.obs[f'{gene_name}_expression'] > 0]

        print(f"Number of cells with {gene_name} expression: {gene_expressing_cells.n_obs}")

        # Plot the spatial locations of gene-expressing cells, colored by expression
        sc.pl.spatial(
            gene_expressing_cells,
            spot_size=30,
            save=f"{gene_name}_expression_spatial.tiff",
            color=f"{gene_name}_expression",
        )

        # Optionally, color by 'predicted.id' to see cell types expressing the gene
        sc.pl.spatial(
            gene_expressing_cells,
            color="predicted.id",
            spot_size=30,
            save=f"{gene_name}_expressing_cells_predicted_id_spatial.tiff",
        )

    else:
        print(f"{gene_name} gene is NOT present in hsmb.var_names")

# Plot OTX2 expression
plot_gene_expression("OTX2")

# Plot MYC expression
plot_gene_expression("MYC")

# Plot MYC expression
plot_gene_expression("CBFA2T2")
plot_gene_expression("TBR1")
plt.show() #show plots.


# Check if both CD34 and PRTG genes exist in the AnnData object
if 'CD34' in hsmb.var_names and 'PRTG' in hsmb.var_names:
    print("CD34 and PRTG genes are present in hsmb.var_names")

    # Get the index of the CD34 gene
    cd34_index = np.where(hsmb.var_names == 'CD34')[0][0]
    # Get the index of the PRTG gene
    prtg_index = np.where(hsmb.var_names == 'PRTG')[0][0]

    # Get the CD34 and PRTG expression values
    if hasattr(hsmb.X, 'toarray'):
        cd34_expression = hsmb.X[:, cd34_index].toarray().flatten()
        prtg_expression = hsmb.X[:, prtg_index].toarray().flatten()
    else:
        cd34_expression = hsmb.X[:, cd34_index]
        prtg_expression = hsmb.X[:, prtg_index]

    # Add the expression values to hsmb.obs
    hsmb.obs['CD34_expression'] = cd34_expression
    hsmb.obs['PRTG_expression'] = prtg_expression

    # Find cells expressing CD34 and PRTG
    cd34_expressing_cells = hsmb[hsmb.obs['CD34_expression'] > 0]
    prtg_expressing_cells = hsmb[hsmb.obs['PRTG_expression'] > 0]

    # Create a new column to mark cells based on expression
    hsmb.obs['expression_type'] = 'None'
    hsmb.obs.loc[cd34_expressing_cells.obs.index, 'expression_type'] = 'CD34+'
    hsmb.obs.loc[prtg_expressing_cells.obs.index, 'expression_type'] = 'PRTG+'
    hsmb.obs.loc[cd34_expressing_cells.obs.index.intersection(prtg_expressing_cells.obs.index), 'expression_type'] = 'CD34+ & PRTG+'

    # Remove the none type cells.
    hsmb_filtered = hsmb[hsmb.obs['expression_type'] != 'None']

    # Plot the spatial locations of CD34 and PRTG expressing cells
    sc.pl.spatial(
        hsmb_filtered,
        spot_size=30,
        save="CD34_PRTG_expressing_cells_spatial.tiff",
        color='expression_type',
        palette = {'CD34+': 'red', 'PRTG+': 'green', 'CD34+ & PRTG+': 'purple'},
    )

    # Plot the spatial locations of Endothelial cells.
    sc.pl.spatial(
        hsmb[hsmb.obs['predicted.id'] == 'Endothelial'],
        spot_size=30,
        save = "Endothelial_cells_spatial.tiff",
        color = 'predicted.id',
        palette = {'Endothelial': 'blue'}
    )

else:
    if 'CD34' not in hsmb.var_names:
        print("CD34 gene is NOT present in hsmb.var_names")
    if 'PRTG' not in hsmb.var_names:
        print("PRTG gene is NOT present in hsmb.var_names")

plt.show() #show plots.











