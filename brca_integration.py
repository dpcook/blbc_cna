import scanpy as sc
import scvi
from scvi.external import CellAssign
import scipy
import numpy as np
import pandas as pd
import anndata as ad

adata = sc.read_h5ad('/global/scratch/hpc3837/tnbc_cna/output/brca_cohort.h5ad')
adata.X = scipy.sparse.csr_matrix(adata.X)

### PREPROCESS ###
#Store full feature space for counts
adata.uns["all_counts"] = adata.X.copy()

#Preprocess
adata.layers["counts"] = adata.X.copy() # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    subset=True,
    batch_key="sample_id"
)

### INTEGRATION WITH SCVI ###
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="sample_id"
)

vae = scvi.model.SCVI(adata, n_layers=2, 
                      encode_covariates=True,
                      deeply_inject_covariates=False,
                      use_layer_norm="both",
                      use_batch_norm="none"
)

vae.train(max_epochs=400,
          early_stopping=True,
          early_stopping_monitor='elbo_validation',
          early_stopping_patience=10
)

adata.obsm["X_scVI"] = vae.get_latent_representation()
          
#Save model
vae.save("/global/scratch/hpc3837/tnbc_cna/output/scVI_model_brca",
          overwrite=True)
          
adata.write_h5ad(filename='/global/scratch/hpc3837/tnbc_cna/output/brca_cohort.h5ad', compression="gzip")
          
### ANNOTATION WITH CELLASSIGN ###
marker_gene_mat = pd.read_csv('/global/scratch/hpc3837/tnbc_cna/data/cellassign_markers.csv', index_col=0)

bdata = ad.AnnData(adata.uns['all_counts'].copy(),
                  obs = adata.obs.copy(),
                  var = adata.raw.var)
                  
lib_size = bdata.X.sum(1)
bdata.obs["size_factor"] = lib_size / np.mean(lib_size)   

bdata = bdata[:, marker_gene_mat.index].copy()

scvi.external.CellAssign.setup_anndata(bdata, size_factor_key="size_factor")

model = CellAssign(bdata, marker_gene_mat)
model.train()

predictions = model.predict()

adata.obs['celltype_pred'] = predictions.idxmax(axis=1).values

del bdata              

### SCANVI ###
vae_ref_scan = scvi.model.SCANVI.from_scvi_model(
    vae,
    unlabeled_category="Unknown",
    labels_key="celltype_pred",
)

vae_ref_scan.train(max_epochs = 800,
                   early_stopping=True, 
                   early_stopping_monitor="elbo_validation")
                                      
adata.obsm["X_scANVI"] = vae_ref_scan.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scANVI")
sc.tl.umap(adata)

# Savepoint
adata.write_h5ad(filename='/global/scratch/hpc3837/tnbc_cna/output/brca_cohort.h5ad', compression="gzip")

