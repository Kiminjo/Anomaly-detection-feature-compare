"""
PCA is conducted before TSNE 
Follow this article 
https://medium.com/rapids-ai/tsne-with-gpus-hours-to-seconds-9d9c17c941db
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca(features: np.array,
        dim_size: int = 50
        ):
    pca = PCA(n_components=dim_size)
    pcaed_features = pca.fit_transform(features)
    return pcaed_features

def tsne(features: np.array,
         dim_size: int = 2
         ):
    tsne = TSNE(n_components=dim_size)
    reduced_features = tsne.fit_transform(features)
    return reduced_features