import pandas as pd
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering,DBSCAN,Birch
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np

data1 = pd.read_csv("Dataset_Siamak_Yousefi_PLOS_One_2018.csv", delimiter=",").to_numpy()
data1[data1=="Enable"] = 1.0
print(data1.shape)

data2 = pd.read_csv("Cluster_Labels_Siamak_Yousefi_PLOS_One_2018.csv", delimiter=",").to_numpy()
print(data2.shape)

y_true = data2[:,2] - 1
y_true = y_true.astype(int)

# pca = PCA(n_components=25,svd_solver="full",iterated_power=5)
# x = pca.fit_transform(data1[:,2:])

x = TSNE(n_components=3).fit_transform(data1[:,2:])


print(x.shape)

# Kmeans Clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
pred = KMeans(n_clusters=4,max_iter=30000,random_state=0).fit_predict(x)
fs = f1_score(y_true, pred, average='weighted')
print("Kmeans F1 Score:",fs)

# Spectral Clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
pred = SpectralClustering(n_clusters=4).fit_predict(x)
fs = f1_score(y_true, pred, average='weighted')
print("Spectral F1 Score:",fs)

# Agglomerative Clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
pred = AgglomerativeClustering(n_clusters=4).fit_predict(x)
fs = f1_score(y_true, pred, average='weighted')
print("Agglomerative F1 Score:",fs)

# Density Based
# needs work https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
pred = DBSCAN(eps=1, min_samples=2, ).fit_predict(x)
fs = f1_score(y_true, pred, average='weighted')
print("Density Based F1 Score:",fs)

# Birch
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch
pred = Birch(n_clusters=4).fit_predict(x)
fs = f1_score(y_true, pred, average='weighted')
print("Birch F1 Score:",fs)
