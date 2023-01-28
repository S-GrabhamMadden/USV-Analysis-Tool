import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
import hdbscan
import pickle
import sys

fileWriteName = 'largeWorkDir/clusterMetrics.txt'
fileReadName = 'largeWorkDir/outputData/precompute_pca32_cls16_mean_pooled/train.npy'

scaledDataArray = np.load(fileReadName)

scaler = preprocessing.StandardScaler().fit(scaledDataArray)
#scaledDataArray = scaler.transform(scaledDataArray)
pca = PCA(n_components = 8, svd_solver="full")
pca.fit(scaledDataArray)
#scaledDataArray = pca.transform(scaledDataArray)
print("Scaling and/or PCA Applied")
print(scaledDataArray.shape)

print("---Processing HDBSCAN Clustering---")

clusterer = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=1, cluster_selection_method='leaf', metric='euclidean', gen_min_span_tree=True, prediction_data=True).fit(scaledDataArray)
labels = clusterer.labels_
print(str(scaledDataArray.shape))
validityScore = clusterer.relative_validity_

print("---Results:---")
print("DBCV-Based Validity Score: "+str(validityScore))
print("Total Clusters: " +str(labels.max()+1))
#print("Noise-Classified Data Points: "+str(np.count_nonzero(labels ==  -1))+" / " +str(len(labels)))

with open("processingObjects/HDBSCANClusterer.txt","wb") as clustererFile:
	pickle.dump(clusterer, clustererFile)

with open("processingObjects/scaler.txt","wb") as scalerFile:
	pickle.dump(scaler, scalerFile)

with open("processingObjects/pca.txt","wb") as pcaFile:
	pickle.dump(pca, pcaFile)