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

highest = 0
bestI = 0
bestJ = 0
scores = []

for i in range(2,21):
	for j in range(1,2):
		clusterer = hdbscan.HDBSCAN(min_cluster_size=i, min_samples=j, cluster_selection_method='leaf', metric='euclidean', gen_min_span_tree=True, prediction_data=True).fit(scaledDataArray)
		labels = clusterer.labels_
		print(str(scaledDataArray.shape))
		validityScore = clusterer.relative_validity_
		if (j == 1):
			scores.append(validityScore)
		if (validityScore > highest) and (8 <= labels.max()+1 <= 20):
			highest = validityScore
			bestI = i
			bestJ = j

print("---Results:---")
print(str(highest))
print(str(bestI))
print(str(bestJ))

print(scores)