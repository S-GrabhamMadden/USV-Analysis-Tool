import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
import sys

fileWriteName = 'largeWorkDir/clusterMetrics.txt'
fileReadName = 'largeWorkDir/outputData/precompute_pca32_cls32_mean_pooled/train.npy'

scaledDataArray = np.load(fileReadName)

scaler = preprocessing.StandardScaler().fit(scaledDataArray)
scaledDataArray = scaler.transform(scaledDataArray)
pca = PCA(n_components = 8, svd_solver="full")
pca.fit(scaledDataArray)
scaledDataArray = pca.transform(scaledDataArray)

with open(fileWriteName, "a") as file_object:
	file_object.write("K,Inertia,Silhouette Score,Calinski-Harabasz Index,Davies-Bouldin Index")
	file_object.write("\n")

print("Scaling/PCA Applied")
print(scaledDataArray.shape)

for i in range(2,51):
	print("---Processing K: "+str(i)+"---")
	print("Iteration: 1")

	kmeans = KMeans(n_clusters=i, random_state=np.random.randint(1234)).fit(scaledDataArray)
	inertia = kmeans.inertia_
	sil = silhouette_score(scaledDataArray, kmeans.labels_)
	chi = calinski_harabasz_score(scaledDataArray, kmeans.labels_)
	dbi = davies_bouldin_score(scaledDataArray, kmeans.labels_)
	clusters = kmeans.cluster_centers_

	for j in range (2,4):
		print("Iteration: "+str(j))
		kmeans = KMeans(n_clusters=i, random_state=np.random.randint(1234)).fit(scaledDataArray)
		if silhouette_score(scaledDataArray, kmeans.labels_) > sil:
			inertia = kmeans.inertia_
			sil = silhouette_score(scaledDataArray, kmeans.labels_)
			chi = calinski_harabasz_score(scaledDataArray, kmeans.labels_)
			dbi = davies_bouldin_score(scaledDataArray, kmeans.labels_)
			clusters = kmeans.cluster_centers_
    
	with open(fileWriteName, "a") as file_object:
		file_object.write(str(i)+","+str(inertia)+","+str(sil)+","+str(chi)+","+str(dbi))
		file_object.write("\n")

	centroidCoordFile = "largeWorkDir/centroids/" + str(i) + "centroids.txt"
	np.savetxt(centroidCoordFile, clusters)