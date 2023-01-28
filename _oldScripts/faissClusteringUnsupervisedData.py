import faiss
import numpy as np
from sklearn.metrics import silhouette_score
import sys

fileWriteName = 'testOutputs/unsupervisedClusteringResults/result.txt'
fileReadName = 'testOutputs/UnsupervisedTestOutput/precompute_pca512_cls128_mean_pooled/train.npy'
#gpuDeviceID = sys.argv[1]

class FaissKMeans:

	def __init__(self, n_k=2, n_init=10, max_iter=50):
		self.n_k = n_k
		self.n_init = n_init
		self.max_iter = max_iter
		self.kmeans = None
		self.centroids = None
		self.inertia = None
		self.silhouette = None
	
	def fit(self, X):
		self.kmeans = faiss.Kmeans(X.shape[1], self.n_k, verbose=True, nredo=3)
		#self.kmeans.gpu = True

		self.kmeans.seed = np.random.randint(1234) #make sure random initial centroids are different
		self.kmeans.max_points_per_centroid = X.shape[0] #some sources say necessary, won't hurt
		self.kmeans.niter = self.max_iter

		#res = faiss.StandardGpuResources()
		#cfg = faiss.GpuIndexFlatConfig()
		#cfg.useFloat16 = False
		#cfg.device = int(gpuDeviceID)
		#index = faiss.GpuIndexFlatL2(res, X.shape[1], cfg)

		self.kmeans.train(X)
		self.centroids = self.kmeans.centroids
		self.inertia = self.kmeans.obj[-1]
	
	def predict(self, X):
		assignedClusters = self.kmeans.index.search(X,1)[1]
		self.silhouette = silhouette_score(X, assignedClusters)
		return assignedClusters

dataArray = np.load(fileReadName)

for i in range(2,3):
    print(i)
    kmeansfaiss = FaissKMeans(n_k = i)
    kmeansfaiss.fit(dataArray)
    predictedClusters = kmeansfaiss.predict(dataArray)
    
    with open(fileWriteName, "a") as file_object:
        file_object.write("--K: " + str(i)+"--")
        file_object.write("\n")
        file_object.write("Inertia: " + str(kmeansfaiss.inertia))
        file_object.write("\n")
        file_object.write("Silhouette Score: " + str(kmeansfaiss.silhouette))
        file_object.write("\n")
        file_object.write("\n")