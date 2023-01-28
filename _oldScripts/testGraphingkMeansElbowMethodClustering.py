import os
import shutil
import random
import torch
import fairseq
import soundfile
import numpy
from sklearn.cluster import KMeans
import kmeans1d
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import matplotlib.pyplot as plt

outputDir = 'testOutputs/clusteringOutputs'
fileWriteName = 'clusteringResult.txt'

realnumbers = []
for i in range(0,512):
	realnumbers.append([])

#get all the values from all the files at the appropriate layers in a single 2d array
j = 1
for root, dirs, files in os.walk(outputDir):
	for file in files:
		if file.endswith(".txt"):
			print("Reading file: " + file + " (" + str(j)+"/"+str(len(files))+")")
			j+=1
			i = 0
			f = open(os.path.join(root,file), "r")
			for lineString in f:
				splitString = lineString.split()
				for num in splitString:
					realnumbers[i].append(float(num))
				i+=1

with open(fileWriteName, "a") as file_object:
	file_object.write("LAYER: OPTIMAL CLUSTERS")
	file_object.write("\n")

#k means time
i = 0
for i in range(0, len(realnumbers), 50):
	
	clusterData = numpy.array(realnumbers[i]).reshape(-1,1)

	distortions = []
	K = range(1,101)
	for k in K:
		#kmeanModel = KMeans(k).fit(clusterData)
		#distortions.append(sum(numpy.min(cdist(clusterData, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / clusterData.shape[0])
		clusters, centroids = kmeans1d.cluster(realnumbers[i], k)
		distortions.append(sum(numpy.min(cdist(clusterData, numpy.array(centroids).reshape(-1,1), 'euclidean'), axis=1)) / clusterData.shape[0])
		print("K: " + str(k))
		print(str(distortions[k-1]))
	
	kneedle = KneeLocator(K, distortions, S=1.0, curve = "convex", direction = "decreasing", interp_method="polynomial")
	print("Optimal Clusters:")
	print(kneedle.knee)

	with open(fileWriteName, "a") as file_object:
		file_object.write(str(i) + ": " + str(kneedle.knee))
		file_object.write("\n")

	with open("_graphingData.txt", "a") as file_object:
		file_object.write(str(i))
		file_object.write("\n")
		for k in K:
			file_object.write(str(k))
			file_object.write(",")
		file_object.write("\n")
		for value in distortions:
			file_object.write(str(value))
			file_object.write(",")
		file_object.write("\n")
		file_object.write("\n")

	plt.plot(K, distortions)
	plt.xlabel("K (K-means clusters)")
	plt.ylabel("Sum of euclidean distance from centroids")
	plt.title("Layer "+str(i))
	plt.show()