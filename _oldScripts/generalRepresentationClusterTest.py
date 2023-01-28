import os
import shutil
import random
import torch
import fairseq
import soundfile
import numpy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator

outputDirFileTest = 'testOutputs/clusteringOutputs/7505-83618-0021.txt'

f = open(outputDirFileTest, "r")
lineString = f.readline()

print(lineString)

splitString = lineString.split()
print(splitString[0])
realnumbers = []

for num in splitString:
	#from scientific notation to decimal
	realnumbers.append(float(num))

print(str(realnumbers[0]))

#k means time
clusterData = numpy.array(realnumbers).reshape(-1,1)

distortions = []
K = range(25,76)
for k in K:
	kmeanModel = KMeans(k).fit(clusterData)
	distortions.append(sum(numpy.min(cdist(clusterData, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / clusterData.shape[0])
	print("K: " + str(k+1))
	print(str(distortions[k-26]))
	
kneedle = KneeLocator(K, distortions, S=1.0, curve = "convex", direction = "decreasing", interp_method="polynomial")
print(kneedle.knee)
