import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn import preprocessing
from sklearn.decomposition import PCA
import hdbscan
import pickle
import sys
import argparse

def sumPairs(l):
	summedLengths = []
	for i in range(0,len(l)):
		if i % 2 == 1:
			summedLengths.append(l[i-1]+l[i])
	if len(l) % 2 == 1:
		summedLengths.append(l[len(l)-1])
	return summedLengths


def get_parser():
	parser = argparse.ArgumentParser(
		description="Provides clustering probabilities for new USV segments"
	)
	parser.add_argument('outFile', help='file to write to')
	parser.add_argument('--fullProb', help='file to write full probabilty data to', default=-1)
	return parser

parser = get_parser()
args = parser.parse_args()

fileWriteName = args.outFile

scaledDataArray = np.load("workingDir/representationData/precompute_pca32_cls16_mean_pooled/train.npy")

with open("processingObjects/scaler.txt","rb") as scalerFile:
	scaler = pickle.load(scalerFile)
#scaledDataArray = scaler.transform(scaledDataArray)

with open("processingObjects/pca.txt","rb") as pcaFile:
	pca = pickle.load(pcaFile)
#scaledDataArray = pca.transform(scaledDataArray)

print("Scaling and PCA Applied")
print(scaledDataArray.shape)

print("Processing HDBSCAN Clustering")
with open("processingObjects/HDBSCANClusterer.txt","rb") as clustererFile:
	clusterer = pickle.load(clustererFile)

result = hdbscan.prediction.membership_vector(clusterer, scaledDataArray)
print(np.argmax(result,axis=1))

predicted = np.argmax(result,axis=1)

with open(fileWriteName, "w") as f:
	f.write("Segment Cluster Codes (lengths in ms below) \n")

#also save a full probabilities result file
if args.fullProb != -1:
	np.savetxt(args.fullProb, result, delimiter=",", fmt="%10.10f")


#reverse-engineer lengths of segments (each step is 10ms)
with open("workingDir/representationData/CLUS16/train.src") as internalClusterAssignments:
	featureStart = 0
	for line in internalClusterAssignments.readlines():
		assignments = list(map(int,line.split()))
		duplicateLengths = []
		newVal = assignments[0]
		length = 1
		for i in assignments[1:]:
			if (i == newVal):
				length = length+1
			else:
				duplicateLengths.append(length)
				length = 1
				newVal = i
		duplicateLengths.append(length)

		summedLengths = sumPairs(duplicateLengths)
		summedLengths = sumPairs(summedLengths)
		
		str1 = ""
		str2 = ""
		for i in range(featureStart,featureStart+len(summedLengths)):
			str1 = str1+str(predicted[i])+","
			str2 = str2+str(summedLengths[i-featureStart]*10)+","
		featureStart = featureStart + len(summedLengths)
		print(str(featureStart))
		with open(fileWriteName, "a") as f:
			f.write(str1+"\n")
			f.write(str2+"\n")
			f.write("\n")



