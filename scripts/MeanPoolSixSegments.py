import argparse
import os
import os.path as osp
import numpy as np
import tqdm
import torch
import math
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.decomposition import PCA

from npy_append_array import NpyAppendArray

def get_parser():
	parser = argparse.ArgumentParser(
		description="mean pools representations by compressing uniform splits of the data"
	)
	parser.add_argument('source', help='directory with features')
	parser.add_argument('--split', help='which split to read', required=True)
	return parser

parser = get_parser()
args = parser.parse_args()
source_path = osp.join(args.source, args.split)

with open(source_path + ".lengths", "r") as lf:
	lengths = lf.readlines()

os.remove(source_path+".lengths")

features = np.load(source_path + ".npy", mmap_mode="r")

npaa = NpyAppendArray(source_path+"1.npy")

fsz = features.shape[-1]
newsize = math.ceil(features.shape[0]/ 6)
start = 0
with torch.no_grad():
	for length in tqdm.tqdm(lengths):
		length = int(length)
		end = start+length
		feats = features[start:end]
		start += length
		x = torch.from_numpy(feats).cuda()
		target_num = math.ceil(length / 6)

		rem = length % target_num
		if rem > 0:
			to_add = target_num - rem
			x = F.pad(x, [0, 0, 0, to_add])
			x[-to_add:] = x[-to_add - 1]

		x = x.view(target_num, -1, fsz)
		x = x.mean(dim=-2)

		npaa.append(x.cpu().numpy())
		del x

		with open(source_path+".lengths", "a") as lengths_out:
			print(target_num, file=lengths_out)

os.remove(source_path+".npy")
os.rename(source_path+"1.npy",source_path+".npy")

#toPCA = np.load(source_path+".npy")
#scaler = preprocessing.StandardScaler().fit(toPCA)
#toPCA = scaler.transform(toPCA)
#pca = PCA(n_components = 128)
#pca.fit(toPCA)
#np.save(source_path+".npy",pca.transform(toPCA))
