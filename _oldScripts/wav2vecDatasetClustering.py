import os
import shutil
import random
import torch
import fairseq
import soundfile
import numpy

randomFilesConstant=100

#first clear the directory
#this will fail if there are subdirectories
print("Clearing random file directories...")
dir = 'testingData/randomlySelectedFiles'
for file in os.scandir(dir):
	os.remove(file.path)
outputDir = 'testOutputs/clusteringOutputs'
for file in os.scandir(outputDir):
	os.remove(file.path)

#now fill with selected amount of random files from librispeech
dataDir = 'datasets/LibriSpeech/train-clean-100'
fileList = []
for root, dirs, files in os.walk(dataDir):
	for file in files:
		if file.endswith(".flac"):
			fileList.append(os.path.join(root,file))
selectedFileList = random.sample(fileList,randomFilesConstant)
for file in selectedFileList:
	shutil.copy(file,dir)
print("Randomly selected "+str(randomFilesConstant)+" LibriSpeech files for clustering")
print("Preparing Wav2Vec Model...")

#use wav2vec to get general representation of each file
cp_path = 'wav2vec_large.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()

print("---WAV2VEC MODEL READY---")

for file in os.listdir(dir):
	#comments on this step can be seen in initialFlacTest.py
	print("Processing: "+file)
	audioInput = soundfile.read(dir+"/"+file)[0]
	tensorAudio = torch.from_numpy(audioInput).unsqueeze(0).float()
	z = model.feature_extractor(tensorAudio)
	c = model.feature_aggregator(z)
	print("Aggregator Size")
	print(c.size())
	#[:-5] slices off the .flac from the filenames for writing
	fname = "testOutputs/clusteringOutputs/"+file[:-5]+".txt"
	with open(fname, "w") as f:
		numpy.savetxt(fname, c.squeeze().detach().numpy())

print("---General Representations Generated---")