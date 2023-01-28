import torch
import fairseq
import soundfile
import os
import numpy

cp_path = 'wav2vec_large.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()

speechFile = "testingData/FlacTest19-198-0001.flac"

if not os.path.exists(speechFile):
	print("Cannot run, data is missing")
else:
	#[0] to get the numpy array from the resulting tuple
	audioInput = soundfile.read(speechFile)[0]
	# then turn the numpy data of the audio into a tensor for the model to use, unsqueeze just solves a shaping issue
	tensorAudio = torch.from_numpy(audioInput).unsqueeze(0).float()

	z = model.feature_extractor(tensorAudio)
	c = model.feature_aggregator(z)

	print("Extractor Size")
	print(z.size())

	print("Aggregator Size")
	print(c.size())


	print("-----Model Feature Extractor Output:-----")
	print(z)

	print("-----Model Feature Aggregator Output:-----")
	print(c)

	with open("testOutputs/flacOutput.txt", "w") as f:
		#f.write("-----Model Feature Extractor Output:----- \n")
		#numpy.savetxt("testOutputs/flacOutput.txt", z.squeeze().detach().numpy())
		f.write("-----Model Feature Aggregator Output:----- \n")
		numpy.savetxt("testOutputs/flacOutput.txt", c.squeeze().detach().numpy())
