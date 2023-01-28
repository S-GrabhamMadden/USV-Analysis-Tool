from scipy.io import wavfile
import numpy as np
import noisereduce as nr
import argparse
import os
import glob


def get_parser():
	parser = argparse.ArgumentParser(
		description="Bandpass Filter + Spectral Gating to isolate USV sound"
	)
	# fmt: off
	parser.add_argument('datadir', help='location of wav files')
	# fmt: on

	return parser

parser = get_parser()
args = parser.parse_args()

for filename in glob.glob(os.path.join(args.datadir, '*.wav')):

	print ("Cleaning: " + filename)

	# load data
	rate, data = wavfile.read(filename)

	#high-pass filter
	def fir_band_pass(samples, fs, fL, fH, NL, NH, outputType):

		fH = fH / fs
		fL = fL / fs

		# Compute a low-pass filter with cutoff frequency fH.
		hlpf = np.sinc(2 * fH * (np.arange(NH) - (NH - 1) / 2.))
		hlpf *= np.blackman(NH)
		hlpf /= np.sum(hlpf)
		# Compute a high-pass filter with cutoff frequency fL.
		hhpf = np.sinc(2 * fL * (np.arange(NL) - (NL - 1) / 2.))
		hhpf *= np.blackman(NL)
		hhpf /= np.sum(hhpf)
		hhpf = -hhpf
		hhpf[int((NL - 1) / 2)] += 1
		# Convolve both filters.
		h = np.convolve(hlpf, hhpf)
		# Applying the filter to a signal s can be as simple as writing
		s = np.convolve(samples, h).astype(outputType)

		return s

	data = fir_band_pass(data, rate, 34000, 68000, 461, 461, np.int16)
	#data = data*2

	# perform noise reduction
	reduced_noise = nr.reduce_noise(y=data, sr=rate)
	#reduced_noise = data
	
	if not os.path.exists(os.path.join(args.datadir,"SqueakyClean")):
		os.mkdir(os.path.join(args.datadir,"SqueakyClean"))
	wavfile.write(os.path.join(args.datadir,"SqueakyClean","SqueakyClean"+os.path.basename(filename)), rate, reduced_noise)