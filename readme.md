# Rat Automatic Transcriber System
### _For USV Call Categorisation_

RATS is a project for transcribing recorded rat USV calls (in WAV format) into a string of call category codes. It makes use of the Wav2Vec2.0 XLSR-53 model by fairseq for initial feature extraction, and a significantly modified version of the Wav2Vec-U audio preprocessing step for processing into segment representations of individual calls.

HDBSCAN Clustering has been performed on a large initial dataset, creating the clustering used at runtime to predict the probabilities that new calls fall into each cluster, for pseudo-categorisation. The set of highest-probability cluster codes are output to a text file, for any further analysis.

##### Note:

- This project is designed for sorting complex 50kHz-range "affective" type USV calls into subcategories. Maternal separation calls and aversive calls at ~22kHz are much simpler in form and often significantly longer.
- The system also assumes appropriate audio preprocessing (such as sample rate adjustment to 16kHz and noise reduction) has already taken place on input audio files. If required, the project includes scripts to assist, see the Preproccessing section below.

## Use

It is assumed that fully preprocessed audio files are used. A guide to preprocessing raw USV audio files (along with hopefully-helpful scripts to replicate the preprocessing used in this project) is provided in the section below.

**Manifest**
With preprocessed USV call recordings ready in a directory, the wav2vec models require a manifest to read from. If the preprocessing steps below were followed, this already exists and this step is unnecessary. The manifest will be in a directory under workingDir named identically to the directory that contained the processed WAV files. If not, the following command will make one:
```sh
python fairseq/examples/wav2vec/wav2vec_manifest.py workingDir/USVFiles --ext wav --dest workingDir/USVFiles --valid-percent 0
```

**Segment Representation Extraction**
Running the following script will start a process similar to that of Wav2Vec-U's audio preprocessing steps, extracting timestep features and processing down to create segment representations. In this case, these are 32-feature representations of individual USV calls.

The dimensionality can optionally be increased from 32 by providing a fourth argument (more correctly, the dimensionality will be less reduced by PCA), but this may result in unexpected behaviour. If preprocessing steps below were used, the directory under workingDir with the manifest files (workingDir/USVFiles) might be differently named. A slightly modified checkpoint of the pretrained xlsr_53_56k.pt model by fairseq is used in the project by default. See the troubleshooting/tips section for how to get the same if needed.
```sh
source UnsupervisedUSVSegmenting.sh workingDir/USVFiles workingDir/representationData xlsr_53_56k_new.pt
```

**Segment Representation Clustering**
After running the above, segment representations (and the feature data at previous steps in the process) are available for analysis in workingDir/representationData. Run the following script to output segment clustering results to a text file:
```sh
python clusterUSVSegments.py result.txt
```

Output is formatted as a line of segment cluster codes, followed by a line of corresponding segment lengths in miliseconds, for each clip. Values are comma-separated.

## Preprocessing

To get from raw USV files to WAVs acceptable to the project, follow these steps:

**File Type Conversion**
Many USV files, particularly those provided by VUW, are in .aif or .aiff format. Converting these to wav format is trivial using sox. Run the following in the directory containing .aif or .aiff files:
```sh
find . -maxdepth 1 \( -iname "*.aif" -o -iname "*.aiff" \) -exec bash -c '$0 "$1" "${1%.*}.wav"' sox {} \;
```

**Sample Rate Conversion**
VUW's USV data is sampled at 192kHz, whereas Wav2Vec systems are used to 16kHz. Downsampling to a factor of twelve will cause significant data loss, but it is possible to simply change the sample rate in the WAV headers without losing any data. It effectively becomes lower pitched and twelve times longer, but this is accounted for within the system. (In fact, it is an expectation. A different ratio of actual sampling rate to displayed 16kHz may produce unexpected behaviour.) A shell script is provided to automatically make this change:
```sh
source SampleRateCorrection.sh dir/with/data dir/to/output
```

**Noise Reduction**
Eliminating sound other than USV calls in the audio clips (including breathing, bumps, and microphone hum) is essential for silence removal to work well, with significant effects on results downstream. Exploring new and improved USV noise isolation techniques is encouraged. For now, a script is provided that runs a combination of a band-pass filter and spectral gating which is effective but imperfect:
```sh
python scripts/SqueakyCleanUSVs.py dir/to/use
```

**Silence Removal**
After USV audio is isolated, there should be large gaps of silence separated by individual USV calls (after sample rate conversion, these should be audible to most humans, and can be visually inspected for performance using spectrograms, such as this online tool: https://academo.org/demos/spectrum-analyzer/). The following script removes these large silence gaps, leaving only a set of separated USV calls for the models to extract features from, with a manifest for the main scripts to use in the working directory.
```sh
source callNoSilenceManifest.sh dir/with/clean/data
```
It is worthwhile to inspect the result of this step, possibly in a spectrum analyzer or tool like audacity. Some noise might have made it through, and it may be worth manually deleting to avoid it messing with clustering.

**Trimming Edges (Optional)**
It might be desirable to trim the leading and trailing silence from each clip. That can be accomplished with sox trimming. The following parameters worked best when I tested using trimmed clips:
```sh
sox path/to/SqueakyCleanaudio.wav trimmed.wav trim 0.12
sox trimmed.wav path/to/SqueakyCleanaudio.wav trim 0 -0.08
```

## Improving Results

Current results in terms of cluster assignment are functional, but not necessarily perfect. If higher performance is desired or necessary, there are ways to make improvements in every area of the project:
1. **Better Data:** A larger dataset would be useful, but more importantly a cleaner one would make a huge difference. Less noise or silence mixed in with the useful data will naturally improve performance. If/when this proves difficult to obtain, making improvements to the performance of the noise isolation script (scripts/SqueakyCleanUSVs.py) can substitute.
2. **Purpose-Built Feature Extraction:** Using the wav2vec2.0 XLSR model for timestep feature extraction and a modified wav2vec-u audio processing pipeline to produce segment representations works, perhaps better than expected. A USV Call segment feature extraction system tailor-made for this purpose would probably produce cleaner results overall.
3. **Refined Clustering:** Presently, hyperparameters for HDBSCAN clustering were tweaked to optimize the DBCV validity score of the clustering result, within the bounds set by an initial K-Means Intertial Elbow Method investigation. Clustering results might be better if we had a target number of clusters to aim for - but this is impossible as finding this number without human interference is part of the project. Training a new machine learning classifier is an alternative option. For now, upstream improvements in data/segment extraction should make better clustering.

Run ```UnsupervisedAudioProcess.sh``` to extract new features on larger dataset after making improvements (this script runs the feature processing python scripts, modify those to change how the feature extraction actually works). If continuing with HDBSCAN clustering, run ```HDBSCANCluster.py``` on new segment feature results to regenerate object files (might need to change hyperparameters depending on cluster count being reasonable). With changed clustering object files, can just rerun ```clusterUSVSegments.py``` on new data as previously.

## Troubleshooting/Tips

- The project has its own python virtual environment with the right versions of needed tools installed. If complaints are raised about missing python modules or similar, the first port of call is to be sure this virtualenv is on. To do so, just run ```source env/bin/activate``` in the main project directory. (Just run ```deactivate``` to leave the virtualenv afterwards).

- OmegaConf key error: "Key 'eval_wer' not in 'AudioPretrainingConfig'": If using the pretrained XLSR model by fairseq for initial feature extraction (available here: https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md), you might see this error. In the _oldScripts directory, find ```omegaConfErrorResolve.py```. Running it should create a new checkpoint of the model that resolves this error.

- Several steps in the process make use of GPU resources through CUDA. Using ssh to connect into a cuda server (if remote connecting to a VUW cuda server, must ssh through an open server such as barretts first), ```nvidia-smi``` will display which GPUs have active processes. Several times I experienced an instance where CUDA would throw an error that no devices are available part-way through processing, when there is one or several GPUs free. The project only needs one GPU to run, but if any other GPUs are in use it can throw this error. Pick a GPU with no active processes and identify its ID. Use ```export CUDA_VISIBLE_DEVICES=x``` setting X to that GPU ID, so CUDA will only see the free GPU being used. This should resolve the error. This environmental variable only lasts within this shell session.