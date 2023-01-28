callDir=$1

python fairseq/examples/wav2vec/wav2vec_manifest.py $callDir --ext wav --dest workingDir --valid-percent 0

python fairseq/examples/wav2vec/unsupervised/scripts/vads.py -r rVADfast < workingDir/train.tsv > workingDir/train.vads

python fairseq/examples/wav2vec/unsupervised/scripts/remove_silence.py --tsv workingDir/train.tsv --vads workingDir/train.vads --out workingDir

python fairseq/examples/wav2vec/wav2vec_manifest.py workingDir/$callDir --ext wav --dest workingDir/$callDir --valid-percent 0