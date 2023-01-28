cd datasets/USVClips
COUNTER=1

for f in *.wav
do
	echo "Trimming $f file..." 
	sox "$f" usvClip$COUNTER.wav trim 0 750
	COUNTER=$[$COUNTER +1]
done

cd ../..