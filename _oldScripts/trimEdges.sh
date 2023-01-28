cd largeTest/preprocessedAudio/SqueakyClean
COUNTER=1

for f in *.wav
do
	echo "Trimming $f file..." 
	sox "$f" SCClipped$COUNTER.wav trim 0.02 -0.02
	COUNTER=$[$COUNTER +1]
	rm "$f"
done

cd ../../..