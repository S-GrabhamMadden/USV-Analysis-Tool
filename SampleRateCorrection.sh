callDir=$1
outDir=$2

for i in $callDir/*.wav; do
	echo "Processing file $i"
	sox -r 16000 $i $outDir/${i##*/}
done