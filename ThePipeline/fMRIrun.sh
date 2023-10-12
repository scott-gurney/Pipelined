#!/bin/bash

mapfile -t participants < $1

unset jid
x=0
for i in "${participants[@]}"
do
	jid[$x]=$(sbatch --parsable $2 $i)
	sleep 1m
	echo $i Started
	x=$((x+1))
done

alljobs=$(printf "%s:" "${jid[@]}")
jobstring=${alljobs%?}
jidfinal=$(sbatch  --dependency=afterok:$jobstring fMRICleanUp.sh $2)
