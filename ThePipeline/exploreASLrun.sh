#!/bin/bash

jid=$(sbatch --parsable exploreASLImp.sh)
echo exloresASLImp Started
echo $jid
jidfinal=$(sbatch  --parsable --dependency=afterok:$jid exploreASLPro.sh)
echo exploresASLPro Queued
echo $jidfinal
sbatch --dependency=afterok:$jidfinal exploreASLCleanUp.sh
echo cleanup Queued