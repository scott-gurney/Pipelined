#!/bin/bash
#SBATCH --mail-type=ALL		#Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=uce19kqu@uea.ac.uk	#Where to send mail
#SBATCH -p compute-24-96		#Which partition to use
#SBATCH --cpus-per-task=4    #Set number of slots required
#SBATCH --mem=16G			#Maximum memory required for job
#SBATCH --time=0-24:00			#Maximum duration of job (DD-HH:MM)
#SBATCH --job-name=fMRIprepLogCU		#Arbitrary name for job
#SBATCH -o fMRIprepLogCU-%j.out		#Standard output log
#SBATCH -e fMRIprepLogCU-%j.err		#Standard error log

module add python/3.8		#Add the required modules for your job
python MainScript_Preprocess_CleanUp.py fMRI
python MainScript_Preanalysis_fMRI.py $1
