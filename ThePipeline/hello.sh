#!/bin/bash -e
#SBATCH --mail-type=ALL #ALL, 	BEGIN, END, FAIL, TIME_LIMIT_90 etc
#SBATCH --mail-user=uqe18wbu@uea.ac.uk #Where to send mail
#SBATCH -p compute-24-96 	#Which partition to use
#SBATCH --mem=12G 		#Maximum memory required for job
#SBATCH --time=0-09:00 		#Maximum duration of job (DD-HH:MM)
#SBATCH --job-name=test_job 	#Arbitrary name for job
#SBATCH -o test_job-%j.out 	#Standard output log
#SBATCH -e test_job-%j.err 	#Standard error log

pwd
