#!/bin/bash
#SBATCH --mail-type=ALL		#Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=uce19kqu@uea.ac.uk	#Where to send mail
#SBATCH -p compute-24-96		#Which partition to use
#SBATCH --cpus-per-task=4    #Set number of slots required
#SBATCH --mem=48G			#Maximum memory required for job
#SBATCH --time=1-24:00			#Maximum duration of job (DD-HH:MM)
#SBATCH --job-name=ExploreASL		#Arbitrary name for job
#SBATCH -o ExploreASL-%j.out		#Standard output log
#SBATCH -e ExploreASL-%j.err		#Standard error log

module add matlab/2020a		#Add the required modules for your job
cd ExploreASL-main
matlab -nodisplay -nojvm -nodesktop -nosplash -r ASLProcess

