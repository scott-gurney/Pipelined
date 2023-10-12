import os
import sys
import subprocess


def generateSbatchScript(source,scriptDir):
	fileToSave = []
	fileToSave.append("#!/bin/bash")
	fileToSave.append("#SBATCH --mail-type=ALL		#Mail events (NONE, BEGIN, END, FAIL, ALL)")
	#fileToSave.append("#SBATCH --mail-user=uce19kqu@uea.ac.uk	#Where to send mail")i
	#fileToSave.append("#SBATCH -p gpu-rtx6000-2       #Which queue to use")
	#fileToSave.append("#SBATCH --qos=gpu-rtx")
	#fileToSave.append("#SBATCH --gres=gpu:1")
	fileToSave.append("#SBATCH -p compute-64-512		#Which partition to use")
	fileToSave.append("#SBATCH --cpus-per-task=24    #Set number of slots required")
	fileToSave.append("#SBATCH --mem=80G			#Maximum memory required for job")
	fileToSave.append("#SBATCH --time=0-24:00			#Maximum duration of job (DD-HH:MM)")
	fileToSave.append("#SBATCH --job-name=fMRIPREP		#Arbitrary name for job")
	fileToSave.append("#SBATCH -o fMRIPREP-%j.out		#Standard output log")
	fileToSave.append("#SBATCH -e fMRIPREP-%j.err		#Standard error log")
	fileToSave.append("user=\"$1\"")
	fileToSave.append("module add fmriprep/21.0.1		#Add the required modules for your job")
	bidsdir = os.path.join(source,"bids")
	outdir = os.path.join(source,"fMRIoutput")
	#os.mkdir(outdir)
	workfolder = os.path.join(source,"fMRIWorkFolder")
	fmricall = "fmriprep " +bidsdir+ " " +  outdir +"/$user participant --participant-label $user --skip_bids_validation -w " + workfolder  +"/$user --fs-license-file $HOME/license.txt --n_cpus 24 --omp-nthreads 1 --mem_mb 16000 --cifti-output --longitudinal --use-aroma" 
	fileToSave.append(fmricall)
	os.chdir(source)

	with open("fMRIJob.sh","w+") as f:
		for row in fileToSave:
			f.write(row + "\n")


def generateExploreASLScripts(ExploreASLDir,ExploreASLData):
	print("Generating Import Script")
	fileToSave = ["[x] = ExploreASL('"+ ExploreASLData +"', 1,0);"]
	os.chdir(ExploreASLDir)
	with open("ASLimport.m","w+") as f:
		for row in fileToSave:
			f.write(row + "\n")

	print("Generating Process Script")
	fileToSave = ["[x] = ExploreASL('"+ ExploreASLData +"', 0,[1 0 0]);"]
	fileToSave.append("[x] = ExploreASL('"+ ExploreASLData +"', 0,[0 1 0]);")
	fileToSave.append("[x] = ExploreASL('"+ ExploreASLData +"', 0,[0 0 1]);")
	with open("ASLProcess.m","w+") as f:
		for row in fileToSave:
			f.write(row + "\n")

def main():
	#First Step
	scriptDir = os.path.dirname(os.path.abspath(__file__))
	print("##########")
	print("Pipeline (Preprocessing) Starting")
	print("Usage: Provide a filepath location to a folder where the setup has occured.")
	for arg in sys.argv:
		try:
			source = sys.argv[1]
		except:
			print("ERROR: No filepath provided")
			print("	Please run python MainScript_Setup.py <filepath>")
			exit()

	print("File path provided is ", source)
	print("Checking expected folders appear in the directory")
	sourcedirs = os.listdir(source)
	if "ADNI" not in sourcedirs:
		print("WARNING: ADNI folder not found!")
	if "bids" not in sourcedirs:
		print("ERROR: bids folder not found in source directory. Please check setup exited successfully!")
		exit()
	if "eASL" not in sourcedirs:
		print("ERROR: eASL folder not found in source directory. Please check setup exited successfully!")
		exit()
	print("Setup was completed successfully. Continuing with preprocessing steps")
	print("##########")
	print("fMRIprep beginning")

	ADNIsubjects = os.path.join(source,"ADNIsubjects.txt")
	generateSbatchScript(source,scriptDir)
	os.chdir(scriptDir)
	fileloc = os.path.join(source,"fMRIJob.sh")
	p = subprocess.Popen(["bash", "fMRIrun.sh",ADNIsubjects,fileloc])
	p.wait()
	print("All fmri jobs queued")
	print("Beginning ExploreASL")
	ExploreASLDir = os.path.join(scriptDir,"ExploreASL-main")
	ExploreASLData = os.path.join(source,'eASL')
	generateExploreASLScripts(ExploreASLDir,ExploreASLData)
	os.chdir(scriptDir)
	p = subprocess.Popen(["bash", "exploreASLrun.sh"])
	p.wait()
	print("ExploreASL jobs started.")



if __name__ == "__main__":
	main()
