import os
import sys
import subprocess

def makeCONNfile(source,scriptdir):
	fileToSave = []
	fileToSave.append("#!/bin/bash")
	fileToSave.append("#SBATCH --mail-type=ALL		#Mail events (NONE, BEGIN, END, FAIL, ALL)")
	#fileToSave.append("#SBATCH --mail-user=uce19kqu@uea.ac.uk	#Where to send mail")
	fileToSave.append("#SBATCH -p compute-24-96		#Which partition to use")
	fileToSave.append("#SBATCH --cpus-per-task=4    #Set number of slots required")
	fileToSave.append("#SBATCH --mem=16G			#Maximum memory required for job")
	fileToSave.append("#SBATCH --time=0-24:00			#Maximum duration of job (DD-HH:MM)")
	fileToSave.append("#SBATCH --job-name=CONN		#Arbitrary name for job")
	fileToSave.append("#SBATCH -o CONN-%j.out		#Standard output log")
	fileToSave.append("#SBATCH -e CONN-%j.err		#Standard error log")
	fileToSave.append("module add matlab/2020a		#Add the required modules for your job")
	changedir = 'cd ' + source
	fileToSave.append(changedir)
	fileToSave.append("matlab -nodisplay -nojvm -nodesktop -nosplash -r CONNscript")
	os.chdir(scriptdir)
	with open("runCONN.sh","w+") as f:
		for row in fileToSave:
			f.write(row + "\n")
	print("Saved Run CONN")

def makeCONNscript(source,pathtomat,scriptdir):
	fileToSave = []
	spm = os.path.join(scriptdir,'spm12')
	addspm = "addpath('"+spm+"')"
	fileToSave.append(addspm)
	conn = os.path.join(scriptdir,'conn')
	addconn = "addpath('"+conn+"')"
	fileToSave.append(addconn)
	loadfile = "load('"+pathtomat+"')"
	fileToSave.append(loadfile) # CHANGE
	fileToSave.append("conn_batch('Setup.done',true)")
	fileToSave.append("conn_batch('Denoising.done',true)")
	fileToSave.append("conn_batch('Analysis.name','RRC','Analysis.type','ROI-to-ROI','Analysis.weight', 'none','Analysis.done',true)")
	os.chdir(source)
	with open("CONNscript.m","w+") as f:
		for row in fileToSave:
			f.write(row + "\n")
	print("Saved CONNSCRIPT")

def main():
	#First Step
	scriptDir = os.path.dirname(os.path.abspath(__file__))
	print("##########")
	print("Pipeline (Analysis) Starting")
	print("Usage: Provide a filepath location to a folder where the setup has occured. There should be a conn file generated and saved in the base directory ending with .mat. This should be the only .mat file in the base directory.")
	for arg in sys.argv:
		try:
			source = sys.argv[1]
		except:
			print("ERROR: No filepath provided")
			print("	Please run python MainScript_Analysis.py <filepath>")
			exit()
	
	sourcedir = os.listdir(source)

	found = False
	for files in sourcedir:
		if files.endswith(".mat"):
			print("Found the following file: ", files)
			found = True
			dirToBe = files
	if not found:
		print("ERROR: The analysis pipeline requires a conn file generated using the conn gui. Please make sure to generate this first and it is the only directory in the source directory ending with .mat")
		print("Please restart once this file has been added to the directory.")
		exit()

	correctdir = os.path.join(source,dirToBe)
	
	#PERFORM CONN
	print("Generating Scripts")
	makeCONNscript(source,correctdir,scriptDir)
	makeCONNfile(source,scriptDir)
	print("All Scripts generated")

	print("Submitting CONN as a script")

	os.chdir(scriptDir)
	p = subprocess.Popen(["sbatch", "runCONN.sh"])
	p.wait()
	print("Job has been queued.")
	print("##############")
	print("Once the job has been completed, return to CONN and run the second level analysis.")

if __name__ == "__main__":
	main()
