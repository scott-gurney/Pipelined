import os
import re
import shutil
import sys

def makeLogDirsCONN(directory):
	newDir = os.path.join(directory,"CONNlogs")
	try:
		os.mkdir(newDir)
	except FileExistsError:
		print("CONNlogs folder already exists.")

	print("Moving CONN logs to ",newDir)
	dirsFiles = os.listdir(directory)
	count = 0
	for files in dirsFiles:
		if re.fullmatch(r"CONN-\d*\.out|CONN-\d*\.err",files):
			toMove = os.path.join(directory,files)
			shutil.move(toMove,newDir)
			count = count + 1

	print(count, " log files moved")

def main():
	try:
		type = sys.argv[1]
	except:
		print("ERROR: No argument passed. Please pass either eASL or fMRI")
		exit()
	scriptDir = os.path.dirname(os.path.abspath(__file__))
	print("######################")
	print("Cleaning up slurm files")
	if type == "eASL":
		makeLogDirsEASL(scriptDir)
	elif type == "fMRI":
		makeLogDirsFMRI(scriptDir)
	else:
		print("Incorrect argument passed. Please pass either eASL or fMRI")
		exit()
	print("CleanUp of ", type, " Completed")
	print("Moving this file to logs too")
	print("#######################")
	
	if type == "eASL":
		newDir = os.path.join(scriptDir,"eASLLogFiles");
		dirsFiles = os.listdir(scriptDir)
		for files in dirsFiles:
			if re.fullmatch(r"ExploreASLLogCU-\d*\.out|ExploreASLLogCU-\d*\.err",files):
				toMove = os.path.join(scriptDir,files)
				print(toMove)
				shutil.move(toMove,newDir)
	else:
		newDir = os.path.join(scriptDir,"fMRIprepLogFiles");
		dirsFiles = os.listdir(scriptDir)
		for files in dirsFiles:
			if re.fullmatch(r"fMRIprepLogCU-\d*\.out|fMRIprepLogCU-\d*\.err",files):
				toMove = os.path.join(scriptDir,files)
				print(toMove)
				shutil.move(toMove,newDir)


if __name__ == "__main__":
	main()