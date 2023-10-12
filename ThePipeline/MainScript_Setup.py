import sys
import os
import re
import subprocess
from subprocess import PIPE
import json
import shutil
import MainScript_Preprocessing as MSP

def main():
	scriptDir = os.path.dirname(os.path.abspath(__file__))
	print("##########")
	print("Pipeline Starting")
	print("Usage: Provide a filepath location to a folder with a subfolder inside called ADNI, storing the ADNI data you wish to work with. Must be working with clinicaEnv conda enviroment")
	try:
		source = sys.argv[1]
	except:
		print("ERROR: No filepath provided")
		print("	Please run python MainScript_Setup.py <filepath>")
		exit()
	
	numargs = len(sys.argv)

	print("File path provided is ", source)
	print("##########")
	print("Creating subject lists for clinica")
	ADNIfolder = os.path.join(source,"ADNI") 

	print("Searching ", ADNIfolder ," for subjects")
	try:
		subject_folders = os.listdir(ADNIfolder)
	except:
		print("ERROR: No ADNI folder found")
		print("	Please provide your ADNI subject data in " , ADNIfolder)
		exit()

	if len(subject_folders) == 0:
		print("ERROR: ADNI folder is empty - Please provide subjects data!")
		exit()

	os.chdir(ADNIfolder)
	os.chdir("..")
	subjectCount = 0
	with open('subjects.txt','w') as file:
		for subject in subject_folders:
			if re.fullmatch(r"\d*_S_\d*",subject):
				file.write(subject+'\n')
				subjectCount = subjectCount + 1
			else:
				print("Omitted ", subject," - does not match recognised ADNI naming pattern for subjects.")
	if subjectCount > 0:
		print("subjects.txt created with ",subjectCount," subjects")
	else:
		print("ERROR: No Subjects Identified in ", subject_folders)
		print("	Make sure the subjects are in the correct subfolder and names match the following pattern: 123_S_1234")
		exit()

	#RUN CLINICA
	if "bids" in os.listdir(source):
		print("bids folder already exists - skipping setup")
		if "ADNIsubjects.txt" in os.listdir(source):
			print("ADNIsubjects present - skipping generation")
		else:
			bidsDir = os.path.join(source,"bids")
			bidssubject_folders = os.listdir(bidsDir)
			for subject in bidssubject_folders:
				test = os.path.join(bidsDir,subject)
				if os.path.isdir(test):
					if re.fullmatch(r"sub-ADNI\d*S\d*",subject):
						with open('ADNIsubjects.txt','a') as file:
							file.write(subject+'\n')
							subjectCount = subjectCount + 1
					else:
						print("Omitted ", subject," - does not match recognised ADNI naming pattern for subjects.")

			if subjectCount > 0:
				print("ADNIsubjects.txt created with ",subjectCount," subjects")
			else:
				print("ERROR: No Subjects Identified in ", bidsDir)
				print("	Make sure the subjects are in the correct subfolder and names match the following pattern: sub-ADNI123S1234")
				exit()
			print("##########")

	else:
		clinicalData = os.path.join(scriptDir,"clinical")
		os.mkdir("bids")
		bidsDir = os.path.join(source,"bids")
		subjectfile=os.path.join(source,"subjects.txt")
		os.chdir(scriptDir)
		print("Running clinica adni-bids convertor")
		logfile = os.path.join(scriptDir,"clinicalog.txt")
		p = subprocess.Popen(["bash", "clinica.sh",ADNIfolder,clinicalData,bidsDir,subjectfile])
		p.wait()
		print("Clinica Completed - Log files available in ", logfile)
		print("Creating updated subject list for fMRIprep")

		os.chdir(source)
		bidssubject_folders = os.listdir(bidsDir)
		subjectCount = 0
		try:
			os.remove(os.path.join(source,"ADNIsubjects.txt"))
		except FileNotFoundError:
			print("No previous list")
		for subject in bidssubject_folders:
			test = os.path.join(bidsDir,subject)
			if os.path.isdir(test):
				if re.fullmatch(r"sub-ADNI\d*S\d*",subject):
					with open('ADNIsubjects.txt','a') as file:
						file.write(subject+'\n')
						subjectCount = subjectCount + 1
				else:
					print("Omitted ", subject," - does not match recognised ADNI naming pattern for subjects.")

		if subjectCount > 0:
			print("ADNIsubjects.txt created with ",subjectCount," subjects")
		else:
			print("ERROR: No Subjects Identified in ", bidsDir)
			print("	Make sure the subjects are in the correct subfolder and names match the following pattern: sub-ADNI123S1234")
			exit()
		print("##########")

	print("Now performing ExploreASL setup")

	os.chdir(scriptDir)

	if "eASL" in os.listdir(source):
		print("eASL directory already exists - skipping eASL setup.")
	else:
		with open('imagingtypes.json') as f:
			datatypes = json.load(f)
		print("Loaded in imaging types")
		eASLfolder = os.path.join(source,"eASL")
		eASLfolderSource = os.path.join(eASLfolder ,"sourcedata")
		os.mkdir(eASLfolder)
		os.mkdir(eASLfolderSource)

		subject_files = os.listdir(ADNIfolder)
		generalsessions = []
		
		modalsnotknown = []
		for Subs in subject_files:
			x= os.path.join(ADNIfolder,Subs)
			datatypes_files = os.listdir(x)
			sessions = []
			for modals in datatypes_files:
				if modals not in datatypes.keys():
					print("The imaging modal ", modals, " is not recognised")
					print("Please add it to the imagingtypes.json file")
					modalsnotknown.append(modals)

		if len(modalsnotknown) > 0:
			print("Please add the above imaging types to the imagingtypes.json file")
			print("Follow bids format, but leave T1w images as T1w instead of anat.")
			exit()

		print("Making directory structure")

		##For subject
		for Subs in subject_files:
			try:
				x= os.path.join(eASLfolderSource,Subs)
				os.mkdir(x)
			except FileExistsError:
				print("WARNING: Subject Folder Already Exists")
			except:
				print("ERROR: Generating Subject Folder")
		#Make File Structure
		for Subs in subject_files:
			x= os.path.join(ADNIfolder,Subs)
			datatypes_files = os.listdir(x)
			sessions = []
			for modals in datatypes_files:
				if modals in datatypes.keys():
					#Make directory with correct name
					y = os.path.join(x,modals)
					sestypes_files = os.listdir(y)
					sestypes_files.sort()
					i=1
					for ses in sestypes_files:
						z = os.path.join(y,ses)
						new = "ses-" + str(i)
						newz = os.path.join(y,new)
						os.rename(z,newz)
						if new not in sessions:
							sessions.append(new)
						i = i + 1
			if len(generalsessions) < len(sessions):
				generalsessions = sessions
			for i in range(0,len(sessions)):
				try:
					dest = "ses-" + str(i+1)
					x= os.path.join(eASLfolderSource,Subs)
					z= os.path.join(x,dest)
					os.mkdir(z)
				except FileExistsError:
					print("WARNING: Session Folder Already Exists")
				except:
					print("ERROR: Generating Subject Folder")
				try:
					for modals in datatypes_files:
						if modals in datatypes.keys():
							if modals == "Accelerated_Sag_IR-FSPGR":
								m = "Accelerated_Sagittal_IR-FSPGR"
							else:
								m = modals
							zx = os.path.join(z,m)
							os.mkdir(zx)
				except FileExistsError:
					print("WARNING: Modal Folder Already Exists")
				except:
					print("ERROR: Generating Subject Folder")
		
		print("Directory structure made")

		print("Making supporting files")
		os.chdir(eASLfolder)
		studyPar = {"LabelingType":"PCASL"}
		with open('studyPar.json','w') as f:
			json.dump(studyPar,f)
		print("studyPar generated")
		
		sourcestructure = {}

		#Creating folder Hierachy

		fh = []
		fh.append("^(\\d{3}_S_\\d{4})")
		sesstring = "^ses-(["
		for x in range(0,len(generalsessions)):
			if x+1 == 9:
				sesstring = sesstring + str(0)
			elif x+1 < 9:
				sesstring = sesstring + str(x+1)
				
		sesstring = sesstring + "])$"
		fh.append(sesstring)

		typestring = "^("
		numkeys = len(datatypes.keys())
		count = 1
		for key in datatypes.keys():
			typestring = typestring+key
			if count < numkeys:
				typestring = typestring+"|"
			count = count + 1
		typestring = typestring + ")$"
		fh.append(typestring)
		#Creating tokenVisitAliases
		tva = []
		sesnum =1
		for session in generalsessions:
			tva.append(session)
			tva.append(str(sesnum))
			sesnum = sesnum +1

		#Creating tokenSessionAliases
		tsa = []
		for x in range(0,len(generalsessions)):
			tempstr = "^" + str(x+1) + "$"
			tsa.append(tempstr)
			tempstr = "ASL_" + str(x+1)
			tsa.append(tempstr)

		#Creating tokenScanAliases
		tsca = []

		for key in datatypes.keys():
			tsca.append("^"+key+"$")
			tsca.append(datatypes.get(key))

		#Creating source structure and saving file
		sourcestructure["folderHierarchy"] = fh
		sourcestructure["tokenOrdering"] = [1, 0, 2, 3]
		sourcestructure["tokenVisitAliases"] = tva
		sourcestructure["tokenSessionAliases"] = tsa
		sourcestructure["tokenScanAliases"] = tsca
		sourcestructure["bMatchDirectories"] = 1

		with open('sourcestructure.json','w') as f:
			json.dump(sourcestructure,f)
		print("sourcestructure generated")

		print("All supporting files generated")
		print("Now copying files across")

		#Move Image files from ADNI to BIDS
		for Subs in subject_files:
			x= os.path.join(ADNIfolder,Subs)
			datatypes_files = os.listdir(x)
			for modals in datatypes_files:
				if modals in datatypes.keys():
					m = modals				
					y = os.path.join(x,modals)
					sestypes_files = os.listdir(y)
					for i in range(len(sestypes_files)):
						z = os.path.join(y,sestypes_files[i])
						imagefiles = os.listdir(z)
						for img in imagefiles:
							xy = os.path.join(z,img)
							allfiles = os.listdir(xy)
							for images in allfiles:
								xyz = os.path.join(xy,images)
								newdir = os.path.join(eASLfolderSource,Subs)
								session = "ses-" + str(i+1)
								newdir = os.path.join(newdir,session)
								newdir = os.path.join(newdir,m)
								print(newdir)
								print("Copying ", xy, " to " , newdir)
								shutil.copy2(xyz, newdir)

	print("####################")
	print("Setup is now completed.")
	if numargs > 2 :
		print("Preprocessing stage will begin shortly")
		print("Starting MainScript_Preprocessing.py...")
		print("####################")
		MSP.main()
		
	else:
		print("	Run the pre-processing pipeline next with the following command")
		print(" python MainScript_Preprocessing.py " + source)
		print("####################")
if __name__ == "__main__":
	main()
