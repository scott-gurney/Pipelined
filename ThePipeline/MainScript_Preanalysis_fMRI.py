import os
import sys
import re
import shutil

def renameSessionFolders(fMRIoutputdir,subject):
	subjectFolder = os.path.join(fMRIoutputdir,subject)
	subjectNestedFolder = os.path.join(subjectFolder,subject)
	try:
		sessionFolders = os.listdir(subjectNestedFolder)
	except:
		print(subject, " file not found")
		return None
	trimmedSession = []
	for sf in sessionFolders:
		if re.fullmatch(r"ses-M\d*",sf):
			trimmedSession.append(sf[5:])
		elif re.fullmatch(r"ses-\d*",sf):
			print("		Already converted to standardised sessions")
			return
	trimmedSession.sort()
	if len(trimmedSession) == 0:
		print("		WARNING: No sessions available for subject ", subject)
		return subject
	else:
		sessioncount = 1
		for val in trimmedSession:
			sf = "ses-M" + val
			oldpath = os.path.join(subjectNestedFolder,sf)
			newses = "ses-" + str(sessioncount)
			newpath = os.path.join(subjectNestedFolder,newses)
			os.rename(oldpath,newpath)
			sessioncount = sessioncount + 1
	print("		Session Folders succesfully renamed")

def moveOLDsessions(fMRIoutputdir,subject,source):
	subjectFolder = os.path.join(fMRIoutputdir,subject)
	subjectNestedFolder = os.path.join(subjectFolder,subject)
	try:
		sessionFolders = os.listdir(subjectNestedFolder)
	except:
		print(subject, " file not found")
		return
	for sf in sessionFolders:
		if sf.endswith("OLD"):
			currentlocation = os.path.join(subjectNestedFolder,sf)
			newlocation = os.path.join(source,"rejectedFMRIsessionsCONN",subject)
			shutil.move(currentlocation,newlocation)
		if re.fullmatch(r"ses-3",sf):
			currentlocation = os.path.join(subjectNestedFolder,sf)
			newlocation = os.path.join(source,"rejectedFMRIsessionsCONN",subject)
			shutil.move(currentlocation,newlocation)	



def extraSessions(fMRIoutputdir,subject):
	subjectFolder = os.path.join(fMRIoutputdir,subject)
	subjectNestedFolder = os.path.join(subjectFolder,subject)
	try:
		sessionFolders = os.listdir(subjectNestedFolder)
	except:
		print(subject, " file not found")
		return
	sessions = []
	for sf in sessionFolders:
		if re.fullmatch(r"ses-\d*",sf):
			sessions.append(sf)

	print(sessions)
	seswithoutfunc = []
	for session in sessions:
		folder = os.path.join(subjectNestedFolder,session,"func")
		try:
			files = os.listdir(folder)
		except FileNotFoundError:
			print("WARNING: Functional files not found for ", subject, " at Session:", session)
			seswithoutfunc.append(session)
	
	newsessions = sessions
	for ses in seswithoutfunc:
		newsessions.remove(ses)
	print("FINAL SESSIONS FOR SUBJECT:",subject)
	print(newsessions)

	for i in range(len(newsessions)):
		changed = False
		print(newsessions[i], "Becomes ses-",i+1)
		sesno = i+1
		session = "ses-"+str(sesno)
		if session == newsessions[i]:
			print("## Session is correct ##")
		else:
			if session in os.listdir(subjectNestedFolder):
				print("## Renaming session ##")
				originalfile = os.path.join(subjectNestedFolder,session)
				newfile = os.path.join(subjectNestedFolder,str(session+"OLD"))
				os.rename(originalfile,newfile)
			
			originalfile = os.path.join(subjectNestedFolder,newsessions[i])
			newfile = os.path.join(subjectNestedFolder,session)
			os.rename(originalfile,newfile)
			print("Session renamed")
			changed = True
		
		if changed:
			renameSessionFolders(fMRIoutputdir,subject)



def renameSessionFiles(fMRIoutputdir,subject):
	subjectFolder = os.path.join(fMRIoutputdir,subject)
	subjectNestedFolder = os.path.join(subjectFolder,subject)
	try:
		sessionFolders = os.listdir(subjectNestedFolder)
	except:
		print(subject, " file not found")
		return

	sessions = []
	for sf in sessionFolders:
		if re.fullmatch(r"ses-\d*",sf):
			sessions.append(sf)

	print("		Renaming func images")
	count = 0
	for session in sessions:
		folder = os.path.join(subjectNestedFolder,session,"func")
		# isError = False
		try:
			files = os.listdir(folder)
		except FileNotFoundError:
			print("WARNING: Functional files not found for ", subject)
			continue

		# if not isError:
		for file in files:
			path=os.path.join(folder,file)
			listnewname = file.split("_")
			und = "_"
			if listnewname[0] == "scrubbing":
				listnewname[2] = session
				newname = und.join(listnewname)
			else:
				listnewname[1] =session
				newname = und.join(listnewname)
			newpath = os.path.join(folder,newname)
			os.rename(path,newpath)
			count = count + 1

	print("		Renaming func images(", count ,") successfully completed ")

	print("		Renaming anat images")
	count = 0
	for session in sessions:
		folder = os.path.join(subjectNestedFolder,session,"anat")
		try:
			files = os.listdir(folder)
		except FileNotFoundError:
			print("WARNING: Anat files not found for ", subject)
			break

		for file in files:
			path=os.path.join(folder,file)
			listnewname = file.split("_")
			und = "_"
			listnewname[1] =session
			newname = und.join(listnewname)
			newpath = os.path.join(folder,newname)
			os.rename(path,newpath)
			count = count + 1
	print("		Renaming anat images(", count ,") succesfully completed")

def main():
	#First Step
	print("##########")
	print("Pipeline Pre-analysis fMRI Starting")
	for arg in sys.argv:
		try:
			source = sys.argv[1]
		except:
			print("ERROR: No filepath provided")
			print("Please run python MainScript_Preanalysis_fMRI.py <filepath>")
			print("Filepath being to the root directory of the individual")
			exit()

	if source.endswith("/fMRIJob.sh"):
		source = source[:-11]

	fMRIoutputdir = os.path.join(source,"fMRIoutput")
	subjects = os.listdir(fMRIoutputdir)
	count = 0
	print("Verifying subject folders")
	for subject in subjects:
		if re.fullmatch(r"sub-ADNI\d*S\d*",subject):
			count = count+1
		else:
			print("WARNING: Potential subject removed: ", subject)
			print("	If this is meant to be a subject please make sure the ouput is sub-ADNI000S0000")
			subjects.remove(subject)

	print(count, " subject folders found")
	print("Checking if these subjects fMRIprep exited succesfully.")
	for subject in subjects:
		subjectFolder = os.path.join(fMRIoutputdir,subject)
		subjectNestedFolder = os.path.join(subjectFolder,subject)
		if not os.path.exists(subjectNestedFolder):
			print("WARNING: Subject ", subject, " unsuccessful fMRIprep completion and therefore removed.")
			subjects.remove(subject)
			count = count - 1
	 
	print(count, " successful subjects to be prepared for analysis")

	print("##########")

	print("Renaming sessions folders to be standardised for all subjects.")
	print(subjects)
	for subject in subjects:
		print("	Renaming session folders for subject: ", subject)
		s = renameSessionFolders(fMRIoutputdir,subject)
		if s is not None:
			subjects.remove(subject)

	print("Renaming each file to the new session name")
	for subject in subjects:
		print("	Renaming session files for subject: ", subject)
		renameSessionFiles(fMRIoutputdir,subject)
	print("##########")

	print("Since more than two sessions were seen, reducing to 2 sessions per subject.")
	for subject in subjects:
		print("	Checking for extra sessions for: ", subject)
		extraSessions(fMRIoutputdir,subject)

	print("Move any session that dont have a func image")
	for subject in subjects:
		moveOLDsessions(fMRIoutputdir,subject,source)
	print("##########")





	print("fMRI pre-analysis complete")
	print("Next run MainScript_analysis_CONN.py")

if __name__ == "__main__":
	main()
