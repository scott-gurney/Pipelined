import copy
import glob
import sys

import numpy as np
import pandas as pd
import os
import nilearn

from nilearn import plotting
from nilearn.image import resample_img, new_img_like, mean_img
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from nilearn.connectome import GroupSparseCovarianceCV, ConnectivityMeasure
from nilearn import datasets

# MSDL Dataset is probabilistic, meaning the parcellations by voxel aren't strict, there is overlap
from nilearn.plotting import plot_roi

#dataset = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-1mm")
#atlas_filename = dataset.maps
#labels = dataset.labels


dataset = datasets.fetch_atlas_msdl()
atlas_filename = dataset.maps
labels = dataset.labels
coords = dataset.region_coords


# Cohort is a class for the types of cohorts to keep track of their subjects n that
class Cohort:
    def __init__(self, name):
        self.name = name
        self.subjects = []

    def setSubjects(self, arr):
        self.subjects = arr

    def getSubjects(self):
        return self.subjects

    def getName(self):
        return self.name


root_dir = " "
pathed = []


# Iterate through all directories which match sub* format and appends them to list
# Returns pathlist
def getPathData(path):
    pathList = []
    basepath = path
    for entry in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath, entry)) and entry[:3] == "sub":
            pathList.append(entry)
        else:
            print("Warning, subfolder titled: " + entry + " does not match spec and is being ignored.")

    return (pathList)


# Extract fMRI bold data using the fMRIprep
def getBoldData(cohort):
    subjectPathList = []
    length = len(cohort.subjects)
    for i in range(length):
        path_list = []
        suffix = "Asym_desc-preproc_bold.nii.gz"
        subject_list = cohort.getSubjects()
        prefix = "sub-ADNI"
        file_name = (prefix + subject_list[i])
        # print("Processing subject " + file_name)
        dir_pattern = os.path.join(root_dir, file_name, file_name, "ses-*")

        for dir_path in glob.glob(dir_pattern):
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    if filename.endswith(suffix):
                        file_path = os.path.join(dirpath, filename)
                        path_list.append((file_path))
        subjectPathList.append(path_list)
    return subjectPathList


# Extracts confounds
def getConfounds(cohort):
    subjectPathList = []
    length = len(cohort.subjects)
    for i in range(length):
        path_list = []
        suffix = "desc-smoothAROMAnonaggr_bold.nii.gz"
        prefix = "sub-ADNI"
        subject_list = cohort.getSubjects()
        file_name = (prefix + subject_list[i])
        print("Processing subject " + file_name)
        # CUT
        dir_pattern = os.path.join(root_dir, file_name, file_name, "ses-*")
        print(dir_pattern)
        for dir_path in glob.glob(dir_pattern):
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    if filename.endswith(suffix):
                        file_path = os.path.join(dirpath, filename)
                        path_list.append((file_path))
        # SNIP
        subjectPathList.append(path_list)
    return subjectPathList


def searchPath(path, prefix):
    pref = prefix.strip()
    corr_file = " "
    for file in os.listdir(path):
        if file.startswith(pref):
            corr_file = os.path.join(path, file)
            return corr_file
    return None


def getTimeSeries(bold, conf, name):
    func = bold
    confounds_fn = conf
    # We need to load the confounds and fill nas
    # The anatomy file for the same type as the BOLD data
    # Imported mask from CONN results

    mask = nilearn.image.load_img(r'/gpfs/scratch/uqe18wbu/fMRIoutput/NewMasjs/Occipital.Mask.nii')
    mask_data = mask.get_fdata()
    mask_indexes = (mask_data == 1)

    confounds = nilearn.interfaces.fmriprep.load_confounds(confounds_fn,
                                                           strategy=("motion", "wm_csf", "global_signal",
                                                                     "compcor", "ica_aroma", "scrub", "high_pass"),
                                                           motion="full", wm_csf="full", global_signal="full",
                                                           compcor="temporal_anat_combined",
                                                           n_compcor="all",
                                                           ica_aroma="full")


    brain_func = nilearn.image.load_img(func)
    func_shape = nilearn.image.index_img(brain_func, 0)

   #mask = resample_img(mask, target_affine=brain_func.affine, target_shape=func_shape.shape, interpolation='nearest')

    # The percentage is the threshold for binarization, as the mask contains varying voxel strengths
    binarised = nilearn.image.binarize_img(mask, "99%")

    # This part extracts the time series
    #masker = NiftiLabelsMasker(atlas_filename, labels, mask_img=binarised)
    masker = NiftiMapsMasker(atlas_filename, standardize=True, memory="nilearn_cache", verbose=1,
                             resampling_target="mask", mask_img=binarised)
    the_masked = masker.fit_transform(brain_func, confounds=confounds)

    # This part builds the connectivity matrix based on the time series
    return the_masked

text = input("Have you run FilePathMaker.py on each cohort subject file you wanted to analyse? y/n \n")

if text.lower() == "y":
    print("Good, we can proceed")
else:
    print("You're going to have to do that first before continuing.")
    sys.exit()


print("Welcome to the pipeline thing, I'm going to need some input:")

# This is the path input

while True:
    try:
        root_dir = input("Please give the path containing all subjects, ensure that they are from fMRIprep output\n")
        # root_dir = r"C:\Users\jamie\Downloads\ADNI"
        pathed = getPathData(root_dir)
        break
    except FileNotFoundError and OSError:
        print("File not found, please try again")

length = len(pathed)
print("The path has a total of " + str(length) + " subjects\n")

# This is the number of cohorts input
while True:
    try:
        numberOfCohorts = int(input("Please input the number of cohorts (Input as an integer, cohorts meaning CN, EMCI,"
                                    " etc.\n"))
        if numberOfCohorts > 5:
            raise NameError("That is too many cohorts, the maximum is 5\n")
        break
    except ValueError:
        print("That's not a Number!\n")
    except NameError:
        print("Try again\n")

cohorts = []
counter = 0
while counter != numberOfCohorts:
    try:
        temp = input("Please input what cohorts are present, in an acronymistic format, and one at"
                     " a time, pressing enter after each cohort (Input such as CN, AD, EMCI\n")
        correctInputs = ["AD", "CN", "EMCI", "MCI", "LMCI"]
        if temp.upper() not in correctInputs:
            raise NameError("Wrong input error")
        else:
            newC = Cohort(temp.upper())
            cohorts.append(newC)
            counter += 1
    except NameError:
        print("Please try again using CN, EMCI, MCI, LMCI, or AD")

names = None
for j in cohorts:
    if names is None:
        names = input("Please input FilePathMaker.py output txt file location (e.g. ~/FilePaths/)\n")
    tempList = []
    file = open(searchPath(names, getattr(j, "name")))
    if file == None:
        print("Subjects file for cohort " + getattr(j, "name") + " not found.")
    for i in file:
        i = i.replace("_", "")
        for k in i.split(","):
            k = k.strip()
            tempList.append(k)
    j.setSubjects(tempList)
    print("For the following cohort: " + getattr(j, "name") + " the following subjects have been chosen:\n")
    for namess in j.getSubjects():
        print(namess)
# This successfully extracts the fMRI BOLD data and confounds
# For each cohort it then
for cohort in cohorts:

    directory = getattr(cohort, "name")
    path = os.path.join(root_dir, directory)
    # Make path to a folder for cohort file name
    try:
        os.mkdir(path)
    except OSError as error:
        print("Directory '%s' can not be created" % directory)

    # Path to each bold scan file
    pathToBold = getBoldData(cohort)
    # Path to each confound scan file
    pathToConfs = getConfounds(cohort)
    for subjects in range(len(pathToBold)):
        combinedScan = []
        for session_scans in range(len(pathToBold[subjects])):
            print("Subject " + str(subjects) + " scan " + str(session_scans))
            try:
                current_scan_confound = copy.deepcopy(pathToConfs[subjects][session_scans])
                current_scan_bold = copy.deepcopy(pathToBold[subjects][session_scans])
            except IndexError:
                print("Subject " + str(subjects) + " has issues with index")
                continue
            try:
                timeseries = getTimeSeries(current_scan_bold, current_scan_confound, getattr(cohort, "name"))
            except ValueError as e:
                print(e)
                continue
            combinedScan.append(timeseries)
        for j in range(len(combinedScan)):
            filePath = os.path.join(path, (
                    getattr(cohort, "name") + "_n_" + str(subjects + 1) + "_scan_" + str(j + 1) + "_mask_" + "_.csv"))
            np.savetxt(filePath, combinedScan[j], delimiter=",")