import os
import re

#Set the directory you want to start from
# Set the output file name
# Process folder names and save them to the output file

first = input("First time using? y/n\n")
if first.lower() == "y":
    print("This script makes a separate .txt file which is then used in TimeSeries.py, in order to do that you need to input\n"
          "the directory in which your preprocessed subjects are stored needs to be given. Please organise the paths in this\n"
          "fashion:\n /gpfs/scratch/uqe18wbu/ADNI_proc/CN/fMRIoutput\n /gpfs/scratch/uqe18wbu/ADNI_proc/EMCI/fMRIoutput\n "
          "/gpfs/scratch/uqe18wbu/ADNI_proc/MCI/fMRIoutput\n"
          "etc. Obviously don't do it in my storage, do it in yours. The fact that all of the subdirectories are within one\n"
          "named ADNI_proc is important, so keep it to this structure. The output will then be saved to a created folder\n"
          "named FilePaths. The required input using the above as an example would be /gpfs/scratch/uqe18wbu/")

dir_path = input("Please input the directory path:")

def find_and_execute_function(directory):
    # Check if the provided directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Search for the "ADNI_proc" folder
    adni_proc_folder = None
    for root, dirs, files in os.walk(directory):
        if "ADNI_proc" in dirs:
            adni_proc_folder = os.path.join(root, "ADNI_proc")
            break

    if adni_proc_folder is None:
        print("The 'ADNI_proc' folder was not found.")
        return

    # Define the naming criteria
    criteria = ["CN", "EMCI", "MCI", "LMCI", "AD"]

    for root, dirs, files in os.walk(adni_proc_folder):
        for folder in dirs:
            if folder in criteria:
                stringed = (folder + "_NAMES.txt")
                subfolder_path = os.path.join(root, "FilePaths")
                searchy = os.path.join(root, folder)
                if not os.path.exists(subfolder_path):
                    try:
                        # Create the subfolder
                        os.makedirs(subfolder_path)
                        print(f"Subfolder '{folder}' created in '{directory}'.")
                    except OSError as e:
                        print(f"Error creating subfolder '{folder}' in '{directory}': {str(e)}")
                else:
                    print(f"Processing '{folder}'")
                subfolders = [f.name for f in os.scandir(searchy) if f.is_dir()]
                output_file = os.path.join(subfolder_path, stringed)

                with open(output_file, "w") as output:
                    for folder in subfolders:
                        converted_name = convert_name(folder)
                        output.write(converted_name + ",")




#Function to convert folder names
def convert_name(folder_name):
    match = re.match(r"sub-ADNI(\d{3})S(\d{4})", folder_name)
    if match:
        first_part, second_part = match.group(1), match.group(2)
        return f"{first_part}_S_{second_part}"
    return folder_name

find_and_execute_function(dir_path)