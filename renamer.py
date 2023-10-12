import os

# Specify the directory path
directory = r'C:\Users\User\Downloads\20230727\REST-fMRI'

# # Get the directory names from the path
# directory_names = directory.split(os.path.sep)
# prefix = "_".join(directory_names[-2:])  # Use the last two directory names as the prefix
#
# # Iterate over all files in the directory
# for filename in os.listdir(directory):
#     # Check if the file is a regular file (not a directory)
#     if os.path.isfile(os.path.join(directory, filename)):
#         # Create a new filename by combining the prefix and the original filename
#         new_filename = f'{prefix}_{filename}'
#
#         # Construct the full paths for the old and new filenames
#         old_filepath = os.path.join(directory, filename)
#         new_filepath = os.path.join(directory, new_filename)
#
#         # Rename the file
#         os.rename(old_filepath, new_filepath)
#         print(f'Renamed: {filename} -> {new_filename}')

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(os.path.join(directory, filename)):
        # Separate the filename and the extension
        base_filename, file_extension = os.path.splitext(filename)

        # Remove the last three characters from the base filename
        new_base_filename = base_filename[:-7]

        # Construct the new full filename
        new_filename = new_base_filename + file_extension

        # Construct the full paths for the old and new filenames
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {filename} -> {new_filename}')