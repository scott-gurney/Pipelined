import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\User\Downloads\idaSearch_10_09_2023_AD.csv')

rsfMRI_descriptions = [
    #'Axial rsfMRI (Eyes Open)',
    'Axial MB rsfMRI (Eyes Open)',
    'Axial MB rsfMRI',
    #'Axial rsfMRI (EYES OPEN)',
    'Axial MB rsfMRI (Eyes Open) straight no angle',
    #'Axial rsfMRI (Eyes Open) -phase P to A',
    'Axial MB rsfMRI AP',
    #'Axial_rsFMRI_Eyes_Open'
]

DTI_descriptions = [
    #'Axial DTI',
    #'Axial_DTI',
    'Axial MB DTI',
]

# Create a boolean mask for rsfMRI descriptions
rsfMRI_mask = df['Description'].isin(rsfMRI_descriptions)

# Create a boolean mask for DTI descriptions
DTI_mask = df['Description'].isin(DTI_descriptions)

grouped = df.groupby('Subject ID').agg({'Description': 'unique'})

selected_subjects = grouped[(grouped['Description'].apply(lambda x: any(item in x for item in rsfMRI_descriptions))) & (grouped['Description'].apply(lambda x: any(item in x for item in DTI_descriptions)))]

# Get the unique Subject IDs of selected subjects
unique_subjects = selected_subjects.index.unique()

# Print the unique Subject IDs
a = 1
for subject_id in unique_subjects:
    print(str(a) + " " + subject_id)
    a += 1

import matplotlib.pyplot as plt

# Data
categories = ['CN', 'EMCI', 'MCI', 'LMCI', 'AD']
non_multiband = [317, 55, 122, 24, 41]
multiband = [118, 19, 58, 9, 11]

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Set the width of the bars
bar_width = 0.35

# Define the x-axis positions for the bars
x = range(len(categories))

# Create the bars for non-multiband data
plt.bar(x, non_multiband, bar_width, label='Non-multiband')

# Create the bars for multiband data
plt.bar([i + bar_width for i in x], multiband, bar_width, label='Multiband')

# Set labels for x-axis and y-axis
plt.xlabel('Cohort')
plt.ylabel('Subject Count')

# Set the x-axis ticks to be at the center of each group of bars
plt.xticks([i + bar_width/2 for i in x], categories)

# Add a legend
plt.legend()

# Set a title
plt.title('Cohort vs. Subject Count (Multiband vs. Non-multiband)')

# Show the plot
plt.show()