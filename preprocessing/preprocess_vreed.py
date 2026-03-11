# Import libraries
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import pickle
from glob import glob
from datetime import datetime
from collections import OrderedDict
#from tqdm import tqdm

### Load DAT files
path = kagglehub.dataset_download("lumaatabbaa/vr-eyes-emotions-dataset-vreed")
dat_directory = os.path.join(path, '05 ECG-GSR Data', '01 ECG-GSR Data (Pre-Processed)')
dat_files = [f for f in os.listdir(dat_directory) if f.endswith('.dat')]

data_list = []
for file_name in dat_files:
    file_path = os.path.join(dat_directory, file_name)
    with open(file_path, 'rb') as f:
        file_data = pickle.load(f)
    data_list.append({
        'ID': file_name.split('_')[0],
        'Data': file_data
    })

df = pd.DataFrame(data_list)
df

# for index, row in df.iterrows():
#     participant_id = row['ID']
#     arrays = row['Data']['Data']
#     print(f"Participant ID: {participant_id}")
#     for i, arr in enumerate(arrays):
#         print(f"  Array {i + 1}: {arr.shape[0]} records")

### Load Ratings
excel_file = os.path.join(path, '03 Self-Reported Questionnaires', '02 Post Exposure Ratings.xlsx')
labels_df = pd.read_excel(excel_file)
labels_df

# Define custom order clearly
custom_order = ['Baseline', 'JNG', 'PRS', 'BRZ', 'RST', 'DST', 'EXR','BNS', 'BOT', 'RFS', 'RPW', 'TNT', 'ZMZ']
#               [4,           1,     2,     0,     1,     0,    3,     1,     2,     2,     0,     3
# This order was determined by first looking at the number of timestamps for each subject and trial as above in the DAT files in dat_directory = os.path.join(path, '05 ECG-GSR Data', '01 ECG-GSR Data (Pre-Processed)').
# It was found that the DAT files for each participant were in the same order aside for Participants 118, 130, 101 who had not watched some videos altogether.
# Then these timestamps were divided by 2000 (sampling rate) to obtain the duration of the DAT files.
# This duration was then cross referenced with the NOTED video stimuli durations, to figure the close  and order.
# The order was also cross referenced with the 'Labels' column in the DAT files to make sure the category was correlated.
# There were some uncertainities however intra-label, for eg JNG and BNS had similar number of timetsamps, TNT, ZMD also not clear so it might be worth checking the results with them interchanged.

# Sort labels clearly
labels_df['Str_Code'] = pd.Categorical(labels_df['Str_Code'], custom_order, ordered=True)
labels_df = labels_df.sort_values(['ID', 'Str_Code'])

# Convert categorical to string before aggregation
labels_df['Str_Code'] = labels_df['Str_Code'].astype(str)

# Aggregate labels clearly with baseline
labels_aggregated = labels_df.groupby('ID').agg({
    'POST_Valence': lambda x: [4.0] + list(x),
    'POST_Arousal': lambda x: [4.0] + list(x),
    'Str_Code': lambda x: ['Baseline'] + list(x),
    'Num_Code': lambda x: [4] + list(x)
}).reset_index()

labels_aggregated

### Merge
# Ensure IDs are strings for merging clearly
labels_aggregated['ID'] = labels_aggregated['ID'].astype(str)
df['ID'] = df['ID'].astype(str)

# Merge labels and DAT data clearly
df = pd.merge(df, labels_aggregated, on='ID', how='left')
df

# Update 'Data' dictionaries clearly
def update_data_dict(row):
    row['Data'].update({
        'POST_Valence': row['POST_Valence'],
        'POST_Arousal': row['POST_Arousal'],
        'Str_Code': row['Str_Code'],
        'Num_Code': row['Num_Code']
    })
    return row['Data']

df['Data'] = df.apply(update_data_dict, axis=1)
df

# Drop redundant columns clearly
df.drop(columns=['POST_Valence', 'POST_Arousal', 'Str_Code', 'Num_Code'], inplace=True)
df

### Explode Data

# Prepare exploded data (efficient method)
expanded_rows = []

for _, row in df.iterrows():
    data = row['Data']
    for trial_idx, (label, data_array, v, a, s, n) in enumerate(zip(
        data['Labels'], data['Data'],
        data['POST_Valence'], data['POST_Arousal'],
        data['Str_Code'], data['Num_Code']
    )):
        if label == 4:
            continue  # Skip baseline clearly here

        # for point in data_array:
        #     expanded_rows.append({
        #         'ID': int(row['ID']),
        #         'Label': label,
        #         'GSR': point[0],
        #         'ECG': point[1],
        #         'POST_Valence': v,
        #         'POST_Arousal': a,
        #         'Str_Code': s,
        #         'Num_Code': n,
        #         'trial': trial_idx
        #     })
        
        for point in data_array[::8]:   # downsample BEFORE expanding
            expanded_rows.append({
                'ID': int(row['ID']),
                'Label': label,
                'GSR': point[0],
                'ECG': point[1],
                'POST_Valence': v,
                'POST_Arousal': a,
                'Str_Code': s,
                'Num_Code': n,
                'trial': trial_idx
            })

exploded_df = pd.DataFrame(expanded_rows)
exploded_df

### Exclude IDs

ids_to_drop = [101, 102, 103, 115, 118, 119, 121, 130]
#130, 118, 101 dropped as they did not complete trials, but the rest were dropped based on analysis of extracted features (paper did not mention which ones but dropped these ones)
exploded_df = exploded_df[~exploded_df['ID'].isin(ids_to_drop)]
exploded_df = exploded_df.reset_index(drop=True)
exploded_df

### Label Encode

from sklearn.preprocessing import LabelEncoder

# Add the combined 'ID_trial' column
exploded_df['ID_video'] = exploded_df['ID'].astype(str) + '_' + exploded_df['Num_Code'].astype(str)

# Label encode the 'subject_video_encoded' column
label_encoder = LabelEncoder()
exploded_df['participant_trial_encoded'] = label_encoder.fit_transform(exploded_df['ID_video'])
exploded_df

# Convert the 'feltARSL' column to numeric, coerce errors to NaN
exploded_df['AR'] = pd.to_numeric(exploded_df['POST_Arousal'], errors='coerce')
# Now create the new column based on the 'feltARSL' column
exploded_df['AR_Rating'] = exploded_df['AR'].apply(lambda x: 1 if x >= 5 else 0)

# Convert the 'feltARSL' column to numeric, coerce errors to NaN
exploded_df['VA'] = pd.to_numeric(exploded_df['POST_Valence'], errors='coerce')
# Now create the new column based on the 'feltARSL' column
exploded_df['VA_Rating'] = exploded_df['VA'].apply(lambda x: 1 if x >= 5 else 0)

exploded_df.reset_index(drop=True, inplace=True)
exploded_df

# Count the number of occurrences of each unique value in the 'VA_Rating' column
va_rating_counts = exploded_df['VA_Rating'].value_counts()
percentage_zeros = (va_rating_counts[0] / exploded_df['VA_Rating'].count()) * 100
print(va_rating_counts, percentage_zeros)

# Count the number of occurrences of each unique value in the 'VA_Rating' column
ar_rating_counts = exploded_df['AR_Rating'].value_counts()
percentage_zeros = (ar_rating_counts[0] / exploded_df['AR_Rating'].count()) * 100
print(ar_rating_counts, percentage_zeros)

### Downsample

# Define the downsample function
# def downsample_data(df, factor):
#     # Keep every 'factor'-th data point
#     return data[::factor]

# # Apply the downsample function to GSR and ECG for each ID_trial
# downsampled_data = exploded_df.groupby('participant_trial_encoded').apply(lambda x: x.iloc[::8, :])

# # Reset the index to maintain a clean DataFrame structure
# downsampled_data = downsampled_data.reset_index(drop=True)
# downsampled_data

# Drop columns
downsampled_data = exploded_df
columns_to_drop = ['Unnamed: 0', 'Label', 'trial', 'POST_Valence', 'POST_Arousal']
downsampled_data = downsampled_data.drop(columns=columns_to_drop, errors='ignore')
downsampled_data = downsampled_data.sort_values(['ID', 'Num_Code']) # for MTL and STL only
downsampled_data = downsampled_data.reset_index(drop=True)
downsampled_data

### Range Check

print(f"ECG Range: {downsampled_data['ECG'].min()} - {downsampled_data['ECG'].max()}")
print(f"GSR Range: {downsampled_data['GSR'].min()} - {downsampled_data['GSR'].max()}")

### Scale

from sklearn.preprocessing import StandardScaler

def scale_columns(group):
    scaler = StandardScaler()
    group[['ECG_scaled', 'GSR_scaled']] = scaler.fit_transform(group[['ECG', 'GSR']])
    return group

df = downsampled_data.groupby('ID').apply(scale_columns).reset_index(drop=True)
df

### Class balance

# Count the number of occurrences of each unique value in the 'VA_Rating' column
va_rating_counts = df['VA_Rating'].value_counts()

# Calculate the percentage of '0' values in the 'VA_Rating' column
percentage_zeros = (va_rating_counts[0] / df['VA_Rating'].count()) * 100

print(va_rating_counts)
print(percentage_zeros)

# Count the number of occurrences of each unique value in the 'VA_Rating' column
ar_rating_counts = df['AR_Rating'].value_counts()

# Calculate the percentage of '0' values in the 'VA_Rating' column
percentage_zeros = (ar_rating_counts[0] / df['AR_Rating'].count()) * 100

print(ar_rating_counts)
print(percentage_zeros)

import matplotlib.pyplot as plt
import seaborn as sns

# Count ratings
va_rating_counts = df['VA_Rating'].value_counts().sort_index()
ar_rating_counts = df['AR_Rating'].value_counts().sort_index()

# Calculate percentage of 0s
va_zero_pct = (va_rating_counts[0] / df['VA_Rating'].count()) * 100 if 0 in va_rating_counts else 0
ar_zero_pct = (ar_rating_counts[0] / df['AR_Rating'].count()) * 100 if 0 in ar_rating_counts else 0

# Set up subplots
plt.figure(figsize=(12, 5))

# VA Rating Distribution
plt.subplot(1, 2, 1)
sns.barplot(x=va_rating_counts.index.astype(str), y=va_rating_counts.values, palette="Blues_d")
plt.title(f'VA_Rating Distribution\n0s: {va_zero_pct:.2f}%')
plt.xlabel('VA_Rating')
plt.ylabel('Count')

# AR Rating Distribution
plt.subplot(1, 2, 2)
sns.barplot(x=ar_rating_counts.index.astype(str), y=ar_rating_counts.values, palette="Greens_d")
plt.title(f'AR_Rating Distribution\n0s: {ar_zero_pct:.2f}%')
plt.xlabel('AR_Rating')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

#df.to_csv('/content/drive/MyDrive/Phase A/data/VREED_data_v2.csv', index=False)
output_path = "data/processed/vreed_processed.csv"
df.to_csv(output_path, index=False)
print(f"Saved processed dataset to {output_path}")