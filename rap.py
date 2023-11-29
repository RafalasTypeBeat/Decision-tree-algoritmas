import pandas as pd
import umap.umap_ as umap
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a DataFrame
csv_file_path = 'Wholesale customers data.csv'
df = pd.read_csv(csv_file_path)

# Drop the 'Region' column
df = df.drop('Region', axis=1)

# Replace values in the 'Channel' column
df['Channel'] = df['Channel'].replace({1: 'Horeca', 2: 'Retail'})

# Convert 'Channel' to categorical data type
df['Channel'] = pd.Categorical(df['Channel'])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Extract features for dimensionality reduction
features1 = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = df[features1]

# Normalised data for dimensionality reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features1)

# Extract features for classifying 
features2 = ['Fresh', 'Grocery']
Xclassify = df[features2]

# Normalised data for classifying
Xclassify_scaled = scaler.fit_transform(Xclassify)
Xclassify_scaled = pd.DataFrame(Xclassify_scaled, columns=features2)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # reduce dimensions with UMAP for further calssifying
# reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0, random_state=42)
# umap_classify = reducer.fit_transform(X_scaled)
# X_scaled['Channel'] = df['Channel']

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Count occurrences of each channel
channel_counts = df['Channel'].value_counts()

# Calculate the percentages of each channel
channel_percentages = df['Channel'].value_counts(normalize=True) * 100

# Display the counts
print("Counts:")
print(channel_counts)

# Display the percentages
print("\nPercentages:")
print(round(channel_percentages, 1))

