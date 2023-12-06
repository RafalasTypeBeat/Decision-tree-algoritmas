import pandas as pd
import umap.umap_ as umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics

# Read the CSV file into a DataFrame
csv_file_path = 'Wholesale customers data.csv'
df = pd.read_csv(csv_file_path)

# Drop the 'Region' column
df = df.drop('Region', axis=1)

# Replace values in the 'Channel' column
#df['Channel'] = df['Channel'].replace({1: 'Horeca', 2: 'Retail'})

# Convert 'Channel' to categorical data type
#df['Channel'] = pd.Categorical(df['Channel'])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Extract features for dimensionality reduction
features1 = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
# create features and labels dataframes
X_features = df[features1]
labels = df['Channel']

# Normalised data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
X_scaled = pd.DataFrame(X_scaled, columns=features1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # reduce dimensions with UMAP for further calssifying
# reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0, random_state=42)
# umap_classify = reducer.fit_transform(X_scaled)
# umap_classify['Channel'] = df['Channel']

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # Count occurrences of each channel
# channel_counts = df['Channel'].value_counts()

# # Calculate the percentages of each channel
# channel_percentages = df['Channel'].value_counts(normalize=True) * 100

# # Display the counts
# print("Counts:")
# print(channel_counts)

# # Display the percentages
# print("\nPercentages:")
# print(round(channel_percentages, 1))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Add the channel column
X_scaled.insert(0, 'Channel', df['Channel'])

# Separate data into class 1 and class 2
class_1_data = X_scaled[X_scaled['Channel'] == 1]
class_2_data = X_scaled[X_scaled['Channel'] == 2]

# Randomly select 22 instances from each class
class_1_sample = class_1_data.sample(n=22)
class_2_sample = class_2_data.sample(n=22)

# Combine the samples to create the balanced test set
df_test_balanced = pd.concat([class_1_sample, class_2_sample])

# Remove the samples used for the test set from the original dataframe
X_remaining = X_scaled.drop(df_test_balanced.index)

print("Size of X_remaining:", X_remaining.shape)

# Split the remaining data into train and validation sets. 44/396 sets the proportions for splitting. We want to have 10% of original data in validate set.
df_train, df_valid = train_test_split(X_remaining, test_size=44/396) #set random_state for consistent results

# Verify the distribution of labels in the balanced test set
test_label_distribution_balanced = df_test_balanced['Channel'].value_counts()
print("Balanced Test Set Label Distribution:")
print(test_label_distribution_balanced)

# Verify the distribution of labels in the validation set
valid_label_distribution = df_valid['Channel'].value_counts()
print("\nValidation Set Label Distribution:")
print(valid_label_distribution)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#X contains features, y contains labels
# X_train, X_test, y_train, y_test = train_test_split(X_features, labels, test_size=0.2, random_state=42)

# # Create the decision tree classifier
# clf = DecisionTreeClassifier()

# # Train the classifier
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)

# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print("Precision:", metrics.precision_score(y_test, y_pred))
# print("Recall:", metrics.recall_score(y_test, y_pred))