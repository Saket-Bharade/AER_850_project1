import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "histogram_plots"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    


import os
import tarfile
import pandas as pd

# Define the paths and filenames
csv_file_path = "Project 1 Data.csv"  # Provide the path to your local CSV file
zip_file_path = "C:/Users/saket/Documents/GitHub/AER_850_project1.zip"  # Provide the desired output ZIP file path

# Step 1: Convert the CSV file to a ZIP file
with tarfile.open(zip_file_path, "w:gz") as zipf:
    zipf.add(csv_file_path, arcname=os.path.basename(csv_file_path))

# Step 2: Open the ZIP file with pandas
with tarfile.open(zip_file_path, "r:gz") as zipf:
    # Find the name of the CSV file within the ZIP file
    csv_filename = os.path.basename(csv_file_path)

    # Extract the CSV file from the ZIP file
    zipf.extract(csv_filename)

# Step 3: Read the CSV file using pandas
df = pd.read_csv(csv_filename)
rawdata = df
# Now you can work with the DataFrame 'df' as needed
print (rawdata.head())
print (rawdata.info())

corr_matrix = rawdata.corr()
corr_matrix["Step"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["X", "Y", "Z","Step"]
scatter_matrix(rawdata[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")

rawdata.plot(kind="scatter", x="Step", y="X", alpha=0.1)
plt.axis([0, 13, 0, 10])
save_fig("X_vs_Step")

rawdata.plot(kind="scatter", x="Step", y="Y", alpha=0.1)
plt.axis([0, 13, 0, 6])
save_fig("Y_vs_Step")

rawdata.plot(kind="scatter", x="Step", y="Z", alpha=0.1)
plt.axis([0, 13, 0, 2])
save_fig("Z_vs_Step")

X = rawdata[['X', 'Y', 'Z']]
y = rawdata['Step']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 14), yticklabels=range(1, 14))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()




