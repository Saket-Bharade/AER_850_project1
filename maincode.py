
#all the import stuff 

import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import joblib
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#making graphs to understand the kind of data we are dealing with

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "All required plots"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

csv_file_path = "Project 1 Data.csv"  
zip_file_path = "C:/Users/saket/Documents/GitHub/AER_850_project1.zip"

with tarfile.open(zip_file_path, "w:gz") as zipf:
    zipf.add(csv_file_path, arcname=os.path.basename(csv_file_path))


with tarfile.open(zip_file_path, "r:gz") as zipf:
 
    csv_filename = os.path.basename(csv_file_path)

 
    zipf.extract(csv_filename)


df = pd.read_csv(csv_filename)
rawdata = df

#understand the data 

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


# machine learning happens here :o



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
X_train = pd.DataFrame(X_train, columns=['X', 'Y', 'Z'])
X_test = pd.DataFrame(X_test, columns=['X', 'Y', 'Z'])



rf_param_grid = {
    'n_estimators': [10, 100, 200],
    'max_depth': [None, 10, 20, 30],
}


rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

best_rf_params = rf_grid_search.best_params_
best_rf_estimator = rf_grid_search.best_estimator_
best_rf_classifier = RandomForestClassifier(n_estimators=best_rf_params['n_estimators'], max_depth=best_rf_params['max_depth'], random_state=42)
best_rf_classifier.fit(X_train, y_train)
rf_y_pred = best_rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)
conf_matrix_rf = confusion_matrix(y_test, rf_y_pred)

print("Random Forest Classifier with Grid Search:")
print(f"Best Hyperparameters: {best_rf_params}")
print(f"Accuracy: {rf_accuracy}")
print("Classification Report:")
print(rf_report)


svmX_train, svmX_test, svmy_train, svmy_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Define the hyperparameter grid for SVM
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
}

svm_grid_search = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

svm_grid_search.fit(svmX_train, svmy_train)
best_svm_params = svm_grid_search.best_params_
best_svm_estimator = svm_grid_search.best_estimator_
best_svm_classifier = SVC(C=best_svm_params['C'], kernel=best_svm_params['kernel'], random_state=42)
best_svm_classifier.fit(svmX_train, svmy_train)
svm_y_pred = best_svm_classifier.predict(svmX_test)
svm_accuracy = accuracy_score(svmy_test, svm_y_pred)
svm_report = classification_report(svmy_test, svm_y_pred)


print("Support Vector Machine (SVM) with Grid Search:")
print(f"Best Hyperparameters: {best_svm_params}")
print(f"Accuracy: {svm_accuracy}")
print("Classification Report:")
print(svm_report)


knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
}


knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
knn_grid_search.fit(X_train, y_train)
best_knn_params = knn_grid_search.best_params_
best_knn_estimator = knn_grid_search.best_estimator_
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'], weights=best_knn_params['weights'])
best_knn_classifier.fit(X_train, y_train)
knn_y_pred = best_knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_report = classification_report(y_test, knn_y_pred)

print("K-Nearest Neighbors (K-NN) with Grid Search:")
print(f"Best Hyperparameters: {best_knn_params}")
print(f"Accuracy: {knn_accuracy}")
print("Classification Report:")
print(knn_report)



print("Confusion Matrix:")
print(conf_matrix_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 14), yticklabels=range(1, 14))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(IMAGES_PATH, "confusion_matrix_plot.png"), format="png", dpi=300)
plt.show()

#we see that RF has the best accuracy from the results so we will only use CVM


joblib.dump(best_rf_classifier, 'best_rf_model.pkl')


def predict_steps(model_path, coordinates_list):
    loaded_model = joblib.load(model_path)
    predictions = loaded_model.predict(coordinates_list)
    return predictions

model_path = 'best_rf_model.pkl'
coordinates_to_predict = [
    [9.375, 3.0625, 1.51],
    [6.995,5.125,0.3875],
    [0,3.0625,1.93],
    [9.4,3,1.8],
    [9.4,3,1.3],
   
]

#print predictions

predicted_steps = predict_steps(model_path, coordinates_to_predict)

for i, predicted_step in enumerate(predicted_steps):
    print(f"\nPredicted Step {i + 1}: {predicted_step}")

