# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#importing the libraries 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


#Reading the data
dataset=pd.read_excel('C:\\Users\\sagarwal4\\Downloads\\post stroke\\stroke data 2.xlsx')
features = dataset.iloc[:, 0:15].values
label = dataset.iloc[:, 15:].values

#Feature scaling
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=label

#Encode the target variable to start from 0
#label_encoder = LabelEncoder()
#y = label_encoder.fit_transform(y) 

#splitting the data
label_encoder = LabelEncoder()
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)
y_train = y_train.ravel()
y_test = y_test.ravel()


# Import necessary libraries
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# Step 1: Apply PCA to the Dataset
# Define the number of components
n_components = 10  # Set this to the number of components you want to retain
pca = PCA(n_components=n_components)

# Fit PCA on features and transform the data
x_pca = pca.fit_transform(x)

# Step 2: Visualize Explained Variance
# Plot a scree plot (variance explained by each principal component)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid()
plt.show()

# Optional: Print the explained variance ratio
print("Explained variance ratio for each component:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

# Importing SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# SVM Classifier
svm_model = SVC(kernel='rbf', random_state=42)

# Train
svm_model.fit(x_train, y_train)

# Predict
y_pred = svm_model.predict(x_test)

# Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Optional: Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)
y_pred = grid.best_estimator_.predict(x_test)

# Evaluate again after tuning
print("Confusion Matrix after Tuning:\n", confusion_matrix(y_test, y_pred))
print("Classification Report after Tuning:\n", classification_report(y_test, y_pred))
print("Accuracy Score after Tuning:", accuracy_score(y_test, y_pred))

# Plot the Confusion Matrix
plt.figure(dpi=300)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SVM without PCA")
plt.show()




#SVM with PCA  
# Import necessary libraries

# Step 1: Apply PCA
# Determine the number of components to retain (adjust n_components based on explained variance)
pca = PCA(n_components=0.95)  # Retain 95% variance
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Step 2: Train SVM Model with PCA
svm_model = SVC(kernel='linear', random_state=42)  # Linear kernel as an example
svm_model.fit(x_train_pca, y_train)

# Step 3: Predict Test Set
y_pred = svm_model.predict(x_test_pca)

# Step 4: Evaluate the Model
print("Confusion Matrix (SVM with PCA):\n", confusion_matrix(y_test, y_pred))
print("Classification Report (SVM with PCA):\n", classification_report(y_test, y_pred))
print("Accuracy Score (SVM with PCA):", accuracy_score(y_test, y_pred))

# Step 5: Plot Confusion Matrix
plt.figure(dpi=300)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SVM (With PCA)")
plt.show()

# Step 6: Perform Cross-Validation
cv_scores = cross_val_score(svm_model, x_train_pca, y_train, cv=5)  # 5-fold cross-validation
print("Cross-Validation Scores (SVM with PCA):", cv_scores)
print("Mean CV Accuracy (SVM with PCA):", cv_scores.mean())




##Decision Tree
# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)

# Predict the Test Set
y_pred = dt_model.predict(x_test)

# Evaluate Decision Tree Model
print("Confusion Matrix after Tuning:\n", confusion_matrix(y_test, y_pred))
print("Classification Report after Tuning:\n", classification_report(y_test, y_pred))
print("Accuracy Score after Tuning:", accuracy_score(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(dpi=300)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Decision Tree without PCA")
plt.show()

# Train Decision Tree Classifier with cross-validation
dt_model = DecisionTreeClassifier(random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(dt_model, x_train, y_train, cv=5, scoring='accuracy')  # 5-fold cross-validation
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Standard Deviation of CV Accuracy:", cv_scores.std())



#Decision Tree with PCA
# Apply PCA (fit on training set, transform both train and test sets)
pca = PCA(n_components=10)  # Retain 10 components or adjust as necessary
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Train Decision Tree on PCA-Transformed Data
dt_model_pca = DecisionTreeClassifier(random_state=42)
dt_model_pca.fit(x_train_pca, y_train)

# Evaluate the Model
y_pred_pca = dt_model_pca.predict(x_test_pca)

# Confusion Matrix and Metrics
print("Confusion Matrix (Decision Tree with PCA):\n", confusion_matrix(y_test, y_pred_pca))
print("Classification Report (Decision Tree with PCA):\n", classification_report(y_test, y_pred_pca))
print("Accuracy Score (Decision Tree with PCA):", accuracy_score(y_test, y_pred_pca))

# Plot Confusion Matrix
plt.figure(dpi=300)
cm = confusion_matrix(y_test, y_pred_pca)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model_pca.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Decision Tree (With PCA)")
plt.show()



#Random Forest 
# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust hyperparameters here
rf_model.fit(x_train, y_train)

#  Predict the test set
y_pred = rf_model.predict(x_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Plot the Confusion Matrix
plt.figure(dpi=300)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest without PCA")
plt.show()

# Optional - Perform Cross-Validation
cv_scores = cross_val_score(rf_model, x_train, y_train, cv=5)  # Perform 5-fold cross-validation
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

##Random Forest with PCA 
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Apply PCA
pca = PCA(n_components=10)  # Adjust the number of components as needed
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Step 2: Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)  # You can tune hyperparameters here
rf_model.fit(x_train_pca, y_train)

# Step 3: Predict the test set
y_pred = rf_model.predict(x_test_pca)

# Step 4: Evaluate the model
print("Confusion Matrix (PCA - Random Forest):\n", confusion_matrix(y_test, y_pred))
print("Classification Report (PCA - Random Forest):\n", classification_report(y_test, y_pred))
print("Accuracy Score (PCA - Random Forest):", accuracy_score(y_test, y_pred))

# Step 5: Plot the Confusion Matrix
plt.figure(dpi=300)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest (with PCA)")
plt.show()

# Step 6: Optional - Perform Cross-Validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf_model, pca.transform(x), y, cv=5)  # Perform 5-fold cross-validation
print("Cross-Validation Scores (PCA - Random Forest):", cv_scores)
print("Mean CV Accuracy (PCA - Random Forest):", cv_scores.mean())




#ADABoost Classifier
# Import necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Initialize AdaBoost classifier
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(x_train, y_train)

# Predict the test set
y_pred = ada_model.predict(x_test)

# Evaluate the model
print(f"Confusion Matrix (AdaBoost without PCA):\n", confusion_matrix(y_test, y_pred))
print(f"Classification Report (AdaBoost without PCA):\n", classification_report(y_test, y_pred))
print(f"Accuracy Score (AdaBoost without PCA):", accuracy_score(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(dpi=300)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - AdaBoost without PCA")
plt.show()

# Perform Cross-Validation
cv_scores = cross_val_score(ada_model, x_train, y_train, cv=5)  # Perform 5-fold cross-validation
print(f"Cross-Validation Scores (AdaBoost without PCA):", cv_scores)
print(f"Mean CV Accuracy (AdaBoost without PCA):", cv_scores.mean())



#AdaBoost Classifier with PCA
# Import necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Apply PCA
n_components = 10  # Set the number of PCA components
pca = PCA(n_components=n_components)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

print(f"Explained Variance Ratio (PCA): {pca.explained_variance_ratio_}")
print(f"Number of Components Used: {pca.n_components_}")

# Initialize AdaBoost classifier
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(x_train_pca, y_train)

# Predict the test set
y_pred = ada_model.predict(x_test_pca)

# Evaluate the model
print(f"Confusion Matrix (AdaBoost with PCA):\n", confusion_matrix(y_test, y_pred))
print(f"Classification Report (AdaBoost with PCA):\n", classification_report(y_test, y_pred))
print(f"Accuracy Score (AdaBoost with PCA):", accuracy_score(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(dpi=300)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - AdaBoost (with PCA)")
plt.show()

# Perform Cross-Validation
cv_scores = cross_val_score(ada_model, x_train_pca, y_train, cv=5)  # Perform 5-fold cross-validation
print(f"Cross-Validation Scores (AdaBoost with PCA):", cv_scores)
print(f"Mean CV Accuracy (AdaBoost with PCA):", cv_scores.mean())

