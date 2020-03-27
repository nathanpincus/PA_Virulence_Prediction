# # Using Random Forest to predict virulence based on Unique AGEs in the BLSF_noCF dataset

# Load in modules needed for data inport/proccessing, analysis, and plotting
# For data inport/proccessing/export
import numpy as np
import pandas as pd
import scipy.stats
import pickle

#For machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve

#For plotting
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

#For plotting learning curves
import sys
sys.path.append('/path/to/python_functions/')
import plot_learning_curves_function_95CI

# Import Unique Groups of SE for BLSF_noCF dataset and label with virulence data

# Read in Unique Groups ≥ 200bp CSV File for BLSF dataset - with genome as index column
UG = pd.read_csv("/path/to/BT_noCF.uniquegroups.gte200.csv", index_col="genome")

# Import in labels.
# Genome name as index column
# Define what third each LD50 value is in
vir = pd.read_csv("/path/to/LD50_estimates_training.csv", index_col=0)
vir["quant"] = pd.qcut(vir.rounded, 3, labels=["high", "moderate", "low"])

# Define rank - high = 1, low = 0
# Filter out moderate isolates using quantile column and then remove quantile column
# Set all column datatypes to float for consistency
UG["quant"] = vir["quant"]
UG_filtered = UG[UG["quant"] != 'moderate'].copy()
UG_filtered.loc[UG["quant"] == "high", "rank"] = 1
UG_filtered.loc[UG["quant"] == "low", "rank"] = 0
UG_filtered = UG_filtered.drop("quant", axis=1)
UG_filtered = UG_filtered.astype("float64")

# Define X as all features and y as all labels
X = UG_filtered.drop("rank", axis=1)
y = UG_filtered["rank"]
print(X.head(), flush=True)
print(y, flush=True)

# Set seed for repeatable results (previously chosen based on call of np.random.randint(low=1, high=1e9))
s = 941335735

# Define how I want to do the crossvalidation - stratified and shuffled
stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)

# Define the parameters I want to search through
param_grid = {'max_features': ['sqrt', 'log2'],
              'min_samples_split': [2, 4],
              'min_samples_leaf': [1, 2, 4, 6],
              'criterion': ['gini', 'entropy'],
              'max_depth': [None, 10, 20, 30]
             }

# Test model performance with nested cross-validation
print("Estimating model performance with nested cross-validation", flush=True)

# Instantiate the estimator/grid search
RF = RandomForestClassifier(n_estimators=10000, n_jobs=1, random_state=s, oob_score=True)
RF_GridSearch = GridSearchCV(RF, param_grid, cv=stratkf, iid=False, n_jobs=-1)

# Define what scores I want - need to make custom score for specificity because not built in
scorer = {
    "accuracy": "accuracy",
    "sensitivity": "recall",
    "specificity": metrics.make_scorer(metrics.recall_score, greater_is_better=True, pos_label=0),
    "PPV": "precision",
    "AUC": "roc_auc",
    "f1": "f1"
}
# Determing CV_scores
CV_scores = cross_validate(RF_GridSearch, X, y, scoring=scorer, cv=stratkf)
print("Cross validation results:", flush=True)
print(CV_scores, flush=True)

#Report test results
CV_acc = CV_scores["test_accuracy"].mean()
print("Cross validation accuracy is " + str(CV_acc), flush=True)
CV_sensi = CV_scores["test_sensitivity"].mean()
print("Cross validation sensitivity is " + str(CV_sensi), flush=True)
CV_speci = CV_scores["test_specificity"].mean()
print("Cross validation specificity is " + str(CV_speci), flush=True)
CV_PPV = CV_scores["test_PPV"].mean()
print("Cross validation PPV is " + str(CV_PPV), flush=True)
CV_AUC = CV_scores["test_AUC"].mean()
print("Cross validation AUC is " + str(CV_AUC), flush=True)
CV_f1 = CV_scores["test_f1"].mean()
print("Cross validation f1 score is " + str(CV_f1), flush=True)

# Convert nested CV results to dataframe and write to file
CV_df = pd.DataFrame.from_dict(CV_scores)
CV_df.to_csv('BLSF_AGEs_ModerateIsolatesFiltered_RF_GSCV_NestedCVResults.csv', index=False)

# Build final model with GSCV using all training data
print("Building final model for export using GSCV with all training data", flush=True)

# Instantiate the grid estimator/grid search
RF = RandomForestClassifier(n_estimators=10000, n_jobs=1, random_state=s, oob_score=True)
RF_GridSearch_final = GridSearchCV(RF, param_grid, cv=stratkf, iid=False, n_jobs=-1)

# Fit the data to the model
RF_GridSearch_final.fit(X, y)

#Print model info
print("Final model RF grid search results:", flush=True)
print(RF_GridSearch_final, flush=True)

#Print internal best score
#Note: this should not be considered a representation of how the model performs on new data
print("Best internal accuracy: " + str(RF_GridSearch_final.best_score_), flush=True)

#Print best parameters
print("Best parameters:", flush=True)
print(RF_GridSearch_final.best_params_, flush=True)

#Print complete CV results
print("Complete CV results:", flush=True)
print(RF_GridSearch_final.cv_results_, flush=True)

#Extract out best Estimator
RF_final = RF_GridSearch_final.best_estimator_
print("Extracted model from GSCV", flush=True)
print(RF_final)
print("OOB score is " + str(RF_final.oob_score_))

#Pickle GCSV, best estimator, X, and y
with open('BLSF_AGEs_ModerateIsolatesFiltered_RF_GSCV_GridSearch.pkl', 'wb') as f:
    pickle.dump(RF_GridSearch_final, f)
with open('BLSF_AGEs_ModerateIsolatesFiltered_RF_GSCV_Model.pkl', 'wb') as f:
    pickle.dump(RF_final, f)
with open('BLSF_AGEs_ModerateIsolatesFiltered_RF_GSCV_Features.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('BLSF_AGEs_ModerateIsolatesFiltered_RF_GSCV_labels.pkl', 'wb') as f:
    pickle.dump(y, f)

# Plot learning curves - looking at accuracy
print("Plot learning curve", flush=True)

# Instantiate the estimator/grid search
RF = RandomForestClassifier(n_estimators=10000, n_jobs=1, random_state=s, oob_score=True)
RF_GridSearch = GridSearchCV(RF, param_grid, cv=stratkf, iid=False, n_jobs=-1)

file_lc = "BLSF_AGEs_ModerateIsolatesFiltered_RF_LearningCurve_95CI.pdf"
plot_learning_curves_function_95CI.plot_learning_curve(
    estimator=RF_GridSearch, X=X, y=y, scoring="accuracy", cv=stratkf, n_jobs=1,
    rs=s, ylab="Accuracy", ylim=(0.4,1.01))
plt.savefig(file_lc)
