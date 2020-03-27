# Using RF to predict virulence (high/low) based on 10-mers - manual nested CV

# ### Load in modules needed for data inport/proccessing, analysis, and plotting
import argparse

#For data inport/proccessing
import numpy as np
import pandas as pd
import pickle

#For machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

# ### Load in command line arguments
parser = argparse.ArgumentParser(description="Manually determine cross-valdidation performance for the BLSF_noCF 10-mer dataset using pre-defined folds")
parser.add_argument("-s", "--split", required=True, help="current split")
args = parser.parse_args()

print("Current split is {0}".format(str(args.split)))

# ### Load in datasets
print("Training set:", flush=True)
with open("BLSF_RF_GSCV_10mers_Xtrain_split{0}.pkl".format(str(args.split)), "rb") as f:
    Xtrain = pickle.load(f)
print(Xtrain.head(), flush=True)
with open("BLSF_RF_GSCV_10mers_ytrain_split{0}.pkl".format(str(args.split)), "rb") as f:
    ytrain = pickle.load(f)
print(ytrain.head(), flush=True)

print("Test set:", flush=True)
with open("BLSF_RF_GSCV_10mers_Xtest_split{0}.pkl".format(str(args.split)), "rb") as f:
    Xtest = pickle.load(f)
print(Xtrain.head(), flush=True)
with open("BLSF_RF_GSCV_10mers_ytest_split{0}.pkl".format(str(args.split)), "rb") as f:
    ytest = pickle.load(f)
print(ytest.head(), flush=True)

# Set seed for repeatable results
s = 828192508
print("Seed used in this analysis is " + str(s), flush=True)

# Define how I want to do crossvalidaiton - stratified and shuffled
stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)
print("Cross validation strategy:", flush=True)
print(stratkf, flush=True)

# Instantiate estimator
RF = RandomForestClassifier(n_estimators=10000, n_jobs=1, random_state=s, oob_score=True)

# Create GSCV object
param_grid = {'max_features': ['sqrt', 'log2'],
              'min_samples_split': [2, 4],
              'min_samples_leaf': [1, 2, 4, 6],
              'criterion': ['gini', 'entropy'],
              'max_depth': [None, 10, 20, 30]}
GSCV_kmer = GridSearchCV(RF, param_grid, cv=stratkf, iid=False, n_jobs=-1)

# Fit estimator to training data for this fold
print("Fit estimator to training dataset and show best parameters")
GSCV_kmer.fit(Xtrain,ytrain)
print(GSCV_kmer, flush=True)

#Print internal best score
#Note: this should not be considered a representation of how the model performs on new data
print("Best internal score: " + str(GSCV_kmer.best_score_), flush=True)

#Print best parameters
print("Best parameters:", flush=True)
print(GSCV_kmer.best_params_, flush=True)

print("Complete CV results:", flush=True)
print(GSCV_kmer.cv_results_, flush=True)

# Pickle model for fold
with open('BLSF_RF_GSCV_10mers_split{0}_GridSearch.pkl'.format(str(args.split)), 'wb') as f:
    pickle.dump(GSCV_kmer, f)

# Test against CV test set for this fold
print("Test performance against test set for this fold", flush=True)
# Predict values for the testing data
ypred = GSCV_kmer.predict(Xtest)
y_pred_prob = GSCV_kmer.predict_proba(Xtest)[:, 1]
# Determining perforance
Acc = metrics.accuracy_score(ytest, ypred)
Sen = metrics.recall_score(ytest, ypred)
Sp = metrics.recall_score(ytest, ypred, pos_label=0)
PPV = metrics.precision_score(ytest, ypred)
AUC = metrics.roc_auc_score(ytest, y_pred_prob)
F1 = metrics.f1_score(ytest, ypred)

# Save CV results for this fold to a file
Scores = pd.DataFrame(columns=["Accuracy", "Sensitivity", "Specificity", "PPV", "AUC", "F1"])
Scores.loc[0] = [Acc,Sen,Sp,PPV,AUC,F1]
print(Scores, flush=True)
Scores.to_csv("BLSF_RF_GSCV_10mers_split{0}_CVResults.csv".format(str(args.split)), index=False)
