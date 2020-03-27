# Splitting BLSF_noCF 10-mer dataset into folds for manual cross-validaiton

# Load in modules needed for data import/processing
#For data inport/proccessing
import numpy as np
import pandas as pd
import pickle

#For machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# Read in k-mer csv file
kmers = pd.read_csv("/path/to/BLSF_noCF_10mers.csv", index_col=0)

# Import in labels.
# Genome name as index column
vir = pd.read_csv("/path/to/LD50_estimates_training.csv", index_col=0)
vir.loc[vir.rounded < vir.rounded.median(), "rank"] = 1
vir.loc[vir.rounded >= vir.rounded.median(), "rank"] = 0

# Set seed for repeatable results
s = 828192508
print("Seed used in this analysis is " + str(s), flush=True)

# Define how I want to do crossvalidaiton - stratified and shuffled
stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)
print("Cross validation strategy:", flush=True)
print(stratkf, flush=True)

# Define X and y (full dataset)
kmers["rank"] = vir["rank"]
# Define X as all features and y as all labels
X = kmers.drop("rank", axis=1)
y = kmers["rank"]
print("Complete Dataset", flush=True)
print("Features:", flush=True)
print(X.head(), flush=True)
print("Labels:", flush=True)
print(y, flush=True)

# Split into folds using stratkf and save these folds to files
split = 0
for train_index, test_index in stratkf.split(X, y):
    print("Split {0}".format(str(split)), flush=True)
    Xtrain = X.iloc[train_index,]
    ytrain = y.iloc[train_index,]
    Xtest = X.iloc[test_index,]
    print("Test Features:", flush=True)
    print(Xtest.head(), flush=True)
    ytest = y.iloc[test_index,]
    print("Test Labels", flush=True)
    print(ytest.head(), flush=True)
    with open('BLSF_RF_GSCV_10mers_Xtrain_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(Xtrain, f)
    with open('BLSF_RF_GSCV_10mers_ytrain_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(ytrain, f)
    with open('BLSF_RF_GSCV_10mers_Xtest_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(Xtest, f)
    with open('BLSF_RF_GSCV_10mers_ytest_split{0}.pkl'.format(str(split)), 'wb') as f:
        pickle.dump(ytest, f)
    split = split + 1
