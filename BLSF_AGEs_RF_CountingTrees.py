# # Looking at final GSCV RF model - how often are features included any any given tree?

# ### Load in modules needed for data import/processing, analysis, and plotting
#For data inport/proccessing
import numpy as np
import pandas as pd
import pickle

#For machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


# ### Load in BLSF_noCF model and training data
# Load in GSCV object
with open("/path/to/BLSF_AGEs_RF_GSCV_GridSearch.pkl", "rb") as f:
    GSCV = pickle.load(f)
print("GSCV Object:")
print(GSCV)
#Load in RF - extracted from GSCV
with open("/path/to/BLSF_AGEs_RF_GSCV_Model.pkl", "rb") as f:
    RF = pickle.load(f)
print("RF object:")
print(RF)
print(RF.oob_score_)
#Load in training features
with open("/path/to/BLSF_AGEs_RF_GSCV_Features.pkl", "rb") as f:
    X = pickle.load(f)
print("Features:")
print(X.shape)
print(X.head())
#Load in training labels
with open("/path/to/BLSF_AGEs_RF_GSCV_labels.pkl", "rb") as f:
    y = pickle.load(f)
print("labels:")
print(y.shape)
print(y.head())

#Make dataframe with unique groups in order as features to hold count of how many trees each feature is in
UG_tree_counts = pd.DataFrame(0, index=["n_trees"], columns=X.columns)

# Count features included in each tree (-2 appears to be the marker for a leaf node with no feature)
# Add them to the DF created above
for t in RF.estimators_:
    features = t.tree_.feature
    for i in features:
        if i != -2:
            UG_tree_counts.iloc[0, i] = UG_tree_counts.iloc[0, i] + 1

# Print filled count dataframe
print(UG_tree_counts, flush=True)

# How many features are in >0 trees?
count = 0
for i in UG_tree_counts.T.n_trees:
    if i > 0:
        count = count + 1
print("Number of features in >0 trees: {0}".format(str(count)))

# Print some summary statistics
print("Mean number of trees per feature:", flush=True)
print(UG_tree_counts.T.mean(), flush=True)
print("Median number of trees per feature:", flush=True)
print(UG_tree_counts.T.median(), flush=True)
print("Max number of trees per feature", flush=True)
print(UG_tree_counts.T.max(), flush=True)

# Export count dataframe to file
UG_tree_counts.to_csv("BLSF_AGEs_RF_tree_counts.csv")
