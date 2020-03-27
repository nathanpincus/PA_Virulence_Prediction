# # Using Random Forest to predict virulence based on Unique AGEs in the BLSF_noCF dataset - using AGE subsets as features

# Load in modules needed for data inport/proccessing, analysis, and plotting
#For data inport/proccessing/export
import numpy as np
import pandas as pd
import pickle

#For machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# Import Unique Groups of SE for BLSF_noCF dataset and label with virulence data

# Read in Unique Groups ≥ 200bp CSV File for BLSF dataset - with genome as index column
UG = pd.read_csv("/path/to/BT_noCF.uniquegroups.gte200.csv", index_col="genome")

# Import in labels.
# Genome name as index column
vir = pd.read_csv("/path/to/LD50_estimates_training.csv", index_col=0)
vir.loc[vir.rounded < vir.rounded.median(), "rank"] = 1
vir.loc[vir.rounded >= vir.rounded.median(), "rank"] = 0

# Combine features and labels for easy splitting into X and y
# Set all column datatypes to float for consistency
UG["rank"] = vir["rank"]
UG = UG.astype("float64")

# Define X as all features and y as all labels
X = UG.drop("rank", axis=1)
y = UG["rank"]
print(X.head())
print(y)


# ### Test RF performance with nested cross-validation for a given input set
def NestedCV(features, labels, subset_name, s):
    # Instantiate estimator
    RF = RandomForestClassifier(n_estimators=10000, random_state=s)
    # Define how I want to do the crossvalidation - stratified and shuffled
    stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)

    # Define the parameters I want to search through
    param_grid = {'max_features': ['sqrt', 'log2'],
                  'min_samples_split': [2, 4],
                  'min_samples_leaf': [1, 2, 4, 6],
                  'criterion': ['gini', 'entropy'],
                  'max_depth': [None, 10, 20, 30]
                  }

    # Instantiate the grid estimator/grid search
    # Estimator - RF
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
    CV_scores = cross_validate(RF_GridSearch, features, labels, scoring=scorer, cv=stratkf)
    # Convert nested CV results to dataframe and write to file
    CV_df = pd.DataFrame.from_dict(CV_scores)
    # Export to CSV
    CV_df.to_csv('BLSF_RF_NestedCVResults_subset' + subset_name + '.csv', index=False)


# Break input features into n subsets
# Loop through the subsets - save feature set to a file, then run through ML pipeline and save nested CV results to a file using the above function
def NestedCV_Feature_Subsets(n, X, y, s):
    #Extract feature names from X
    features = X.columns.values
    #Set seed for random permutation
    print("Seed is " + str(s), flush=True)
    np.random.seed(s)
    #Randomly permute features
    features_permuted = np.random.permutation(features)
    # Define breakpoints based on how many partitions you want
    breakpoints = [i for i in range(0, len(features), round(len(features) / n))]
    # In a loop - subset X into n partitions and run nested CV using each
    # Note that last partition needs to be treated differently to ensure all features accounted for
    for i in range(n):
        if i == n - 1:
            subset = str(n) + "of" + str(n)
            feature_subset = features_permuted[breakpoints[i]:len(features)]
        else:
            subset = str(i + 1) + "of" + str(n)
            feature_subset = features_permuted[breakpoints[i]:breakpoints[i + 1]]
        print("Running subset " + subset, flush=True)
        X_subset = X.copy()[feature_subset]
        X_subset.to_csv('BLSF_features_subset' + subset + '.csv', index=False)
        NestedCV(features=X_subset, labels=y, subset_name=subset, s=s)


# set seed
seed = 611368986
# Run splitting into 2 subsets
NestedCV_Feature_Subsets(n=2, X=X, y=y, s=seed)

# set seed
seed = 204139744
# Run splitting into 4 subsets
NestedCV_Feature_Subsets(n=4, X=X, y=y, s=seed)

# set seed
seed = 778863368
# Run splitting into 10 subsets
NestedCV_Feature_Subsets(n=10, X=X, y=y, s=seed)
