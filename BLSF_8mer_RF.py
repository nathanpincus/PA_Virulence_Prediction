# # Using RF to predict virulence (high/low) based on 8-mer counts

# Load in modules needed for data inport/proccessing, analysis, and plotting
#For data inport/proccessing
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


# Read in k-mer csv file
kmers = pd.read_csv("/path/to/BLSF_noCF_8mers.csv", index_col=0)

# Import in labels.
# Genome name as index column
vir = pd.read_csv("/path/to/LD50_estimates_training.csv", index_col=0)
vir.loc[vir.rounded < vir.rounded.median(), "rank"] = 1
vir.loc[vir.rounded >= vir.rounded.median(), "rank"] = 0

# Set seed for repeatable results (previously chosen based on call of np.random.randint(low=1, high=1e9))
s = 326269767

# Define how I want to do crossvalidaiton - stratified and shuffled
stratkf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s)
print("Cross validation strategy:", flush=True)
print(stratkf, flush=True)

# Definine scorer for nested cross-validation
scorer = {
    "accuracy": "accuracy",
    "sensitivity": "recall",
    "specificity": metrics.make_scorer(metrics.recall_score, greater_is_better=True, pos_label=0),
    "PPV": "precision",
    "AUC": "roc_auc",
    "f1": "f1"
}
print("Scoring strategy for CV:", flush=True)
print(scorer, flush=True)
print("", flush=True)

# Define X and y
kmers["rank"] = vir["rank"]
# Define X as all features and y as all labels
X = kmers.drop("rank", axis=1)
y = kmers["rank"]
print("Features:", flush=True)
print(X.head(), flush=True)
print("Labels:", flush=True)
print(y, flush=True)


# Instantiate estimator
RF = RandomForestClassifier(n_estimators=10000, n_jobs=1, random_state=s, oob_score=True)

# Create GSCV object
param_grid = {'max_features': ['sqrt', 'log2'],
              'min_samples_split': [2, 4],
              'min_samples_leaf': [1, 2, 4, 6],
              'criterion': ['gini', 'entropy'],
              'max_depth': [None, 10, 20, 30]}
RF_GSCV = GridSearchCV(RF, param_grid, cv=stratkf, iid=False, n_jobs=-1)

# Fit estimator to all data and show best params
print("Fit estimator to complete dataset and show best parameters", flush=True)
RF_GSCV.fit(X,y)
print(RF_GSCV, flush=True)

#Print internal best score
#Note: this should not be considered a representation of how the model performs on new data
print("Best internal score: " + str(RF_GSCV.best_score_), flush=True)

#Print best parameters
print("Best parameters:", flush=True)
print(RF_GSCV.best_params_, flush=True)

print("Complete CV results:", flush=True)
print(RF_GSCV.cv_results_, flush=True)

# Pickle final model
with open('BLSF_RF_GSCV_8mers_GridSearch.pkl', 'wb') as f:
    pickle.dump(RF_GSCV, f)

# Run nested CV
RF_GSCV = GridSearchCV(RF, param_grid, cv=stratkf, iid=False, n_jobs=-1)
CV_scores = cross_validate(RF_GSCV, X, y, scoring=scorer, cv=stratkf)
print("Estimate generalization performance through nested CV:", flush=True)
print(CV_scores, flush=True)

# nested CV results
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
print("")

# Save nested CV results to file
CV_df = pd.DataFrame.from_dict(CV_scores)
CV_df.to_csv('BLSF_RF_GSCV_8mers_NestedCVResults.csv', index=False)
