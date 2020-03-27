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

# Plot learning curves - looking at accuracy
print("Plot learning curve", flush=True)
RF_GSCV = GridSearchCV(RF, param_grid, cv=stratkf, iid=False, n_jobs=-1)
file_CG = "BLSF_8mers_RF_LearningCurve_95CI.pdf"
plot_learning_curves_function_95CI.plot_learning_curve(
    estimator=RF_GSCV, X=X, y=y, scoring="accuracy", cv=stratkf, n_jobs=1,
    rs=s, ylab="Accuracy", ylim=(0.4,1.01))
plt.savefig(file_CG)
print("", flush=True)
