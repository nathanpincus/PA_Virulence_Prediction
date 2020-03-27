# # Using RF to predict virulence (high/low) based on Core Genome - using one-hot encoding of features

# ### Load in modules needed for data inport/proccessing, analysis, and plotting

#For data inport/proccessing
import numpy as np
import pandas as pd
import pickle

#For machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

#For plotting
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# ### Import Core Genome SNVs and virulence labels

# Read in fasta file
fasta = open("/path/to/BLSF_vs_PAO1_95.replaced.filtered.fasta").readlines()

# Extract out all genomes in alignment
genomes = []
for line in fasta:
    if line.startswith(">"):
        genomes.append(line.lstrip(">").rstrip("\n"))

# Add the sequence for each genome to a dictionary
alignment = {}
for line in fasta:
    for g in genomes:
        if line.rstrip("\n") == (">" + g):
            seqindex = fasta.index(line) + 1
            seq = fasta[seqindex].rstrip("\n")
            alignment[g] = [base for base in seq]

# Convert alignment dictionary to a pandas dataframe
alignDF = pd.DataFrame.from_dict(alignment, orient='index')
alignDF.index.name = "genome"
# Convert column names to strings by adding "x" to start avoid problems later
alignDF = alignDF.add_prefix("x")

# Import in labels.
# Genome name as index column
vir = pd.read_csv("/path/to/LD50_estimates_training.csv", index_col=0)
vir.loc[vir.rounded < vir.rounded.median(), "rank"] = 1
vir.loc[vir.rounded >= vir.rounded.median(), "rank"] = 0

# Set seed for repeatable results (based on previous call of np.random.randint(low=1, high=1e9))
s = 144005885
print("Seed used in this analysis is " + str(s), flush=True)

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
alignDF["rank"] = vir["rank"]
# Define X as all features and y as all labels
X = alignDF.drop("rank", axis=1)
y = alignDF["rank"]
print("Features:", flush=True)
print(X.head(), flush=True)
print("Labels:", flush=True)
print(y, flush=True)


# Create pipeline to do both one-hot encoding and then random forest
pipe_CG = Pipeline(steps=[
    ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore")),
    ("rf", RandomForestClassifier(n_estimators=10000, n_jobs=1, random_state=s, oob_score=True))
])

# Create GSCV estimator
param_grid = {'rf__max_features': ['sqrt', 'log2'],
              'rf__min_samples_split': [2, 4],
              'rf__min_samples_leaf': [1, 2, 4, 6],
              'rf__criterion': ['gini', 'entropy'],
              'rf__max_depth': [None, 10, 20, 30]}
GSCV_CG = GridSearchCV(pipe_CG, param_grid, cv=stratkf, iid=False, n_jobs=-1)

# Fit estimator to all data and show best params
print("Fit estimator to complete dataset and show best parameters")
GSCV_CG.fit(X,y)
print(GSCV_CG, flush=True)

#Print internal best score
#Note: this should not be considered a representation of how the model performs on new data
print("Best internal score: " + str(GSCV_CG.best_score_), flush=True)

#Print best parameters and complete CV results
print("Best parameters:", flush=True)
print(GSCV_CG.best_params_, flush=True)

print("Complete CV results:", flush=True)
print(GSCV_CG.cv_results_, flush=True)

# Pickle final model
with open('BLSF_CG_OHE_RF_GridSearch.pkl', 'wb') as f:
    pickle.dump(GSCV_CG, f)

# Run nested CV
GSCV_CG = GridSearchCV(pipe_CG, param_grid, cv=stratkf, iid=False, n_jobs=-1)
CV_scores = cross_validate(GSCV_CG, X, y, scoring=scorer, cv=stratkf)
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

# Save nested CV results to file
CV_df = pd.DataFrame.from_dict(CV_scores)
CV_df.to_csv('BLSF_CG_OHE_RF_NestedCVResults.csv', index=False)
