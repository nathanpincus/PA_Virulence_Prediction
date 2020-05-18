# # Predicting virulence (High/Low) of 25 PASP isolates using BLSF_noCF final GSCV-tuned RF model - trained on 8-mers feature set

# ### Load in modules needed for data inport/proccessing, analysis, and plotting
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

#For plotting
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# ### Load in BLSF_noCF model
# Load in GSCV object
with open("/path/to/BLSF_RF_GSCV_8mers_GridSearch.pkl", "rb") as f:
    GSCV = pickle.load(f)
print("GSCV Object:")
print(GSCV)

# ### Load in training virulence values - so that I can determine cutoff for PASP strains
# Import in labels.
# Genome name as index column
vir = pd.read_csv("/path/to/LD50_estimates_training.csv", index_col=0)
vir.loc[vir.rounded < vir.rounded.median(), "rank"] = 1
vir.loc[vir.rounded >= vir.rounded.median(), "rank"] = 0

threshold = vir.rounded.median()
print("Threshold for high virulence is: {0}".format(str(threshold)))

# ### Load in PASP features (8mers)
PASP = pd.read_csv("/path/to/PASP_8mers.csv", index_col=0)
print("PASP Test Set Features:")
PASP.head()

# ### Add in true labels for PASP isolates based on LD50 cuttoff for high/low and split in to Xtest and ytest
# Import in test labels.
# Genome name as index column
PASP_vir = pd.read_csv("/path/to/LD50_estimates_testing.csv", index_col=0)
PASP_vir.loc[PASP_vir.rounded < threshold, "rank"] = 1
PASP_vir.loc[PASP_vir.rounded >= threshold, "rank"] = 0
print("PASP Test Set True Labels:")
print(PASP_vir)

# Add labels to PASP object
PASP["rank"] = PASP_vir["rank"]
print("combined object")
print(PASP.head())

# Define test set features and labels
Xtest = PASP.drop("rank", axis=1)
print("Test features:")
print(Xtest.head())

ytest = PASP["rank"]
print("Test true labels:")
print(ytest)

# Deterimine the prevelance of high virulence in the test set (based on true labels)
Hprev = ytest.value_counts()[1]/(ytest.value_counts()[1] + ytest.value_counts()[0])

# ### Predict virulence of PASP isolates using GSCV RF model
# ### Compare predictions to true values
# Predict labels for test set
ypred = GSCV.predict(Xtest)
y_pred_prob = GSCV.predict_proba(Xtest)[:, 1]
print("Predicted virulence labels:")
print(ypred)

# Determining perforance
Acc = metrics.accuracy_score(ytest, ypred)
Sen = metrics.recall_score(ytest, ypred)
Sp = metrics.recall_score(ytest, ypred, pos_label=0)
PPV = metrics.precision_score(ytest, ypred)
AUC = metrics.roc_auc_score(ytest, y_pred_prob)
F1 = metrics.f1_score(ytest, ypred)

# Save test set results to a file
Scores = pd.DataFrame(columns=["HighVir", "Accuracy", "Sensitivity", "Specificity", "PPV", "AUC", "F1"])
Scores.loc[0] = [Hprev,Acc,Sen,Sp,PPV,AUC,F1]
print("Model performance on test set:")
print(Scores, flush=True)
Scores.to_csv("PASP_VirulencePredictions_BLSF_RF_8mers.csv", index=False)

# Plot ROC curve
print("Plot AUC curve:")
file_auc = "PASP_VirulencePredictions_BLSF_RF_8mers_AUC.pdf"
fpr, tpr, thresholds = metrics.roc_curve(ytest, y_pred_prob)
print("FPR:")
print(fpr)
print("TPR:")
print(tpr)
print("Thresholds:")
print(thresholds)
plt.figure(1)
plt.plot(fpr, tpr, linewidth=2.8, color="red")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.savefig(file_auc)

# Export csv of true vs predicted virulence values
PASP["pred"] = ypred
Comp = PASP[["rank","pred"]]
Comp.to_csv("PASP_virrank_vs_prediction_8mers.csv")
