# # Looking at performance of BLSF_noCF final GSCV-tuned RF model on the Training Data

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

# Extract OOB accuracy from the final RF model
OOB_Acc = RF.oob_score_

# ### Predict virulence of training isolates using GSCV RF model
# ### Compare predictions to true values
ypred = GSCV.predict(X)
y_pred_prob = GSCV.predict_proba(X)[:, 1]
print("Predicted virulence labels:")
print(ypred)

# Determining perforance
Acc = metrics.accuracy_score(y, ypred)
Sen = metrics.recall_score(y, ypred)
Sp = metrics.recall_score(y, ypred, pos_label=0)
PPV = metrics.precision_score(y, ypred)
AUC = metrics.roc_auc_score(y, y_pred_prob)
F1 = metrics.f1_score(y, ypred)

# Save test set results to a file
Scores = pd.DataFrame(columns=["OOB Accuracy", "Accuracy", "Sensitivity", "Specificity", "PPV", "AUC", "F1"])
Scores.loc[0] = [OOB_Acc,Acc,Sen,Sp,PPV,AUC,F1]
print("Model performance on test set:")
print(Scores, flush=True)
Scores.to_csv("BLSF_RF_TrainingPerformance.csv", index=False)

# Plot ROC curve
print("Plot AUC curve:")
file_auc = "BLSF_RF_TrainingPerformance_AUC.pdf"
fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_prob)
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
