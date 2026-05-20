#idk if i have to explain all the imports and my reason behind them but that will be worried about later
import sqlite3
import os
import pandas as pd
from features import getstructurefeatures
from features import getsequencefeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import joblib
import matplotlib.pyplot as plt
import numpy as np

#load data
connect=sqlite3.connect('../data/labels.sqlite')
dataframe=pd.read_sql_query("Select * FROM peptides", connect)
connect.close()
#defining the classes
classes= ['antibacterial', 'anticancer', 'antifungal', 'antihypertensive', 'antiviral', 'antiparasitic', 'antimicrobial', 'drug_delivery_vehicle', 'toxic', 'cell_cell_communication']
#extract features
def getallfeatures(row, pdbfolder):
    seqfeatures=getsequencefeatures(row['sequence'])
    pdbfile=os.path.join(pdbfolder, row['ID']+".pdb")
    if os.path.exists(pdbfile):
        structfeatures= getstructurefeatures(pdbfile)
    else:
        structfeatures=[0]*3
    return seqfeatures+ structfeatures
pdbfolder="../data/pdb"
X=dataframe.apply(lambda row: getallfeatures(row, pdbfolder), axis=1).tolist()
Y=dataframe[classes].values

#split
Xtrain, Xtest, Ytrain, Ytest=train_test_split(X, Y, test_size=0.2, random_state=42)
#train
model=MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
model.fit(Xtrain, Ytrain)
#evaluate the model and get the f1 score and report 
yprediction = model.predict(Xtest)

#threshold tradeoff plot
thresholds= np.arange(0.1, 0.9, 0.05)
f1scores= []
yprobs = []
for i in model.estimators_:
    classprobs=i.predict_proba(Xtest)[:,1]
    yprobs.append(classprobs)
yprobs = np.array(yprobs).T

for i in thresholds:
    ypredthresh= (yprobs>=i).astype(int)
    f1= f1_score(Ytest, ypredthresh, average='macro')
    f1scores.append(f1)
    
optimalthresh = thresholds[np.argmax(f1scores)]
print(f"Optimal Threshold: {optimalthresh:.2f} with F1: {max(f1scores):.4f}")
plt.figure(figsize=(8,5))
plt.plot(thresholds, f1scores, marker='o')
plt.title('Macro F1 vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Macro F1 Score')
plt.axvline(x=optimalthresh, color='red', linestyle='--', label=f'Optimal: {optimalthresh:.2f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../results/threshold_plot.png')
print("Threshold plot saved")

#cross fold validation
from sklearn.model_selection import cross_val_score
cvscores= cross_val_score(model, X, Y, cv=5, scoring='f1_macro')
print(f"Cross-val Macro F1 Scores: {cvscores.mean():.4f} (+/- {cvscores.std():.4f})")

#your probably wondering why im using f1 locally, and im telling you that its to be cheeky, and i know where i stand without having to do submissions on kaggle over and over and over again; im sure you get the point
macrof1=f1_score(Ytest, yprediction, average='macro')
print(f"Macro F1 Score: {macrof1:.4f}")
print(classification_report(Ytest, yprediction, target_names=classes))

#saving and a cool message that says that its saved
joblib.dump(model, '../models/structuremodel.pkl')
print("structure and sequence model saved")

# train and save sequence only model
Xseq = dataframe['sequence'].apply(getsequencefeatures).tolist()
Xseqtrain, Xseqtest, Yseqtrain, Yseqtest = train_test_split(Xseq, Y, test_size=0.2, random_state=42)
seqmodel = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
seqmodel.fit(Xseqtrain, Yseqtrain)
joblib.dump(seqmodel, '../models/sequencemodel.pkl')
print("sequence model saved")