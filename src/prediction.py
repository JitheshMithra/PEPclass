import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from features import getsequencefeatures
from features import getstructurefeatures

#define classes 
classes= ['antibacterial', 'anticancer', 'antifungal', 'antihypertensive', 'antiviral', 'antiparasitic', 'antimicrobial', 'drug_delivery_vehicle', 'toxic', 'cell_cell_communication']
def predictsequence(sequence, model):
    features=[getsequencefeatures(sequence)]
    probabilities=model.predict_proba(features)
    results={}
    for i, class1 in enumerate(classes):
        results[class1]= probabilities[i][0][1] #this is the probability of positive class (yeahhh peep the jargon)
    return results

def predictstructure(sequence, pdbfile, model):
    seqfeatures=getsequencefeatures(sequence)
    structfeatures=getstructurefeatures(pdbfile)
    features= [seqfeatures+ structfeatures]
    probabilities=model.predict_proba(features)
    results={}
    for i, class1 in enumerate(classes):
        results[class1]= probabilities[i][0][1] #once again probabilities of the positive class
    return results

#plotting results
def plotresults(results, sequence):
    classes=list(results.keys())
    probs=list(results.values())
    plt.figure(figsize=(10,6))
    bars=plt.barh(classes,probs,color="steelblue")
    plt.xlabel("Probability")
    plt.title(f"PEPclass Predictions\n{sequence}")
    plt.xlim(0,1)
    
    for bar, prob in zip(bars, probs):
        plt.text(prob+0.01,bar.get_y()+bar.get_height()/2,f"{prob:.2f}", va="center")
        
    plt.tight_layout()
    plt.savefig("../results/prediction-plot.png")
    print("Plot saved to results/")
    plt.show()
    
#this is how inputs are done in cl and the main engine/function
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--pdb", type=str,default=None)
    args=parser.parse_args()
    if args.pdb:
        model=joblib.load("../models/structuremodel.pkl")
        results=predictstructure(args.sequence, args.pdb, model)
        print(f"\nPredictions (Sequence and Structure): {args.sequence}")
    else:
        model =joblib.load("../models/sequencemodel.pkl")
        results=predictsequence(args.sequence, model)
        print(f"\nPredictions (only sequence): {args.sequence}")
        
    for class1, prob in results.items():
        print(f"{class1}: {prob:.4f}")
    plotresults(results, args.sequence)

if __name__=="__main__":
    main()
    
