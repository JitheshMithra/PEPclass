import argparse
import joblib
import pandas as pd
from features import getsequencefeatures

#define classes 
classes= ['antibacterial', 'anticancer', 'antifungal', 'antihypertensive', 'antiviral', 'antiparasitic', 'antimicrobial', 'drug_delivery_vehicle', 'toxic', 'cell_cell_communication']
def predictsequence(sequence, model):
    features=[getsequencefeatures(sequence)]
    probabilities=model.predict_proba(features)
    results={}
    for i, class1 in enumerate(classes):
        results[class1]= probabilities[i][0][1] #this is the probability of positive class (yeahhh peep the jargon)
    return results

#this is how inputs are done in cl and the main engine/function
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--pdb", type=str,default=None)
    args=parser.parse_args()
    
    model =joblib.load("../models/sequencemodel.pkl")
    results=predictsequence(args.sequence, model)
    
    print(f"\nPredictions: {args.sequence}")
    for class1, prob in results.items():
        print(f"{class1}: {prob:.4f}")

if __name__=="__main__":
    main()
    
