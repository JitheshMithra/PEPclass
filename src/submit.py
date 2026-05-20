import os
import joblib
import pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from features import getsequencefeatures, getstructurefeatures
import warnings
warnings.filterwarnings('ignore')

classes=['antibacterial','anticancer','antifungal','antihypertensive', 'antimicrobial','antiparasitic','antiviral','cell_cell_communication', 'drug_delivery_vehicle','toxic']

def getsequencefrompdb(pdbfile):
    parser=PDBParser()
    structure=parser.get_structure('peptide', pdbfile)
    ppb=PPBuilder()
    sequence=''
    for pp in ppb.build_peptides(structure):
        sequence+=str(pp.get_sequence())
    return sequence

testpdbfolder='../data/test/pdb'
pdbfiles=sorted(os.listdir(testpdbfolder))

# sequence only 
seqmodel=joblib.load('../models/sequencemodel.pkl')
seqrows=[]
for pdbfile in pdbfiles:
    peptideid=pdbfile.replace('.pdb','')
    fullpath=os.path.join(testpdbfolder, pdbfile)
    sequence=getsequencefrompdb(fullpath)
    features=[getsequencefeatures(sequence)]
    preds=seqmodel.predict(features)[0]
    row={'ID': peptideid}
    for i, cls in enumerate(classes):
        row[cls]=int(preds[i])
    seqrows.append(row)

seqdf=pd.DataFrame(seqrows)
seqdf.to_csv('../results/submission_sequence.csv', index=False)
print("Sequence only submission saved")

# sequence & structure 
structmodel=joblib.load('../models/structuremodel.pkl')
structrows=[]
for pdbfile in pdbfiles:
    peptideid=pdbfile.replace('.pdb','')
    fullpath=os.path.join(testpdbfolder, pdbfile)
    sequence=getsequencefrompdb(fullpath)
    seqfeatures=getsequencefeatures(sequence)
    structfeatures=getstructurefeatures(fullpath)
    features=[seqfeatures+structfeatures]
    preds=structmodel.predict(features)[0]
    row={'ID': peptideid}
    for i, cls in enumerate(classes):
        row[cls]=int(preds[i])
    structrows.append(row)

structdf=pd.DataFrame(structrows)
structdf.to_csv('../results/submission_structure.csv', index=False)
print("Structure and sequence submission saved")