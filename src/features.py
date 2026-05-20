from Bio.PDB import PDBParser
import numpy as np
import warnings
warnings.filterwarnings("ignore")

aminoacids = "ACDEFGHIKLMNPQRSTVWY"

def getsequencefeatures(sequence):
    #this will extract the feautures from the peptide sequence
    sequence= sequence.upper().strip()
    length= len(sequence)
    if length==0:
        return [0]*23
    #percentage count of each amino acid
    composition={}
    for i in aminoacids:
        composition[i]= sequence.count(i)/length
        
    #this is for the physiochemical properties, will be explained in the presentation icl
    positive= sum(sequence.count(i) for i in 'KRH')
    negative=sum(sequence.count(i) for i in 'DE')
    charge= (positive - negative)/length
    hydrophobic= sum(sequence.count(i) for i in 'VILMFYW')
    hydrophobicity=hydrophobic/length
    
    #features vector build lol
    features=list(composition.values()) + [length, charge, hydrophobicity]
    return features

def getstructurefeatures(pdbfile):
    #it does exactly what the name says it does
    parser=PDBParser()
    structure=parser.get_structure("peptide", pdbfile)
    #gets CA atoms, check out the cool variable name (you cant tell me it isnt funny)
    catoms=[]
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    catoms.append(residue["CA"].get_vector())
    if len(catoms)<2:
        return [0]*3
    
    endtoend=(catoms[-1]-catoms[0]).norm()
    
    #computes the radius of gyration which is how compact the peptide is
    coord=np.array([[v[0], v[1], v[2]] for v in catoms])
    center=coord.mean(axis=0)
    rog=np.sqrt(((coord - center)**2).sum(axis=1).mean())
    
    #avg b factor, means flexibility
    bfactor=[]
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    bfactor.append(residue["CA"].get_bfactor())
    
    if len(bfactor)>0:
        avgbfactor= np.mean(bfactor)
    else:
        avgbfactor=0
    
    return [endtoend, rog, avgbfactor]