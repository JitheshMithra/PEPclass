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