# PEPclass - Peptide Classification Project

Reproducible and Open-source machine learning simulation framework that predicts peptide biological activity classes from sequence and structural features.

PEPclass predicts whether a peptide belongs to one or more of 10 biological activity classes using amino acid composition, physiochemical properties, and 3D structural features extracted from PDB files. See data/ for specific files.

## Methodology
Feature extraction -> Random Forest classification (100 Trees) -> Multilabel prediction

Sequence: Amino Acid Composition, length, charge, hydrophobicity

Structure: End-to-end distance, radius of gyration, average B-factor

## Installation 
```bash
git clone https://github.com/JitheshMithra/PepClass
cd Pepclass
pip install -r requirements.txt
mkdir models
mkdir results
cd src
python train.py
```
## Prediction
Sequence only:
```bash
python prediction.py --sequence KWKKLLKKPPPLLKKLLKKL
```
Sequence and Structure:
```bash
python prediction.py --sequence KWKKLLKKPPPLLKKLLKKL --pdb ../data/pdb/satpdb10001.pdb
```

CSV file export:
```bash
python submit.py
```
