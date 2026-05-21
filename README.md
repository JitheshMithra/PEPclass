# PEPclass - Peptide Classification Project

Reproducible and Open-source machine learning simulation framework that predicts peptide biological activity classes from sequence and structural features.

PEPclass predicts whether a peptide belongs to one or more of 10 biological activity classes using amino acid composition, physiochemical properties, and 3D structural features extracted from PDB files. See data/ for specific files.

## Methodology
Feature extraction -> Random Forest classification (80 Trees) -> Multilabel prediction

Sequence: Amino Acid Composition, length, charge, hydrophobicity

Structure: End-to-end distance, radius of gyration, average B-factor

## Installation and training
```bash
git clone https://github.com/JitheshMithra/PEPclass
cd PEPclass
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

## Results
- Kaggle Macro F1: 0.235 (sequence & structure mode)
- Local threshold optimized F1: 0.4 (tuned on same split)
- Cross validated Macro F1: 0.29
- Optimal threshold: 0.15 (determined via threshold tradeoff plot analysis)
- Best performing class: antimicrobial (F1 = 0.84)
- Weakest classes: antiparasitic, antihypertensive, cell_cell_communication (F1 = 0.00-0.09, class imbalance)

## Figures
- results/threshold_plot.png: Optimal threshold analysis
- results/f1_plot.png: Per-class F1 scores
- results/auc_plot.png: ROC curves per class
- results/oob_curve.png: OOB error vs number of trees

## Limitations
- B-Factor values in PDB files are 0.00
- Class imbalance impacts the rare class prediction
- Random forest is not optimal for multilabel classification, gradient boosting or neural network approach would perform better
- Sequence features show composition but not positional information
- Threshold optimized on validation split may not generalize to perfectly unseen data
