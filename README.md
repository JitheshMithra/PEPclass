# Peptide Classification Project

Build a multi-label peptide activity classifier.

Each peptide can belong to more than one activity class. The data in this repository is the public training set only; final evaluation uses a held-out peptide set that is not included here.

## Data

- `data/labels.sqlite` contains one table: `peptides`
- `data/pdb/` contains one PDB file per peptide ID
- PDB filenames match the `ID` column, for example `satpdb10001.pdb`

The `peptides` table has these columns:

- `ID`
- `sequence`
- `length`
- `number_of_classes`
- one binary `0/1` column for each activity class

Activity classes:

- `antibacterial`
- `anticancer`
- `antifungal`
- `antihypertensive`
- `antimicrobial`
- `antiparasitic`
- `antiviral`
- `cell_cell_communication`
- `drug_delivery_vehicle`
- `toxic`

## Challenge

Train a model that predicts all applicable activity classes for each peptide.

Two input settings are valid:

- sequence only
- sequence plus structure

There is no required architecture. Classical features, embeddings, sequence models, structure-aware methods, and hybrid systems are all valid if the result is reproducible.

## Evaluation

Evaluation is multi-label classification on a held-out set with the same columns as `data/labels.sqlite`.

Submissions should produce one score or probability per class for each peptide ID. The input setting must be declared as either sequence only or sequence plus structure.

Primary metric:

- macro F1 across classes

Secondary metrics:

- micro F1
- per-class F1
- label-set exact match

Predicted probabilities are preferred so decision thresholds can be applied consistently across submissions.
