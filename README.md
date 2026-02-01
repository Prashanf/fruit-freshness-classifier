# Fruit Freshness Classification (Learning & R&D Project)

## Overview
This project explores fruit freshness classification using deep learning.
The primary objective was to understand how CNN architectures learn visual
patterns rather than optimizing only for accuracy.

## Dataset
Images are categorized into:
- freshapples
- freshbanana
- freshoranges
- rottenapples
- rottenbanana
- rottenoranges

Dataset is organized using ImageFolder-style directories.

## Models
- Normal CNN (baseline)
- TreeCNN (custom experimental architecture)

TreeCNN showed interesting generalization behavior despite lower validation accuracy.

## Key Learnings
- Architecture design influences decision consistency
- Accuracy alone is not sufficient to judge model intelligence
- Data augmentation significantly improves generalization

## Repository Structure
<<<<<<< HEAD
(brief structure)
=======
fruit-freshness-classifier/
├── notebooks/
│   └── fruit_classifier_rnd.ipynb
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── augmentations.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── models/
│   └── model1.pth
│
├── requirements.txt
├── README.md
└── .gitignore

>>>>>>> 07e9e83d39566d05cc1125c1c4b8c856c4c7b69b

## Future Work Scope
- Transfer learning
- Web deployment
- Dataset expansion
