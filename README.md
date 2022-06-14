# CNN/CLIP Shortcut Feature Reliance
Author: Anil Palepu

HST PhD Student,
Beam Lab

6/14/22

Code to accompany SCIS workshop paper, Self-Supervision on Images and Text Reduces
Reliance on Visual Shortcut Feature

Use src/models/train.py & src/models/train_vision_only.py to train CLIP/CNN models respectively. Use options within to train on shortcut data.

Finetune models with src/models/finetune_chexpert.py. Evaluate zeroshot CLIP, CNN, and finetuned models on real/shortcut/adversarial data by using src/evaluate/evaluate_chexpert_classification.py.

Produce Integrated Gradient maps for individual images or entire datasets with src/models/Integrated_Gradients.py. Analyze integrated gradient similarities with src/evaluate/analyze_ig_chexpert.py.
