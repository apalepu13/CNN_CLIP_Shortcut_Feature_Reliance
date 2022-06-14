import sys
sys.path.insert(0, '../models/')
import pandas as pd
import numpy as np
from Data_Loading import *
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
from Transformer import *
from HelperFunctions import *

multilabel=True #Set to determine script behavior


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion', 'Pneumonia',
                  'No Finding', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Other',
                  'Pneumonia','Pneumothorax', 'Support Devices'])
mimic_dat = getDatasets(source='m', subset = ['test'], get_text=False, heads=heads)
[mimic_loader] = getLoaders(mimic_dat, subset=['test'])

outDF = []
print("Hi")
if multilabel: #Images can have multiple positive labels
    for i, (im1, im2, df, text) in enumerate(mimic_loader):
        dfvals = np.array([df[h].numpy() for h in heads]).T
        dfvals = (dfvals == 1).astype(int)
        for r in np.arange(dfvals.shape[0]):
            myrow = [dfvals[r, k] for k in np.arange(dfvals.shape[1])]
            myrow.append(text[r])
            outDF.append(myrow)

    df = pd.DataFrame(outDF, columns=list(heads).append('Description'))
    df.to_csv('/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries_multilabel.csv')
    print(df.shape)
    print(df.head())
else: #Only including exclusive positivity images
    for i, (im1, im2, df, text) in enumerate(mimic_loader):
        dfvals = np.array([df[h].numpy() for h in heads]).T
        dfvals = (dfvals == 1).astype(int)
        dfsums = np.sum(dfvals, axis=1)
        for r in np.arange(dfvals.shape[0]):
            if dfsums[r] < 1:
                continue
            head_ind = np.argwhere(dfvals[r, :])[0]
            for myh in head_ind:
                myhead = heads[myh]
                for k in np.arange(np.ceil(len(heads) / dfsums[r])):
                    outDF.append([myhead, text[r]])

    df = pd.DataFrame(outDF, columns=['Variable', 'Text'])
    df.to_csv('/n/data2/hms/dbmi/beamlab/anil/CNN_CLIP_Shortcut_Feature_Reliance/data/mimic_label_queries.csv')
    print(df.shape)
    print(np.unique(df['Variable'].values, return_counts=True))


