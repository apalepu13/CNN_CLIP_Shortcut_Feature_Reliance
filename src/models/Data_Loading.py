import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re

def textProcess(text):
    sections = 'WET READ:|FINAL REPORT |INDICATION:|HISTORY:|STUDY:|COMPARISONS:|COMPARISON:|TECHNIQUE:|FINDINGS:|IMPRESSION:|NOTIFICATION:'
    mydict = {}
    labs = re.findall(sections, text)
    splits = re.split(sections, text)
    for i, l in enumerate(splits):
        if i == 0:
            continue
        else:
            if len(splits[i]) > 50 or labs[i-1] == 'IMPRESSION:':
                mydict[labs[i-1]] = splits[i]
    if 'FINDINGS:' in mydict.keys():
        if 'IMPRESSION:' in mydict.keys():
            mystr =  "FINDINGS: " + mydict['FINDINGS:'] + "IMPRESSION: " + mydict['IMPRESSION:']
        else:
            mystr =  "FINDINGS: " + mydict['FINDINGS:']
    else:
        mystr = ""
        if 'COMPARISONS:' in mydict.keys():
            mystr = mystr + "COMPARISONS: " + mydict['COMPARISONS:']
        if 'COMPARISON:' in mydict.keys():
            mystr = mystr + "COMPARISONS: " + mydict['COMPARISON:']
        if 'IMPRESSION:' in mydict.keys():
            mystr = mystr + "IMPRESSION: " + mydict['IMPRESSION:']
    if len(mystr) > 100:
        return mystr
    else:
        if 'FINAL REPORT ' in mydict.keys() and len(mydict['FINAL REPORT ']) > 100:
            return mydict['FINAL REPORT '] + mystr
        else:
            return text


def make_synthetic(image, incorrect, correct, p_watermark = .9, p_correct = .9, overwrite = False, get_good = False, get_adversary = False):
    if isinstance(overwrite, str):
        overwrite = [overwrite]
    if get_adversary:
        p_watermark=1
        p_correct=0
    elif get_good:
        p_watermark=1
        p_correct=1

    do_watermark, do_correct = np.random.random((2,))
    do_watermark = do_watermark < p_watermark
    do_correct = do_correct < p_correct
    watermark_image = image.copy()
    shortcut = False
    shortcutDict = {'Atelectasis': 'A', 'Cardiomegaly':'C', 'Consolidation':'N', 'Edema':'E', 'Pleural Effusion': 'P',
                    'A': 'A', 'C':'C', 'N':'N', 'E':'E', 'P':'P'}
    locDict = {'A':20, 'C':60, 'N':100, 'E':140, 'P':180}

    if do_watermark or overwrite:
        draw = ImageDraw.Draw(watermark_image)
        font = ImageFont.truetype("/usr/share/fonts/urw-base35/P052-Roman.otf", 30)

        if not overwrite:
            if do_correct and len(correct) > 0:
                #shortcut = np.random.choice(correct)
                for s in correct:
                    swrite = shortcutDict[s]
                    change = np.random.randint(11, size=2) - 5
                    sloc = (locDict[swrite] + change[0], 20 + change[1])
                    draw.text(sloc, swrite, (255, 255, 255), font=font)
                return watermark_image, correct
            elif not do_correct and len(incorrect) > 0:
                #shortcut = np.random.choice(incorrect)
                for s in incorrect:
                    swrite = shortcutDict[s]
                    change = np.random.randint(11, size=2) - 5
                    sloc = (locDict[swrite] + change[0], 20 + change[1])
                    draw.text(sloc, swrite, (255, 255, 255), font=font)
                return watermark_image, incorrect
            else:
                shortcut = False
                return watermark_image, shortcut
        else:
            for s in overwrite:
                if get_adversary and overwrite[0] in correct:
                    return watermark_image, overwrite
                elif get_good and overwrite[0] not in correct: #make sure we are actually going in here.
                    return watermark_image, overwrite
                else:
                    swrite = shortcutDict[s]
                    change = np.random.randint(11, size=2) - 5
                    sloc = (locDict[swrite] + change[0], 20 + change[1])
                    draw.text(sloc, swrite, (255, 255, 255), font=font)
            return watermark_image, overwrite


    return watermark_image, shortcut

class Image_Text_Dataset(Dataset):
    """Chx - Report dataset.""" #d

    def __init__(self, source = 'mimic_cxr', group='train',
                 synth = False,overwrite = False, get_good=False, get_adversary=False,
                 get_text = True, grayscale=True, get_seg=False,
                 out_heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 mimic_csv_file='/n/data2/hms/dbmi/beamlab/mimic_cxr/mimic-cxr-2.0.0-split.csv', mimic_root_dir='/n/data2/hms/dbmi/beamlab/mimic_cxr/',
                 mimic_chex_file='/n/data2/hms/dbmi/beamlab/mimic_cxr/mimic-cxr-2.0.0-chexpert.csv', chexpert_root_dir='/n/data2/hms/dbmi/beamlab/chexpert/',
                 text_process=True):

        self.heads = out_heads
        self.out_heads = out_heads

        self.synth = synth
        self.get_good=get_good
        self.get_adversary=get_adversary
        self.overwrite = overwrite

        self.get_text = get_text
        self.text_process = text_process
        self.source = source
        self.grayscale = grayscale
        if self.source == 'm':
            self.source = 'mimic_cxr'
        elif self.source == 'c':
            self.source = 'chexpert'
        self.group = group

        # Preprocessing
        self.im_preprocessing_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, ratio=(.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(20, translate=(.1, .1), scale=(.95, 1.05)),
            transforms.ColorJitter(brightness=.4, contrast=.4),
            transforms.GaussianBlur(kernel_size=15, sigma=(.1, 3.0)),
            transforms.Resize(size=(224, 224))])

        self.im_finish = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.totensor = transforms.Compose([
            transforms.ToTensor()
        ])


        self.im_preprocessing_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)])

        if self.source == 'mimic_cxr':
            self.im_list_labels = pd.read_csv(mimic_chex_file)
            self.im_list = pd.read_csv(mimic_csv_file)
            self.im_list = self.im_list.merge(self.im_list_labels, on=['subject_id', 'study_id'])
            if group == 'tiny':
                group = 'train'
                self.im_list = self.im_list.iloc[::1000, :]
            if group == 'tinyval':
                group = 'val'
                self.im_list = self.im_list.iloc[::50, :]

            if group == 'train':
                self.im_list = self.im_list[self.im_list['split'] != 'test']
            elif group == 'val' or group == 'test':
                self.im_list = self.im_list[self.im_list['split'] == 'test']

            self.im_list['pGroup'] = np.array(["p" + pg[:2] for pg in self.im_list['subject_id'].values.astype(str)])
            self.im_list['pName'] = np.array(["p" + pn for pn in self.im_list['subject_id'].values.astype(str)])
            self.im_list['sName'] = np.array(["s" + sn for sn in self.im_list['study_id'].values.astype(str)])
            #self.im_list = self.im_list.drop_duplicates(subset = ['pName', 'sName']) #only 1 image per report
            self.root_dir = mimic_root_dir

        elif self.source == 'chexpert':
            self.root_dir = chexpert_root_dir
            im_list_train = pd.read_csv(chexpert_root_dir + 'CheXpert-v1.0-small/train.csv')
            im_list_val = pd.read_csv(chexpert_root_dir + 'CheXpert-v1.0-small/valid.csv')
            if group == 'train':
                self.im_list = im_list_train
                print(self.im_list.index)
                self.im_list = self.im_list[self.im_list.index % 100 != 0]
            elif group == 'valtrain':
                self.im_list = im_list_train
                self.im_list = self.im_list[self.im_list.index % 100 == 0]
            elif group == 'valval':
                self.im_list = im_list_train
                self.im_list = self.im_list[self.im_list.index % 500 == 1]
            elif group == 'test':
                self.im_list = im_list_val
            elif group=='candidates' or group == 'synth_candidates':
                self.im_list = im_list_train
                temp = self.im_list.loc[:, self.out_heads]
                tempsum = (temp.values == 1).sum(axis = 1) == 1
                unknownsum = (temp.values == -1).sum(axis = 1) == 0
                both = np.logical_and(tempsum, unknownsum)
                uniquePos = self.im_list.iloc[both, :]
                uniquePosPer = [uniquePos[uniquePos[h] == 1] for h in self.out_heads]
                uniquePosLim = [u.iloc[:100, :] for u in uniquePosPer]
                self.im_list = pd.concat(uniquePosLim)


            #pd.read_csv(chexpert_root_dir + 'convirt-retrieval/image-retrieval/candidate.csv')
            elif group=='queries' or group=='synth_queries':
                self.im_list = im_list_train
                temp = self.im_list.loc[:, self.out_heads]
                tempsum = (temp.values == 1).sum(axis=1) == 1
                unknownsum = (temp.values == -1).sum(axis=1) == 0
                both = np.logical_and(tempsum, unknownsum)
                uniquePos = self.im_list.iloc[both, :]
                uniquePosPer = [uniquePos[uniquePos[h] == 1] for h in self.out_heads]
                uniquePosLim = [u.iloc[100:120, :] for u in uniquePosPer]
                self.im_list = pd.concat(uniquePosLim)


            elif group == 'all':
                self.im_list = pd.concat((im_list_train, im_list_val), axis=0)

            if group != 'queries':
                self.im_list['Sex'] = (self.im_list['Sex'].values == 'Male').astype(int)


        print(self.source, group + " size= " + str(self.im_list.shape))



    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.source == 'mimic_cxr':
            ims = self.im_list.iloc[idx, :]
            study = ims['sName']
            img_name = os.path.join(self.root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['dicom_id'] + '.jpg')
            image = Image.open(img_name)
            #print(image.size)
            if self.grayscale:
                image = image.convert("RGB")

            image1 = self.im_preprocessing_train(image)
            image2 = self.im_preprocessing_train(image)
            if self.synth:
                df = ims.loc[self.heads]
                incorrect = [s for s in self.heads if df[s] != 1.0]
                correct = [s for s in self.heads if df[s] == 1.0]
                if not self.overwrite:
                    image1, shortcut1 = make_synthetic(image1_orig, incorrect, correct, get_adversary=self.get_adversary,get_good=self.get_good)
                    image2, shortcut2 = make_synthetic(image2_orig, incorrect, correct, get_adversary=self.get_adversary,get_good=self.get_good)
                else:
                    image1, shortcut1 = make_synthetic(image1_orig, incorrect, correct, overwrite=self.overwrite, get_adversary=self.get_adversary,get_good=self.get_good)
                    image2, shortcut2 = make_synthetic(image2_orig, incorrect, correct, overwrite=self.overwrite, get_adversary=self.get_adversary,get_good=self.get_good)

            image1 = self.im_finish(image1)
            image2 = self.im_finish(image2)


            text_name = os.path.join(self.root_dir, ims['pGroup'], ims['pName'], ims['sName'], ims['sName'] + '.txt')
            with open(text_name, "r") as text_file:
                text = text_file.read()
            text = text.replace('\n', '')
            if self.text_process:
                text = textProcess(text)

            sample = (image1, image2, text, study)

            if not self.get_text:
                df = ims.loc[self.out_heads]
                return (image1, image2, df.to_dict(), text)

            return sample

        elif self.source == 'chexpert':
            df = self.im_list.iloc[idx, :]
            if self.synth:
                img_name = self.root_dir + df['Path']
            else:
                img_name = self.root_dir + df['Path']

            image = Image.open(img_name)
            image = image.convert("RGB")
            image = self.im_preprocessing_test(image)

            if self.synth:
                incorrect = [s for s in self.heads if df[s] != 1.0]
                correct = [s for s in self.heads if df[s] == 1.0]
                if not self.overwrite:
                    image, shortcut = make_synthetic(image, incorrect, correct, get_adversary=self.get_adversary, get_good=self.get_good)
                else:
                    image, shortcut = make_synthetic(image, incorrect, correct, overwrite=self.overwrite, get_adversary=self.get_adversary, get_good=self.get_good)

            image = self.im_finish(image)

            df = df.loc[self.out_heads]

            sample = (image, df.to_dict())
            return sample


train_dat = Image_Text_Dataset(source = 'chexpert', group = 'all', synth=False)
for i in np.arange(3333,3337):
    result = train_dat.__getitem__(i)

