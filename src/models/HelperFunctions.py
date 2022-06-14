import regex as re
from Data_Loading import *
import torch
import torch.nn as nn
import numpy as np

def getDatasets(source, subset = ['train', 'val', 'test'], synthetic = False,
                get_text = True, get_good=False, get_adversary = False, get_overwrites=False, get_seg=False,
                heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']):
    s = source
    datlist = {}
    if type(subset) == str:
        subset = [subset]

    if get_overwrites:
        for h in heads:
            datlist[h] = Image_Text_Dataset(source=s, group=subset[0], synth=True, overwrite=[h], get_good=get_good, get_adversary=get_adversary)
    else:
        for sub in subset:
            datlist[sub] = Image_Text_Dataset(source=s, group=sub, synth=synthetic,
                                              get_good=get_good, get_adversary=get_adversary, get_text = get_text, get_seg=get_seg,
                                              out_heads = heads)
    return datlist

def getLoaders(datasets, args=None, subset = ['train', 'val', 'test'], num_work = 16, shuffle=True):
    loaders = []
    for sub in subset:
        if args:
            if num_work == 1:
                loaders.append(DataLoader(datasets[sub], batch_size=args.batch_size, shuffle=shuffle))
            else:
                loaders.append(DataLoader(datasets[sub], batch_size=args.batch_size, shuffle=shuffle, num_workers=num_work,
                                  prefetch_factor=max(1, int(args.batch_size/num_work)), pin_memory=True))
        else:
            loaders.append(DataLoader(datasets[sub], batch_size=32, shuffle=shuffle, num_workers=16,
                                      prefetch_factor=2, pin_memory=True))

    return loaders

def getExperiment(args, mp):
    exp = args.exp
    if exp is -1:
        if not os.listdir(os.path.join(mp)):
            print("No models exist, creating directory")
            if not args.debug:
                fp = os.path.join(mp, 'exp1')
        elif os.listdir(os.path.join(mp)):
            all_files = os.listdir(os.path.join(mp))
            je_exps = [exp for exp in all_files if 'exp' in exp]
            num = [int(re.search('\d+', exp).group(0)) for exp in je_exps]
            highest_ind = np.argmax(np.array(num))
            highest = num[highest_ind]
            if not args.resume:
                highest = highest + 1
            fp = os.path.join(mp, 'exp'+str(highest))
        else:
            print("Model doesn't exist, creating directory")
            if not args.debug:
                fp = os.path.join(mp, 'exp'+str(exp))
    else:
        fp = os.path.join(mp, 'exp'+str(exp))
    return fp

def startExperiment(args, je_model, optimizer, fp):
    if args.resume:
        if os.listdir(os.path.join(fp)):
            all_files = os.listdir(os.path.join(fp))
            je_files = [file for file in all_files if 'je_model' in file]
            num = [int(re.search('\d+', file).group(0)) for file in je_files]
            highest = np.argmax(np.array(num))
            loadpath = os.path.join(fp, np.array(je_files)[highest])
            print("Loading " + loadpath)
            checkpoint = torch.load(loadpath)
            je_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']
            best_val_loss = checkpoint['val_loss']
            args = checkpoint['args']
        else:
            print("Experiment doesnt exist", fp)
            start, best_val_loss, args = -1
    else:
        print("Starting from scratch")
        start = 0
        best_val_loss = 1000000000
        if not args.debug:
            os.makedirs(fp)
            txt = args.desc
            with open(os.path.join(fp, "desc.txt"), "w") as text_file:
                text_file.write(txt)
    return start, best_val_loss, args

def clip_loss(im_logits, text_logits, device, loss_weight = 1, criterion = nn.CrossEntropyLoss()):
    samp = torch.tensor(np.arange(im_logits.shape[0]))
    loss_a = criterion(im_logits, samp.to(device))
    loss_b = criterion(text_logits, samp.to(device))
    closs = (loss_a + loss_b) / 2
    closs = closs * loss_weight
    return closs

def train(device, je_model, im1, im2, texts, tokenizer, optimizer):
    je_model.zero_grad(set_to_none=True)
    # Set mini-batch dataset
    images1 = im1.to(device)
    images2 = im2.to(device)
    texts = tokenizer.do_encode(texts=texts).to(device)

    # Forward, backward and optimize
    im_logits1, text_logits1 = je_model(images1, texts)
    im_logits2, text_logits2 = je_model(images2, texts)
    cl1 = clip_loss(im_logits1, text_logits1, device)
    cl2 = clip_loss(im_logits2, text_logits2, device)
    iloss = clip_loss(im_logits1, im_logits2, device)
    loss = cl1 + cl2 + iloss
    return loss



def validate(device, val_data_loader, tokenizer, je_model, criterion, source = "MIMIC", proportion = 1.0):
    vlosses = []
    with torch.no_grad():
        for j, (valims1, valims2, valtexts, patients) in enumerate(val_data_loader):
            gen = np.random.rand(1)
            if gen >= proportion:
                continue

            valims1 = valims1.to(device)
            valims2 = valims2.to(device)
            valtexts = tokenizer.do_encode(texts=valtexts).to(device)
            val_im1, val_t1 = je_model(valims1, valtexts)
            val_im2, val_t2 = je_model(valims2, valtexts)
            myloss = clip_loss(val_im1, val_t1, device)
            myloss += clip_loss(val_im2, val_t2, device)
            myloss += clip_loss(val_im1, val_im2, device)
            vlosses.append(myloss.cpu())

    vlloss = np.mean(np.array(vlosses))
    print(source + ' Val Loss: ' + str(vlloss))
    return vlloss




def b_loss(impreds, labels, device, heads, criterion):
    losses = torch.zeros(len(heads))
    for i, h in enumerate(heads):
        label = labels[h]
        if h == 'Edema' or h == 'Atelectasis':
            label[label == -1.0] = float('nan')
        else:
            label[label == -1.0] = float('nan')
        label[label == 0.0] = 0
        label[label == 1.0] = 1
        #label[torch.isnan(label)] = 0
        label = label.float().to(device)
        mypreds = impreds[torch.logical_not(torch.isnan(label)), i]
        mylabels = label[torch.logical_not(torch.isnan(label))]
        losses[i] = criterion(mypreds, mylabels)
    losses = losses[torch.logical_not(torch.isnan(losses))]
    loss = torch.mean(losses)
    if torch.isnan(loss):
        loss = 0
    return loss

def train_vision(device, vision_model, im1, im2, labels, heads, criterion=torch.nn.BCEWithLogitsLoss(), useOne=False):
    vision_model.zero_grad(set_to_none=True)
    # Set mini-batch dataset
    images1 = im1.to(device)
    if not useOne:
        images2 = im2.to(device)

    # Forward, backward and optimize
    impreds1 = vision_model(images1)
    cl1 = b_loss(impreds1, labels, device, heads, criterion)
    if not useOne:
        impreds2 = vision_model(images2)
        cl2 = b_loss(impreds2, labels, device, heads, criterion)
        loss = cl1 + cl2
    else:
        loss = cl1
    return loss

def validate_vision(device, val_data_loader, vision_model, heads, criterion, source = "MIMIC", proportion = 1.0):
    vlosses = []
    with torch.no_grad():
        for j, res in enumerate(val_data_loader):
            if source == "MIMIC":
                valims1, valims2, val_labels, patients = res
            else:
                valims1, val_labels = res
                valims2 = None

            gen = np.random.rand(1)
            if gen >= proportion:
                continue

            valims1 = valims1.to(device)
            valpred1= vision_model(valims1)
            myloss = b_loss(valpred1, val_labels, device, heads, criterion)
            if valims2 is not None:
                valims2 = valims2.to(device)
                valpred2 = vision_model(valims2)
                myloss = myloss + b_loss(valpred2, val_labels, device, heads, criterion)

            vlosses.append(myloss.cpu())

    vlloss = np.mean(np.array(vlosses))
    print(source + ' Val Loss: ' + str(vlloss))
    return vlloss

def getLabels(df, heads):
    labels = None
    for i, h in enumerate(heads):
        label = df[h].float()
        label[label == -1.0] = float('nan')
        label[label == 0.0] = 0.0
        label[label == 1.0] = 1.0
        if labels is None:
            labels = label
            labels = labels[:, None]
        else:
            labels = torch.cat((labels, label[:, None]), axis=1)

    return labels

def get_all_preds(DL, mod, heads = ['covid19', 'No Finding'], device='cuda'):
    tp, tt = None, None
    for i, res in enumerate(DL):
        try:
            im1, im2, df, study = res
        except:
            im1, df = res

        images = im1.to(device)
        preds = mod(images).to(device)
        labels = getLabels(df, heads).to(device)

        if tp is None:
            tp = preds
        else:
            tp = torch.cat((tp, preds), axis=0)
        if tt is None:
            tt = labels
        else:
            tt = torch.cat((tt, labels), axis=0)

    return tp, tt


