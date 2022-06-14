import argparse
import pandas as pd
from CNN import *
import torch
from HelperFunctions import *
import matplotlib.pyplot as plt
import numpy as np
import saliency.core as saliency
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def plot_orig_im(img, fig, ax, x, y, title = "Original image"):
    img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
    img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
    img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
    img = img.permute(0,2,3,1).squeeze()
    if y == -1:
        ax[x].imshow(img, plt.cm.gray, vmin=0, vmax=1)
        ax[x].set_title(title)
        ax[x].set_xticks([])
        ax[x].set_xticks([], minor=True)
        ax[x].set_yticks([])
        ax[x].set_yticks([], minor=True)
    else:
        ax[x,y].imshow(img, plt.cm.gray, vmin=0, vmax=1)
        ax[x,y].set_title(title)
        ax[x,y].set_xticks([])
        ax[x,y].set_xticks([], minor=True)
        ax[x,y].set_yticks([])
        ax[x,y].set_yticks([], minor=True)

def myVisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.
  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
  image_2d = np.sum(image_3d, axis=2)

  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)

  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def call_model_function(img, call_model_args, expected_keys):
    img = torch.tensor(img, dtype=torch.float32)
    #print("Init:", img.shape)
    img = img.permute(0, 3, 1, 2)
    img=img.requires_grad_(True)
    target_class_idx = call_model_args['targind']

    model = call_model_args['model']
    img = img.to(device)
    model = model.to(device)
    output = model(img)
    outputs = output[:, target_class_idx]
    grads = torch.autograd.grad(outputs, img, grad_outputs=torch.ones_like(outputs))
    grads = torch.movedim(grads[0], 1, 3)
    #print("final:", grads.shape)
    gradients = grads.cpu().detach().numpy()
    return {saliency.INPUT_OUTPUT_GRADIENTS: gradients}

def plot_ig_saliency(img, targind, model, myfig, myax, x, y, use_abs = True, actually_plot=True, modname='CNN'):
    ig = saliency.IntegratedGradients()
    baseline = np.zeros(img.shape)
    sig = ig.GetSmoothedMask(img, call_model_function, {'model':model, 'targind':targind}, x_steps=5, x_baseline=baseline, batch_size=20)
    if use_abs:
        gs = saliency.VisualizeImageGrayscale(sig)
    else:
        gs = myVisualizeImageGrayscale(sig)
    if actually_plot:
        myax[x,y].imshow(gs, plt.cm.plasma, vmin=0, vmax=1)
        source = ['NA', 'Real ', 'Synth ']
        if x == 0:
            myax[x, y].set_title(source[y] + modname)
        myax[x,y].set_xticks([])
        myax[x,y].set_xticks([], minor=True)
        myax[x,y].set_yticks([])
        myax[x,y].set_yticks([], minor=True)
    return gs

#Compute and (possibly) plot integrated gradients
def getAttributions(im_dict, real_model,synth_model, heads, target='Cardiomegaly', mod_name='vision', im_number=0, df=None, use_abs = True, actually_plot = True):
    if actually_plot:
        myfig, myax = plt.subplots(2, 3, figsize=(12, 7))
    else:
        myfig, myax = 0, 0

    labstr = "Labels: "
    for i, h in enumerate(heads):
        if h == target:
            myind = i
        if df and df[h] == 1:
            labstr += h
            labstr += ", "

    gsrr = plot_ig_saliency(im_dict['real'].clone().permute(0,2,3,1).numpy().squeeze(), myind,real_model, myfig, myax, 0, 1, use_abs, actually_plot, modname=mod_name)
    gssr = plot_ig_saliency(im_dict[target].clone().permute(0,2,3,1).numpy().squeeze(), myind,real_model, myfig, myax, 1, 1, use_abs, actually_plot, modname=mod_name)
    gsrs = plot_ig_saliency(im_dict['real'].clone().permute(0,2,3,1).numpy().squeeze(), myind,synth_model, myfig, myax, 0, 2, use_abs, actually_plot, modname=mod_name)
    gsss = plot_ig_saliency(im_dict[target].clone().permute(0,2,3,1).numpy().squeeze(), myind,synth_model, myfig, myax, 1, 2,use_abs, actually_plot, modname=mod_name)

    dist_real_im = getIGdistance(gsrr, gsrs)[1]
    dist_synth_im = getIGdistance(gssr, gsss)[1]
    dist_real_mod = getIGdistance(gsrr, gssr)[1]
    dist_synth_mod = getIGdistance(gsrs, gsss)[1] #cosinesim
    if actually_plot:
        print(mod_name)
        print("real image, real vs synth", dist_real_im)
        print("synth image, real vs synth", dist_synth_im)
        print("real model, real im vs synth im", dist_real_mod)
        print("synth model, real im vs synth im", dist_synth_mod)
        plot_orig_im(im_dict['real'].clone(), myfig, myax, 0, 0, title="Test Image")
        plot_orig_im(im_dict[target].clone(), myfig, myax, 1, 0, title="Synthetic Test Image")
        if labstr == "Labels: ":
            labstr = "Labels: No Finding"
        myfig.suptitle("Integrated Gradient, " + mod_name + " model.")
        plt.savefig(args.results_dir + "Integrated_grad_abs_"+str(use_abs)+"_" + target + "_" + mod_name + "_" + str(im_number) + ".png", bbox_inches='tight')
        return
    else:
        return dist_real_im, dist_synth_im, dist_real_mod, dist_synth_mod

#Quantify distance between two integrated gradient maps in several ways (euclidean, cosine sim)
def getIGdistance(sal1, sal2):
    sal1 = sal1.flatten()
    sal2 = sal2.flatten()
    tot2 = np.sqrt(np.sum(np.square(sal2)))
    tot1 = np.sqrt(np.sum(np.square(sal1)))
    normsal2 = sal2/tot2
    normsal1 = sal1/tot1
    return np.sqrt(np.sum(np.square(sal2-sal1))), np.dot(normsal1, normsal2)

def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    dat_normal = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=False)
    dat_overwrites = getDatasets(source=args.sr, subset=['test'], heads=heads, synthetic=True, get_overwrites=True)
    #[loader_normal] = getLoaders(dat_normal, subset=['test'])
    #loader_synths = getLoaders(dat_overwrites, subset=heads)
    dat_normal = dat_normal['test']

    #Vision model
    vision_model_real = getVisionClassifier(args.model_path_real, args.model_real, device, args.embed_size, heads)
    vision_model_synth = getVisionClassifier(args.model_path_synth, args.model_synth, device, args.embed_size,heads)
    je_model_real, transformer_real, tokenizer_real = getSimilarityClassifier(args.je_model_path_real, args.je_model_real, device, args.embed_size, heads,
                                            text_num=1, avg_embedding=True)
    je_model_synth, transformer_synth, tokenizer_synth = getSimilarityClassifier(args.je_model_path_synth, args.je_model_synth, device, args.embed_size, heads,
                                            text_num=1, avg_embedding=True)

    if args.get_finetuned:
        finetuned_vision_model_synth = getVisionClassifier(args.model_path_synth, args.model_synth, device, args.embed_size, heads, add_finetune=True)
        finetuned_je_model_synth = getVisionClassifier(args.je_model_path_synth, args.je_model_synth, device,args.embed_size, heads, add_finetune=True)

    #Just plot for one image
    if args.getOne:
        im_number = 99
        im_dict = {}
        normIm, normDf = dat_normal.__getitem__(im_number)
        normIm = normIm.reshape(1, 3, 224, 224)
        im_dict['real'] = normIm
        for h in heads:
            myim, mydf = dat_overwrites[h].__getitem__(im_number)
            myim = myim.reshape(1, 3, 224, 224)
            im_dict[h] = myim
        for target in heads:
            getAttributions(im_dict, vision_model_real, vision_model_synth, heads, target=target, mod_name="CNN", im_number=im_number, df = normDf, use_abs=False)
            getAttributions(im_dict, je_model_real, je_model_synth, heads, target=target, mod_name="CLIP", im_number=im_number, df = normDf, use_abs=False)
            if args.get_finetuned:
                getAttributions(im_dict, finetuned_vision_model_synth, finetuned_je_model_synth, heads, target=target, mod_name="finetuned", im_number=im_number, df=normDf, use_abs=False)
    #Compute across all images and save results for analysis
    else:
        simDict = {'Vrim':[], 'Vsim':[], 'Vrmo':[], 'Vsmo':[],
                   'Crim':[], 'Csim':[], 'Crmo':[], 'Csmo':[],
                   'Atelectasis':[],'Cardiomegaly':[],'Consolidation':[], 'Edema':[], 'Pleural Effusion':[]}
        for im_number in range(dat_normal.__len__()):
            im_dict = {}
            normIm, normDf = dat_normal.__getitem__(im_number)
            normIm = normIm.reshape(1,3,224,224)
            im_dict['real'] = normIm
            for h in heads:
                myim, mydf = dat_overwrites[h].__getitem__(im_number)
                myim = myim.reshape(1,3,224,224)
                im_dict[h] = myim

            for target in heads:
                Vrim,Vsim,Vrmo,Vsmo = getAttributions(im_dict, vision_model_real, vision_model_synth, heads, target=target, mod_name="vision", im_number=im_number, df=normDf, use_abs=False, actually_plot=False)
                Crim, Csim, Crmo, Csmo = getAttributions(im_dict, je_model_real, je_model_synth, heads, target=target, mod_name="clip",im_number=im_number, df=normDf, use_abs=False, actually_plot=False)

            simDict['Vrim'].append(Vrim)
            simDict['Vsim'].append(Vsim)
            simDict['Vrmo'].append(Vrmo)
            simDict['Vsmo'].append(Vsmo)
            simDict['Crim'].append(Crim)
            simDict['Csim'].append(Csim)
            simDict['Crmo'].append(Crmo)
            simDict['Csmo'].append(Csmo)

            for h in heads:
                simDict[h].append(normDf[h])

        simDF = pd.DataFrame(simDict)
        simDF.to_csv(args.results_dir + 'chexpert_test_similarity.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path_real', type=str,default='/n/data2/hms/dbmi/beamlab/anil/CNN_CLIP_Shortcut_Feature_Reliance/models/je_model/exp6/')
    parser.add_argument('--model_path_real', type=str, default='/n/data2/hms/dbmi/beamlab/anil/CNN_CLIP_Shortcut_Feature_Reliance/models/vision_model/vision_CNN_real/')
    parser.add_argument('--je_model_path_synth', type=str, default='/n/data2/hms/dbmi/beamlab/anil/CNN_CLIP_Shortcut_Feature_Reliance/models/je_model/synth/exp7/')
    parser.add_argument('--model_path_synth', type=str, default='/n/data2/hms/dbmi/beamlab/anil/CNN_CLIP_Shortcut_Feature_Reliance/models/vision_model/vision_CNN_synthetic/', help='path for saving trained models')

    parser.add_argument('--je_model_real', type=str, default='je_model-28.pt')
    parser.add_argument('--model_real', type=str, default='model-14.pt')
    parser.add_argument('--je_model_synth', type=str, default='je_model-12.pt')
    parser.add_argument('--model_synth', type=str, default='model-14.pt', help='path from root to model')

    parser.add_argument('--results_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/CNN_CLIP_Shortcut_Feature_Reliance/results/integrated_gradients/')
    parser.add_argument('--sr', type=str, default='c') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--synth', type=bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')
    parser.add_argument('--usemimic', type=bool, default=False, const=True, nargs='?', help='Use mimic to alter zeroshot')
    parser.add_argument('--getOne', type=bool, default=True, const=False, nargs='?')
    parser.add_argument('--get_finetuned', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1) #32 normally
    args = parser.parse_args()
    print(args)
    main(args)