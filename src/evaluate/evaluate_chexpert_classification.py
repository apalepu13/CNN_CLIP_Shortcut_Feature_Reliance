import argparse
import sys
import pickle
sys.path.insert(0, '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/src/models/joint_embedding_model/')
import torch
from CNN import *
from Pretraining import *
from Transformer import *
from torchmetrics import AUROC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
from sklearn import metrics
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    heads_order = np.array(['Pleural Effusion', 'Edema', 'Consolidation', 'Cardiomegaly', 'Atelectasis'])
    colors = {'Average':'k','Atelectasis': 'r', 'Cardiomegaly': 'tab:orange', 'Consolidation': 'g', 'Edema': 'c',
              'Pleural Effusion': 'tab:purple'}
    test_dat_dict = {'Real Test': 'o', 'Synthetic Test': '^', 'Adversarial Test': 'v'}

    #all 8
    mod_order = ['Real CNN', 'Zeroshot Real CLIP', 'Finetuned Real CNN', 'Finetuned Real CLIP',
                 'Shortcut CNN', 'Zeroshot Shortcut CLIP', 'Finetuned Shortcut CNN', 'Finetuned Shortcut CLIP']
    # just 4
    mod_order2 = ['Finetuned Real CNN', 'Finetuned Real CLIP', 'Finetuned Shortcut CNN', 'Finetuned Shortcut CLIP']
    mod_order3 = ['Real CNN', 'Real CLIP', 'Shortcut CNN','Shortcut CLIP']

    vision_models = ['vision_CNN_real/model-14.pt', 'vision_CNN_synthetic/model-14.pt',
                     'vision_CNN_real/finetuned_model-14.pt', 'vision_CNN_synthetic/finetuned_model-14.pt']
    je_models_zero = ['exp6/je_model-28.pt', 'synth/exp7/je_model-12.pt']
    je_models_fine = ['exp6/finetuned_je_model-28.pt', 'synth/exp7/finetuned_je_model-12.pt']
    all_models = vision_models + je_models_zero + je_models_fine
    name_mods = {'Real CNN': vision_models[0], 'Shortcut CNN': vision_models[1], 'Finetuned Real CNN': vision_models[2],
                 'Finetuned Shortcut CNN': vision_models[3],
                 'Zeroshot Real CLIP': je_models_zero[0], 'Zeroshot Shortcut CLIP': je_models_zero[1],
                 'Finetuned Real CLIP': je_models_fine[0], 'Finetuned Shortcut CLIP': je_models_fine[1]}
    mod_names = {v: k for k, v in name_mods.items()}

    if args.generate:
        if args.sr == 'chexpert' or args.sr == 'c':
            args.sr = 'c'

        if args.subset == 'a' or args.subset == 'all':
            subset = ['all']
        elif args.subset == 't' or args.subset == 'test':
            subset = ['test']

        dat = getDatasets(source=args.sr, subset=subset, synthetic=False) #Real
        [DL] = getLoaders(dat, args, subset=subset)

        dat_synth = getDatasets(source=args.sr, subset=subset, synthetic=True, get_good=True, get_overwrites=True) #Synths
        DL2 = getLoaders(dat_synth, args, subset=heads)

        dat_adv = getDatasets(source=args.sr, subset=subset, synthetic=True, get_adversary=True, get_overwrites=True) #Advs
        DL3 = getLoaders(dat_adv, args, subset=heads)

        models = {}
        for vclass in vision_models:
            models[vclass] = getVisionClassifier(args.vision_model_path, vclass, heads=heads)
            models[vclass].eval()
        for venc in je_models_zero:
            models[venc], transformer, tokenizer = getSimilarityClassifier(args.je_model_path, venc, heads=heads, avg_embedding=True, text_num=30, use_convirt=True, soft=False)
            models[venc].eval()
        for vclass in je_models_fine:
            models[vclass] = getVisionClassifier(args.je_model_path, vclass, heads=heads)
            models[vclass].eval()

        all_aucs, all_synths, all_advs = {}, {}, {}
        for m in all_models:
            print(m)
            outm = m.replace('/', '_')
            isvis = "vision" in outm
            issynth = "synth" in outm
            isfine = "finetuned" in outm

            vision_model = models[m]
            aucs, aucs_synth, aucs_adv, tprs, fprs, thresholds = {}, {}, {}, {}, {}, {}
            auroc = AUROC(pos_label=1)

            #Real
            test_preds, test_targets = get_all_preds(DL, vision_model, heads, device)
            test_preds = test_preds.cpu()
            test_targets = test_targets.cpu()
            for i, h in enumerate(heads):
                fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(test_targets[:, i].int().detach().numpy(), test_preds[:, i].detach().numpy())
                aucs[h] = auroc(test_preds[:, i], test_targets[:, i].int()).item()

            #Synth each label
            for j, h in enumerate(heads):
                test_preds_synth, test_targets_synth = get_all_preds(DL2[j], vision_model, heads, device)
                test_preds_synth = test_preds_synth.cpu()
                test_targets_synth = test_targets_synth.cpu()
                aucs_synth[h] = auroc(test_preds_synth[:, j], test_targets_synth[:, j].int()).item()

            #Adversarial each label
            for j, h in enumerate(heads):
                test_preds_adv, test_targets_adv = get_all_preds(DL3[j], vision_model, heads, device)
                test_preds_adv = test_preds_adv.cpu()
                test_targets_adv = test_targets_adv.cpu()
                aucs_adv[h] = auroc(test_preds_adv[:, j], test_targets_adv[:, j].int()).item()

            #Total auc
            aucs['Total'] = np.mean(np.array([aucs[h] for h in heads]))
            aucs_synth['Total'] = np.mean(np.array([aucs_synth[h] for h in heads]))
            aucs_adv['Total'] = np.mean(np.array([aucs_adv[h] for h in heads]))

            #Output and figures
            print("Normal")
            print("Total AUC avg: ", aucs['Total'])
            for i, h in enumerate(heads):
                print(h, aucs[h])

            #ROC Curve
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = {'Atelectasis': 'r', 'Cardiomegaly': 'tab:orange', 'Consolidation': 'g', 'Edema': 'c',
                      'Pleural Effusion': 'tab:purple'}
            for i, h in enumerate(heads):
                ax.plot(fprs[h], tprs[h], color=colors[h], label=h + ", AUC = " + str(np.round(aucs[h], 4)))
            xrange = np.linspace(0, 1, 100)
            avgTPRS = np.zeros_like(xrange)
            for i, h in enumerate(heads):
                avgTPRS = avgTPRS + np.interp(xrange, fprs[h], tprs[h])
            avgTPRS = avgTPRS / 5
            ax.plot(xrange, avgTPRS, color='k', label="Average, AUC = " + str(np.round(aucs['Total'], 4)))
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_title("ROC Curves for labels", size=30)
            ax.set_xlabel("False Positive Rate", size=24)
            ax.set_ylabel("True Positive Rate", size=24)
            ax.legend(prop={'size': 16})
            plt.savefig(args.results_dir + outm + "_roc_curves.png", bbox_inches="tight")

            print("Synthetic")
            print("Total AUC avg: ", aucs_synth['Total'])
            for i, h in enumerate(heads):
                print(h, aucs_synth[h])

            print("Adversarial")
            print("Total AUC avg: ", aucs_adv['Total'])
            for i, h in enumerate(heads):
                print(h, aucs_adv[h])
            '''
            '''
            fig, ax = plt.subplots()
            x = np.arange(len(heads) + 1)
            width = .2
            ax.bar(x, aucs.values(), width, color='r', label='real test')
            ax.bar(x + width, aucs_synth.values(), width, color='b', label='synthetic test')
            ax.bar(x + 2 * width, aucs_adv.values(), width, color='g', label='adversarial test')
            ax.set_ylabel('AUC')
            ax.set_ylim(0, 1)
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(aucs.keys())
            ax.set_xlabel('Class')
            ax.legend()
            title = mod_names[m] + " AUCs"
            ax.set_title(title)

            plt.savefig(args.results_dir + outm + "_AUCs.png", bbox_inches="tight")
            all_aucs[m] = aucs
            all_synths[m] = aucs_synth
            all_advs[m] = aucs_adv

    if args.generate:
        with open(args.dat_dir + 'all_aucs.pickle', 'wb') as handle:
            pickle.dump([all_aucs, all_synths, all_advs], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.dat_dir + 'all_aucs.pickle', 'rb') as handle:
            all_aucs, all_synths, all_advs = pickle.load(handle)

    fig, axs = plt.subplots(3, 1, figsize = (6, 9), sharex=True)

    x = np.arange(len(mod_order2))
    width = 0.1
    centeroffset = np.array([4, 2, 0, -2, -4])/2

    for z, m in enumerate(mod_order2):
        axs[0].bar(x[z], all_aucs[name_mods[m]]['Total'],width * 7, color = 'k')
        axs[0].text(x[z], 1.01, str(np.round(all_aucs[name_mods[m]]['Total'], 3)), color='k', fontweight='bold', ha='center')
        axs[1].bar(x[z] , all_synths[name_mods[m]]['Total'],width * 7, color = 'k')
        axs[1].text(x[z], 1.01, str(np.round(all_synths[name_mods[m]]['Total'], 3)), color='k', fontweight='bold', ha='center')
        axs[2].bar(x[z], all_advs[name_mods[m]]['Total'],width * 7, color='k')
        axs[2].text(x[z], 1.01, str(np.round(all_advs[name_mods[m]]['Total'], 3)), color='k', fontweight='bold', ha='center')

    for k, h in enumerate(heads_order):
        for z, m in enumerate(mod_order2):
            axs[0].bar(x[z] + width * centeroffset[k], all_aucs[name_mods[m]][h], width, color = colors[h], alpha = 0.9)
            axs[1].bar(x[z] + width * centeroffset[k], all_synths[name_mods[m]][h],width, color = colors[h], alpha = 0.9)
            axs[2].bar(x[z] + width * centeroffset[k], all_advs[name_mods[m]][h],width, color = colors[h], alpha = 0.9)

    #ax.set_title("AUCS for various models")
    auc_names = ["Real AUC", "Shortcut AUC", "Adversarial AUC"]
    for k, ax in enumerate(axs):
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(auc_names[k], size=14, fontweight='bold')
        if k == 2:
            ax.set_xticks(x)
            ax.set_xticklabels(mod_order3, rotation=30, ha='right')
            ax.set_xlabel("Fine-tuned Model", size=14, fontweight='bold')

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles1 = [f("o", colors[i]) for i in colors.keys()]
    labels1 = [c for c in colors.keys()]
    axs[0].legend(handles1, labels1, loc=3, framealpha=1, title="Clinical Label", ncol=2)
    for k, ax in enumerate(axs):
        ax.axvline(x=1.5, color='k', label='axvline - full height')
        #ax.axhline(y=1.0, color='k')
        #ax.axvline(x=3.5, color='k', label='axvline - full height')
        #ax.axvline(x=5.5, color='k', label='axvline - full height')

    plt.savefig(args.results_dir + "all_AUCs.png", bbox_inches="tight")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_model_path', type=str, default = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/vision_model/')
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/', help='path for saving trained models')

    parser.add_argument('--sr', type=str, default='c') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--generate', type=bool, default=False, const=True, nargs='?', help='Regenerate aucs and ROC curves')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/zeroshot/')
    parser.add_argument('--dat_dir', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/')
    args = parser.parse_args()
    print(args)
    main(args)