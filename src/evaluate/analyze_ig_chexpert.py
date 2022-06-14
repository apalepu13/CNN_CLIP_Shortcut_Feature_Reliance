import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

heads = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
real_ims = ['Vrim', 'Crim']
synth_ims = ['Vsim', 'Csim']
real_mods = ['Vrmo', 'Crmo']
synth_mods = ['Vsmo', 'Crmo']
resultsDir = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/results/integrated_gradients/total/'
mycsv = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/chexpert_test_similarity.csv'
mydf = pd.read_csv(mycsv)

print(mydf.head())

#group_pairs = [real_ims, synth_ims, real_mods, synth_mods]
group_pairs = [synth_ims, synth_mods]
#titles = ['Training Data', 'Invariance to Training Data', 'Testing Data', 'Invariance to Testing Data']
titles = ['Invariance to Training Data', 'Invariance to Testing Data']
#colors = [['g', 'g'], ['tab:purple', 'y'], ['g', 'g'], ['tab:orange', 'c']]
colors = [['tab:purple', 'y'], ['tab:orange', 'c']]
for h in heads:
    print(h)
    fig, axs = plt.subplots(2, 1, figsize=(4, 7.5))
    axs = axs.flatten()
    print(axs)
    hDF = mydf[mydf[h] == 1.0]
    boxprops = dict(color="black", linewidth=1.5)
    medianprops = dict(color="black", linewidth=1.5)
    for i, group in enumerate(group_pairs):
        print(group)
        lol = axs[i].boxplot([hDF[group[0]].values, hDF[group[1]].values], patch_artist=True, boxprops=boxprops,medianprops=medianprops)
        for j, patch in enumerate(lol['boxes']):
            patch.set_facecolor(colors[i][j])

        print(np.mean(hDF[group[0]].values), np.mean(hDF[group[1]].values))
        if i == 100:
            axs[i].set_title("Integrated Gradient Cosine Similarity\n" + titles[i])
        else:
            axs[i].set_title(titles[i])

        axs[i].set_ylabel("Cosine similarity")
        axs[i].set_xticklabels(['CNN', 'CLIP'])
        for j, t in enumerate(axs[i].get_xticklabels()):
            t.set_color(colors[i][j])

        if i > 1:
            axs[i].set_xlabel("Model")

        axs[i].set_ylim([0, 1])
    #fig.suptitle("Integrated Gradient Cosine Similarities", fontweight='bold', size=16)
    plt.savefig(resultsDir + h + '.png', bbox_inches='tight')

