import time
t = time.time()
import argparse
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
import torch.nn as nn
from CNN import *
from HelperFunctions import *
from Vision_Transformer import *
import os
import regex as re
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Specifies which vision model version
def get_loadpath(args):
    save_path = args.model_path + 'vision'
    if args.vit:
        save_path = save_path + '_VIT'
    else:
        save_path = save_path + '_CNN'

    if args.synthetic:
        save_path = save_path + '_synthetic'
    else:
        save_path = save_path + '_real'

    all_files = os.listdir(os.path.join(save_path))
    vis_files = [file for file in all_files if 'model' in file]
    num = [int(re.search('\d+', file).group(0)) for file in vis_files]
    highest = np.argmax(np.array(num))
    loadpath = os.path.join(save_path, np.array(vis_files)[highest])
    return loadpath

#Train CNN/ViT if desired
def main(args):
    criterion = torch.nn.BCEWithLogitsLoss()
    out_heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    if not args.vit:
        cnn_rep = CNN_Embeddings(args.embed_size)
        vision_model = CNN_Classifier(cnn_rep, args.embed_size, freeze = False, num_heads = len(out_heads)).to(device)
    else:
        vision_model = VisionClassifier(output_heads = len(out_heads), output_dim = args.embed_size).to(device)

    params = list(vision_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)

    if args.resume:
        try:
            loadpath = get_loadpath(args)
            print("Resuming",loadpath)
            checkpoint = torch.load(loadpath)
            vision_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']
            val_loss = checkpoint['val_loss']
            args = checkpoint['args']
        except:
            print("Resume failed, Starting from scratch")
            start = 0

    else:
        start = 0


    # Build data
    if args.debug:
        subset = ['tiny', 'tinyval']
    else:
        subset = ['train', 'val']

    mimic_dat = getDatasets(source='m', subset = subset, synthetic = args.synthetic, get_text = False, heads = out_heads)
    train_data_loader_mimic, val_data_loader_mimic = getLoaders(mimic_dat, args, subset=subset, num_work = 16)
    total_step_mimic = len(train_data_loader_mimic)

    anneal_ct = 0

    for epoch in range(start, args.num_epochs):
        vision_model.train()
        tmimic = time.time()
        for i, (im1, im2, labels, patient) in enumerate(train_data_loader_mimic):
            loss = train_vision(device, vision_model, im1, im2, labels, out_heads, criterion)
            loss.backward()
            optimizer.step()
            if i % args.log_step == 0:
                print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step_mimic, loss.item()))
        print("Mimic Epoch time: " + str(time.time() - tmimic))

        if epoch % args.val_step == 0:
            vision_model.eval()
            tval = time.time()
            val_loss = validate_vision(device, val_data_loader_mimic, vision_model, out_heads, criterion, source="MIMIC")

            if val_loss:
                print("Saving model")
                if not args.debug:
                    save_path = args.model_path + 'vision'
                    if args.vit:
                        save_path = save_path + '_VIT'
                    else:
                        save_path = save_path + '_CNN'

                    if args.synthetic:
                        save_path = save_path + '_synthetic'
                    else:
                        save_path = save_path + '_real'

                    torch.save({'epoch': epoch+1,
                                'model_state_dict': vision_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': val_loss,
                                'args': args}, os.path.join(save_path,'model-{}.pt'.format(epoch)))

            print("Val time " + str(time.time() - tval))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/CNN_CLIP_Shortcut_Feature_Reliance/models/vision_model/', help='path for saving trained models')
    parser.add_argument('--log_step', type=int, default=100000, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=2, help='step size for printing val info')
    parser.add_argument('--debug', type=bool, default=False, const = True, nargs='?', help='debug mode, dont save')
    parser.add_argument('--vit', type=bool, default=False, const = True, nargs='?', help='Use vision transformer')
    parser.add_argument('--synthetic', type =bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')
    parser.add_argument('--resume', type=bool, default=False, const=True, nargs='?', help='Resume training')
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32) #32 vs 16
    parser.add_argument('--learning_rate', type=float, default=.0001) #.0001
    args = parser.parse_args()
    print(args)
    main(args)