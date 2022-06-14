import time
t = time.time()
import argparse
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
import torch.nn as nn
from Transformer import *
from jointEmbedding import JointEmbeddingModel
from HelperFunctions import *
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

elapsed = time.time() - t
print("Start (time = " + str(elapsed) + ")")


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    t = time.time()
    # Build the models
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    je_model = JointEmbeddingModel(args.embed_size, args.vit).to(device)
    params = list(je_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)
    tokenizer = Bio_tokenizer()

    mp = args.model_path
    if hasattr(args, 'synthetic') and args.synthetic:
        mp = mp + 'synth/'
    print(mp)

    fp = getExperiment(args, mp)
    print("experiment path", fp)
    start, best_val_loss, args = startExperiment(args, je_model, optimizer, fp)
    # Build data
    if args.debug:
        subset = ['tiny', 'tinyval']
    else:
        subset = ['train', 'val']

    if not hasattr(args, 'synthetic') or not args.synthetic:
        print("Real Images")
        mimic_dat = getDatasets(source='m', subset = subset)
        indiana_dat = getDatasets(source='i', subset = subset)
        train_data_loader_mimic, val_data_loader_mimic = getLoaders(mimic_dat, args, subset=subset, num_work=16)
        #train_data_loader_indiana, val_data_loader_indiana = getLoaders(indiana_dat, args, subset=subset)
        # Train the models
        total_step_mimic = len(train_data_loader_mimic)
        #total_step_indiana = len(train_data_loader_indiana)
    else:
        print("Synthetic Images")
        mimic_dat = getDatasets(source='m', subset = subset, synthetic=True)
        train_data_loader_mimic, val_data_loader_mimic = getLoaders(mimic_dat, args, subset=subset, num_work=16)
        # Train the models
        total_step_mimic = len(train_data_loader_mimic)






    anneal_ct = 0

    for epoch in range(start, args.num_epochs):
        je_model.train()
        tmimic = time.time()
        for i, (im1, im2, texts, patient) in enumerate(train_data_loader_mimic):
            loss = train(device, je_model, im1, im2, texts, tokenizer, optimizer)
            loss.backward()
            optimizer.step()
            if i % args.log_step == 0:
                print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step_mimic, loss.item()))
        print("Mimic Epoch time: " + str(time.time() - tmimic))

        '''
        if not hasattr(args, 'synthetic') or not args.synthetic:
            tindiana = time.time()
            for i, (im1,im2, texts, patient) in enumerate(train_data_loader_indiana):
                loss = train(device, je_model, im1, im2, texts, tokenizer, optimizer) * 20
                loss.backward()
                optimizer.step()
                if i % args.log_step == 0:
                    print('Indiana Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step_indiana, loss.item()))
            print("Indiana Epoch time " + str(time.time() - tindiana))
        '''

        if epoch % args.val_step == 0:
            je_model.eval()
            tval = time.time()
            if not hasattr(args, 'synthetic') or not args.synthetic:
                val_loss = validate(device, val_data_loader_mimic, tokenizer, je_model, criterion, source="MIMIC") # * (3.0/4)
                #val_loss += validate(device, val_data_loader_indiana, tokenizer, je_model, criterion, source="Indiana") * (1.0/4)
            else:
                val_loss = validate(device, val_data_loader_mimic, tokenizer, je_model, criterion, source="MIMIC")

            if val_loss:
                anneal_ct = 0
                print("Saving model")
                if not args.debug:
                    torch.save({'epoch': epoch+1,
                                'model_state_dict': je_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': val_loss,
                                'args': args}, os.path.join(fp, 'je_model-{}.pt'.format(epoch)))
            else:
                anneal_ct += 1
                if anneal_ct == 3:
                    anneal_ct = 0
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * 0.5

            print("Val time " + str(time.time() - tval))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/models/je_model/', help='path for saving trained models')
    parser.add_argument('--log_step', type=int, default=100000, help='step size for printing log info')
    parser.add_argument('--val_step', type=int, default=2, help='step size for printing val info')
    parser.add_argument('--resume', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--exp', type=int, default=-1, help='experiment number')
    parser.add_argument('--desc', type=str, default="", help='experiment description')
    parser.add_argument('--debug', type=bool, default=False, const = True, nargs='?', help='debug mode, dont save')
    parser.add_argument('--vit', type=bool, default=False, const = True, nargs='?', help='Use vision transformer')
    parser.add_argument('--synthetic', type =bool, default=False, const=True, nargs='?', help='Train on synthetic dataset')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32) #32 vs 16
    parser.add_argument('--learning_rate', type=float, default=.0001) #.0001
    args = parser.parse_args()
    print(args)
    main(args)