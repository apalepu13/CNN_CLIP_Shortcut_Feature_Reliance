import torch
from torch import nn
import numpy as np
import Transformer
import jointEmbedding
import Vision_Transformer


class CNN_Embeddings(nn.Module):
    def __init__(self, embed_dim=128, imagenet = True, freeze = False):
        super().__init__()

        if imagenet:
            self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        else:
            self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            for param in self.resnet.parameters():
                param.requires_grad = True

        self.resnet.fc = nn.Linear(2048, embed_dim)

    def forward(self, image):
        im_embeddings = self.resnet(image)
        return im_embeddings

class CNN_Classifier(nn.Module):
    def __init__(self, cnn_model, embed_size=128, freeze=True, num_heads=5):
        super().__init__()

        if freeze:
            for param in cnn_model.parameters():
                param.requires_grad = False
        else:
            for param in cnn_model.parameters():
                param.requires_grad = True

        self.cnn_model = cnn_model
        self.relu = nn.ReLU()
        self.classification_head = nn.Linear(embed_size, num_heads)

    def forward(self, image):
        embedding = self.cnn_model(image)
        output = self.relu(embedding)
        output = self.classification_head(output)
        return output

class CNN_Similarity_Classifier(nn.Module):
    def __init__(self, cnn_model,transformer_model, tokenizer, embed_size=128, freeze=True,
               heads=np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
               , use_convirt=False, device='cuda', get_num=20, avg_embedding=True, soft=False):

        super().__init__()
        if freeze:
            for param in cnn_model.parameters():
                param.requires_grad = False
        self.cnn_model = cnn_model
        self.transformer_model = transformer_model
        self.tokenizer = tokenizer
        self.embed_size = embed_size
        self.heads = heads
        self.device = device
        self.tembed, self.tlab = Transformer.getTextEmbeddings(heads=self.heads, transformer=self.transformer_model, tokenizer=self.tokenizer,
                                                               use_convirt=use_convirt, device=device,
                                                               get_num=get_num)
        self.get_num = get_num
        self.avg_embedding = avg_embedding
        self.softmax = nn.Softmax(dim=1)
        self.soft = soft
    def forward(self, image):
        embedding = self.cnn_model(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        class_score = torch.zeros(embedding.shape[0], self.heads.shape[0]).to(self.device)
        for i, h in enumerate(self.heads):
            t = self.tembed[self.tlab == i, :]
            tembed = t / t.norm(dim=-1, keepdim=True)
            if self.avg_embedding:
                if self.get_num > 1:
                    tembed = tembed.mean(dim=0)
                tembed = tembed/tembed.norm(dim=-1, keepdim=True)
                head_sim = embedding @ tembed.t()
                head_sim = head_sim.squeeze()
                class_score[:, i] = head_sim
            else:
                head_sims = embedding @ tembed.t()
                if self.get_num > 1:
                    class_score[:, i] = head_sims.mean().squeeze()
        if self.soft:
            return self.softmax(class_score)
        else:
            return class_score


#Outputs pred probabilities
def getVisionClassifier(modpath, mod="", device='cuda', embed_size=512,
                heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']),
                        je=False, getFrozen=True, add_finetune=False):
    if add_finetune:
        loadpath=modpath + 'finetuned_' + mod
    else:
        loadpath = modpath + mod

    if device == 'cuda':
        checkpoint = torch.load(loadpath)
    else:
        checkpoint = torch.load(loadpath, map_location=torch.device('cpu'))

    if je:
        jemodel = jointEmbedding.JointEmbeddingModel(embed_dim=embed_size)
        jemodel.load_state_dict(checkpoint['model_state_dict'])
        cnn = jemodel.cnn
        vision_model = CNN_Classifier(cnn, embed_size=embed_size, freeze=getFrozen, num_heads= heads.shape[0]).to(device)
        vision_model.eval()
    else:
        if 'VIT' in modpath or 'VIT' in mod:
            vision_model = Vision_Transformer.VisionClassifier(len(heads), embed_size).to(device)
        else:
            cnn = CNN_Embeddings(embed_size).to(device)
            vision_model = CNN_Classifier(cnn, embed_size, freeze=getFrozen, num_heads = heads.shape[0]).to(device)
        vision_model.load_state_dict(checkpoint['model_state_dict'])
        vision_model.eval()

    if getFrozen:
        for param in vision_model.cnn_model.parameters():
            param.requires_grad = False
    else:
        for param in vision_model.cnn_model.parameters():
            param.requires_grad = True

    return vision_model

#Outputs cosine similarities
def getSimilarityClassifier(modpath, mod="", device='cuda', embed_size=512,
                heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']),
                            text_num=20, avg_embedding=True, use_convirt=False, soft=False):
    je_model_path = modpath + mod
    if device == 'cuda':
        checkpoint = torch.load(je_model_path)
    else:
        checkpoint = torch.load(je_model_path, map_location=torch.device('cpu'))

    je_model = jointEmbedding.JointEmbeddingModel(embed_size).to(device)
    je_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    transformer_model = je_model.transformer
    tokenizer = Transformer.Bio_tokenizer()
    if not hasattr(checkpoint['args'], 'vit') or not checkpoint['args'].vit:
        cnn = je_model.cnn
        je_vision_model = CNN_Similarity_Classifier(cnn_model=cnn, transformer_model=transformer_model,
                                                    tokenizer=tokenizer, heads = heads,
                                                    device=device, get_num=text_num, avg_embedding=avg_embedding, use_convirt=use_convirt, soft=soft)
    else:
        je_vision_model = je_model.vit

    je_vision_model.to(device)
    je_vision_model.eval()
    transformer_model.eval()
    return je_vision_model, transformer_model, tokenizer


