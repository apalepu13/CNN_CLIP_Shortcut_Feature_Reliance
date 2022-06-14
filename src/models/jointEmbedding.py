from torch import nn
import torch
import CNN
import Transformer
import numpy as np
import Vision_Transformer

class JointEmbeddingModel(nn.Module):
    def __init__(self, embed_dim, use_vit = False, imagenet = True):
        super().__init__()
        self.cnn = CNN.CNN_Embeddings(embed_dim=embed_dim, imagenet=imagenet, freeze=False)
        self.vit = Vision_Transformer.VisionTransformer(output_dim=embed_dim)
        self.use_vit = use_vit
        self.transformer = Transformer.Transformer_Embeddings(embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text):
        if self.use_vit:
            image_features = self.vit(image)
        else:
            image_features = self.cnn(image)

        text_features = self.transformer(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text