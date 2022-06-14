import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

def getTextEmbeddings(heads, transformer, tokenizer, use_convirt = False, device = 'cuda', get_num=1):
    if use_convirt:
        filename = '/n/data2/hms/dbmi/beamlab/chexpert/convirt-retrieval/text-retrieval/query_custom.csv'
    else:
        filename = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/mimic_label_queries.csv'

    covid_filename = '/n/data2/hms/dbmi/beamlab/anil/Med_ImageText_Embedding/data/covid_queries.csv'
    covidcsv = pd.read_csv(covid_filename)
    zeroshots = ['covid19', 'Pneumonia', 'No Finding', 'Lungs', 'LeftLung', 'RightLung']

    mycsv = pd.read_csv(filename)
    if not use_convirt:
        l = []
        for h in heads:
            if h in zeroshots:
                temp = covidcsv[covidcsv['Variable'] == h]
                l.append(temp.sample(n=get_num, replace=True))
            else:
                temp = mycsv[mycsv['Variable'] == h]
                l.append(temp.sample(n=get_num, replace=True))
        mycsv = pd.concat(l)

    lab = mycsv['Variable']
    desc = mycsv['Text'].values
    bsize = 32
    numbatches = int(desc.shape[0] / bsize) + 1
    e = None
    for i in np.arange(numbatches):
        mydesc = desc[i * bsize:((i + 1) * bsize)]
        t = tokenizer.do_encode(list(mydesc)).to(device)
        if e is None:
            e = torch.tensor(transformer(t))
        else:
            e = torch.cat((e, torch.tensor(transformer(t))), axis=0)

    head_dict = {}
    for A, B in zip(heads, np.arange(heads.shape[0])):
        head_dict[A] = B
    outlabs = torch.tensor(lab.map(head_dict).values)
    return torch.tensor(e), outlabs

class Bio_tokenizer():
    def __init__(self, use_cxr_bert = True):
        self.use_cxr_bert = use_cxr_bert
        if not self.use_cxr_bert:
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            url = "microsoft/BiomedVLP-CXR-BERT-specialized"
            self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    def do_encode(self, texts):
        texts = [t for t in texts]
        if not self.use_cxr_bert:
            encodings = self.tokenizer(texts, padding=True, truncation=True, max_length = 256, return_tensors="pt")
        else:
            encodings = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                        add_special_tokens=True,
                                        padding=True, truncation=True, max_length=256,
                                        return_tensors='pt')
        return encodings

class Transformer_Embeddings(nn.Module):
    def __init__(self, embed_dim = 128, use_cxr_bert = True):
        super().__init__()
        self.use_cxr_bert = use_cxr_bert

        if not self.use_cxr_bert:
            self.embed_dim = embed_dim
            self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            modules = [self.model.embeddings, *self.model.encoder.layer[:8]]  # freeze bottom 8 layers only
            self.linear1 = nn.Linear(768, self.embed_dim)  # fc layer to embed dim
        else:
            url = "microsoft/BiomedVLP-CXR-BERT-specialized"
            self.model = AutoModel.from_pretrained(url, trust_remote_code=True)
            modules = [self.model.bert.embeddings, *self.model.bert.encoder.layer[:], self.model.cls, self.model.cls_projection_head ]

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, text):
        if self.use_cxr_bert:
            embeddings = self.model.get_projected_text_embeddings(input_ids=text.input_ids,attention_mask=text.attention_mask)
        else:
            embeddings = self.model(input_ids = text['input_ids'], attention_mask = text['attention_mask'])
            embeddings = self.linear1(embeddings.pooler_output)
        return embeddings

if __name__ == '__main__':
    bt = Bio_tokenizer()
    te = Transformer_Embeddings(128)
    toks = bt.do_encode(["Hi my name is Anil."])
    embeds = te(toks)
    print(te)
    print(toks)
