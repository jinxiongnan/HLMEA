import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import MPNetModel, MPNetConfig, AutoTokenizer, AutoModel, BertModel, BertTokenizerFast
from tqdm import tqdm
import json
from dataloader import create_dataloader
import argparse

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


class CustomMPNetModel(nn.Module):
    def __init__(self):
        super(CustomMPNetModel, self).__init__()
        self.mpnet = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def forward(self, input_ids, attention_mask=None, eval=False):
        outputs = self.mpnet(input_ids=input_ids, attention_mask=attention_mask)
        if eval:
            return outputs
        return outputs.pooler_output


class CustomE5Model(nn.Module):
    def __init__(self):
        super(CustomE5Model, self).__init__()
        self.e5 = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

    def forward(self, input_ids, attention_mask=None, eval=False):
        outputs = self.e5(input_ids=input_ids, attention_mask=attention_mask)
        if eval:
            return outputs
        return outputs.pooler_output


class CustomLabseModel(nn.Module):
    def __init__(self):
        super(CustomLabseModel, self).__init__()
        self.labse = BertModel.from_pretrained("setu4993/LaBSE")

    def forward(self, input_ids, attention_mask=None, eval=False):
        outputs = self.labse(input_ids=input_ids, attention_mask=attention_mask)
        if eval:
            return outputs
        return outputs.pooler_output


class CustomMinilmModel(nn.Module):
    def __init__(self):
        super(CustomMinilmModel, self).__init__()
        self.minilm = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def forward(self, input_ids, attention_mask=None, eval=False):
        outputs = self.minilm(input_ids=input_ids, attention_mask=attention_mask)
        if eval:
            return outputs
        return outputs.pooler_output


def pairwise_margin_loss(embeddings_anchor, embeddings_positive, embeddings_negative, margin=10.0):
    positive_distance = (embeddings_anchor - embeddings_positive).pow(2).sum(1)
    negative_distance = (embeddings_anchor - embeddings_negative).pow(2).sum(1)
    losses = torch.relu(positive_distance - negative_distance + margin)
    return losses.mean()


def tokenize_and_convert_to_tensor(ptm_model_name: str, texts, max_length=512, device=None):
    if 'e5' in ptm_model_name:
        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    elif 'labse' in ptm_model_name:
        tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    elif 'mpnet' in ptm_model_name:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    elif 'minilm' in ptm_model_name:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    else:
        raise RuntimeError('Invalid PTM model name %s for tokenizer' % ptm_model_name)
    ## Tokenize all texts at once, ensuring consistent padding
    try:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    except Exception as e:
        print('Error:', e)
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    if device:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    return input_ids, attention_mask


def evaluate(ptm_model_name, model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            text_descriptions = batch['text_description']
            descriptions_pos = batch['description_pos']
            descriptions_neg = batch['description_neg']

            text_ids, text_mask = tokenize_and_convert_to_tensor(ptm_model_name, text_descriptions, device=device)
            pos_ids, pos_mask = tokenize_and_convert_to_tensor(ptm_model_name, descriptions_pos, device=device)
            neg_ids, neg_mask = tokenize_and_convert_to_tensor(ptm_model_name, descriptions_neg, device=device)

            text_ids, pos_ids, neg_ids = text_ids.to(device), pos_ids.to(device), neg_ids.to(device)
            text_mask, pos_mask, neg_mask = text_mask.to(device), pos_mask.to(device), neg_mask.to(device)

            embeddings_anchor = model(text_ids, text_mask)
            embeddings_pos = model(pos_ids, pos_mask)
            embeddings_neg = model(neg_ids, neg_mask)

            loss = pairwise_margin_loss(embeddings_anchor, embeddings_pos, embeddings_neg)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss


# Training function
def train(ptm_model_name, ptm_model, train_dataloader, dev_dataloader, optimizer, device, epochs=10, patience=3,
          save_path='best_model.pth'):
    ptm_model.train()
    ptm_model.to(device)
    best_dev_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            text_descriptions = batch['text_description']
            descriptions_pos = batch['description_pos']
            descriptions_neg = batch['description_neg']

            text_ids, text_mask = tokenize_and_convert_to_tensor(ptm_model_name, text_descriptions, device=device)
            pos_ids, pos_mask = tokenize_and_convert_to_tensor(ptm_model_name, descriptions_pos, device=device)
            neg_ids, neg_mask = tokenize_and_convert_to_tensor(ptm_model_name, descriptions_neg, device=device)

            text_ids, pos_ids, neg_ids = text_ids.to(device), pos_ids.to(device), neg_ids.to(device)
            text_mask, pos_mask, neg_mask = text_mask.to(device), pos_mask.to(device), neg_mask.to(device)

            embeddings_anchor = ptm_model(text_ids, text_mask)
            embeddings_pos = ptm_model(pos_ids, pos_mask)
            embeddings_neg = ptm_model(neg_ids, neg_mask)

            loss = pairwise_margin_loss(embeddings_anchor, embeddings_pos, embeddings_neg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_dataloader)
        dev_loss = evaluate(ptm_model_name, ptm_model, dev_dataloader, device)

        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_dataloader)}, Dev Loss: {dev_loss}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            no_improvement = 0
            torch.save(ptm_model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with Dev Loss: {dev_loss}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"No improvement in Dev Loss for {patience} consecutive epochs. Stopping training early.")
                break

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Dev Loss: {dev_loss}")