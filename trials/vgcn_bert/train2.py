import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=512,
                                              return_tensors='pt')
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = criterion(output.logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * label.size(0)
        train_correct += (torch.argmax(output.logits, dim=1) == label).sum().item()
        train_total += label.size(0)
    train_loss /= train_total
    train_acc = train_correct / train_total
    return train_loss, train_acc


def validate(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0
    valid_correct = 0
    valid_total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output.logits, label)
            valid_loss += loss.item() * label.size(0)
            valid_correct += (torch.argmax(output.logits, dim=1) == label).sum().item()
            valid_total += label.size(0)
            y_true += label.tolist()
            y_pred += torch.argmax(output.logits, dim=1).tolist()
    valid_loss /= valid_total
    valid_acc = valid_correct / valid_total
    valid_f1 = f1_score(y_true, y_pred)
    return valid_loss, valid_acc, valid_f1


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output.logits, label)
            test_loss += loss.item() * label
