import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
import time

# Load data
df = pd.read_csv("data.csv")

# Split data into train, valid, test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize data and create DataLoader
def tokenize_data(data):
    input_ids = []
    attention_masks = []
    labels = []
    tokenized = tokenizer.batch_encode_plus(
        data['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=512
    )
    inputs = {
        'input_ids': torch.tensor(tokenized['input_ids']),
        'attention_mask': torch.tensor(tokenized['attention_mask']),
        'labels': torch.tensor(data['label'].tolist())
    }
    dataset = TensorDataset(**inputs)
    return dataset

train_dataset = tokenize_data(train_df)
valid_dataset = tokenize_data(valid_df)
test_dataset = tokenize_data(test_df)

# Dataloader
batch_size = 16
train_dataloader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset),
    batch_size = batch_size
)
valid_dataloader = DataLoader(
    valid_dataset,
    sampler = SequentialSampler(valid_dataset),
    batch_size = batch_size
)
test_dataloader = DataLoader(
    test_dataset,
    sampler = SequentialSampler(test_dataset),
    batch_size = batch_size
)

# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * 10
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, total_steps=total_steps)

# Train and Validation functions
def train(model, train_dataloader, valid_dataloader, optimizer, device, num_epochs=10):
    best_f1 = 0
    best_model = None
    for epoch in range(num_epochs):
        start_time = time.time()
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        predictions, true_labels = [], []
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
            train_loss += loss.item() * label.size(0)
            train_correct += (torch.argmax(output.logits, dim=1) == label).sum().item()
            train_total += label.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total
        return train_loss, train_acc
