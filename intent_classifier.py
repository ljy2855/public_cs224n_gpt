#!/usr/bin/env python3

'''
Trains and evaluates GPT2IntentClassifier on Amazon MASSIVE dataset
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import csv
import json
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from tqdm import tqdm
from data_loader import load_intent_classification_data, IntentClassificationDataset, IntentClassificationTestDataset

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MetricsTracker:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create metrics directory: {e}")
        self.metrics = {
            'train_loss': [],
            'dev_loss': [],
            'train_acc': [],
            'train_f1': [],
            'dev_acc': [],
            'dev_f1': [],
            'train_f1_weighted': [],
            'dev_f1_weighted': [],
            'test_acc': None,
            'test_f1': None,
            'test_f1_weighted': None
        }
    
    def update(self, epoch, train_loss, train_acc, train_f1, dev_acc, dev_f1, train_f1_weighted, dev_f1_weighted, dev_loss):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['dev_loss'].append(dev_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['train_f1'].append(train_f1)
        self.metrics['dev_acc'].append(dev_acc)
        self.metrics['dev_f1'].append(dev_f1)
        self.metrics['train_f1_weighted'].append(train_f1_weighted)
        self.metrics['dev_f1_weighted'].append(dev_f1_weighted)
        
        # Save metrics to JSON
        try:
            with open(self.save_dir / 'metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metrics to JSON: {e}")
        
        # Plot metrics
        try:
            self.plot_metrics()
        except Exception as e:
            print(f"Warning: Could not plot metrics: {e}")
    
    def plot_metrics(self):
        # Create x-axis values (epoch numbers)
        epochs = list(range(len(self.metrics['train_loss'])))
        
        # Plot training and dev loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_loss'], label='Training Loss')
        plt.plot(epochs, self.metrics['dev_loss'], label='Dev Loss')
        plt.title('Training and Dev Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(epochs)  # Show all epoch numbers
        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_metrics.png')
        plt.close()
        
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_acc'], label='Train Accuracy')
        plt.plot(epochs, self.metrics['dev_acc'], label='Dev Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.xticks(epochs)  # Show all epoch numbers
        plt.tight_layout()
        plt.savefig(self.save_dir / 'accuracy_metrics.png')
        plt.close()
        
        # Plot F1 scores
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.metrics['train_f1'], label='Train F1 (Macro)')
        plt.plot(epochs, self.metrics['train_f1_weighted'], label='Train F1 (Weighted)')
        plt.plot(epochs, self.metrics['dev_f1'], label='Dev F1 (Macro)')
        plt.plot(epochs, self.metrics['dev_f1_weighted'], label='Dev F1 (Weighted)')
        plt.title('F1 Scores over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.xticks(epochs)  # Show all epoch numbers
        plt.tight_layout()
        plt.savefig(self.save_dir / 'f1_metrics.png')
        plt.close()

class GPT2IntentClassifier(torch.nn.Module):
    '''
    This module performs intent classification using GPT2 in a cloze-style (fill-in-the-blank) task.
    '''

    def __init__(self, config):
        super(GPT2IntentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.gpt = GPT2Model.from_pretrained()

        # Pretrain mode does not require updating GPT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.gpt.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        '''Takes a batch of sentences and returns logits for intent classes'''
        outputs = self.gpt(input_ids, attention_mask)
        last_token = outputs['last_token']
        last_token = self.dropout(last_token)
        logits = self.classifier(last_token)

        return logits

# Evaluate the model on dev examples.
def model_eval(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    texts = []
    sent_ids = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_labels, b_texts, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                          batch['labels'], batch['texts'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / b_ids.size(0)
            total_loss += loss.item()
            num_batches += 1
            
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()

            b_labels = b_labels.cpu().numpy().flatten()  # Move to CPU before converting to numpy
            y_true.extend(b_labels)
            y_pred.extend(preds)
            texts.extend(b_texts)
            sent_ids.extend(b_sent_ids)

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / num_batches

    return acc, f1_macro, f1_weighted, y_pred, y_true, texts, sent_ids, avg_loss

# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_pred = []
    texts = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_texts, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                            batch['texts'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        texts.extend(b_texts)
        sent_ids.extend(b_sent_ids)

    return y_pred, texts, sent_ids

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(args.metrics_dir)
    
    # Load intent classification data
    train_data = load_intent_classification_data(split='train')
    dev_data = load_intent_classification_data(split='validation')

    # Create datasets
    train_dataset = IntentClassificationDataset(train_data, args)
    dev_dataset = IntentClassificationDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': len(set([x[1] for x in train_data])),
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = GPT2IntentClassifier(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    best_dev_f1 = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                     batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, train_f1_weighted, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, dev_f1_weighted, _, _, _, _, dev_loss = model_eval(dev_dataloader, model, device)
        

        # Update metrics tracker
        metrics_tracker.update(epoch, train_loss, train_acc, train_f1, dev_acc, dev_f1, train_f1_weighted, dev_f1_weighted, dev_loss)

        if dev_acc > best_dev_acc or dev_f1 > best_dev_f1:
            best_dev_acc = dev_acc
            best_dev_f1 = dev_f1
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, train f1 :: {train_f1 :.3f}, train weighted f1 :: {train_f1_weighted :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}, dev weighted f1 :: {dev_f1_weighted :.3f}, dev loss :: {dev_loss :.3f}")

def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = GPT2IntentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")

        # Initialize metrics tracker
        metrics_tracker = MetricsTracker(args.metrics_dir)

        dev_data = load_intent_classification_data(split='validation')
        dev_dataset = IntentClassificationDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                  collate_fn=dev_dataset.collate_fn)

        test_data = load_intent_classification_data(split='test')
        test_dataset = IntentClassificationTestDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=test_dataset.collate_fn)

        # Evaluate on dev set
        dev_acc, dev_f1, dev_f1_weighted, dev_pred, dev_true, dev_texts, dev_sent_ids, dev_loss = model_eval(dev_dataloader, model, device)
        print('DONE DEV')

        # Evaluate on test set
        test_pred, test_texts, test_sent_ids = model_test_eval(test_dataloader, model, device)
        print('DONE Test')

        # Save predictions
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Intent \n")
            for p, s in zip(dev_sent_ids, dev_pred):
                f.write(f"{p}, {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Intent \n")
            for p, s in zip(test_sent_ids, test_pred):
                f.write(f"{p}, {s} \n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true', default=False)
    parser.add_argument("--metrics_dir", type=str, default="report",
                      help="Directory to save training metrics and plots")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-3)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    print('Training Intent Classifier on Amazon MASSIVE...')
    config = SimpleNamespace(
        filepath='intent-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        fine_tune_mode=args.fine_tune_mode,
        dev_out='predictions/' + args.fine_tune_mode + '-intent-dev-out.csv',
        test_out='predictions/' + args.fine_tune_mode + '-intent-test-out.csv',
        metrics_dir=args.metrics_dir + '/' + args.fine_tune_mode
    )

    train(config)
    print('Evaluating on Intent Classification...')
    test(config) 