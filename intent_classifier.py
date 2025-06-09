#!/usr/bin/env python3

'''
Trains and evaluates GPT2IntentClassifier on Amazon MASSIVE dataset
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import csv

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
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                      batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids

# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                            batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids

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

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = GPT2IntentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")

        dev_data = load_intent_classification_data(split='validation')
        dev_dataset = IntentClassificationDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                  collate_fn=dev_dataset.collate_fn)

        test_data = load_intent_classification_data(split='test')
        test_dataset = IntentClassificationTestDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
        print('DONE DEV')

        test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
        print('DONE Test')

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
    parser.add_argument("--use_gpu", action='store_true', default=True)

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
        test_out='predictions/' + args.fine_tune_mode + '-intent-test-out.csv'
    )

    train(config)
    print('Evaluating on Intent Classification...')
    test(config) 