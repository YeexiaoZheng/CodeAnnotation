from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.DataProcess import Processor
from Config import config

class Trainer:

    def __init__(self, config: config, processor: Processor, model, optimizer, scheduler, device):
        self.config = config
        self.processor = processor

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
    
    # Train the model on the train_loader.
    def train(self, train_loader: DataLoader):
        print('[Training info] Num examples: {}, Batch size: {}, Batch: {}'
        .format(len(train_loader.dataset), train_loader.batch_size, len(train_loader)))

        self.model.train()
        loss_list = []
        for batch in tqdm(train_loader, desc='----- [Training]'):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch

            if self.config.model_type.lower() == 'codet5':
                loss = self.model(input_ids=source_ids, attention_mask=source_mask.gt(0), 
                                  labels=target_ids, decoder_attention_mask=target_mask.gt(0)).loss
            else:
                loss, _, _  = self.model(source_ids=source_ids, source_mask=source_mask, 
                                         target_ids=target_ids, target_mask=target_mask)

            loss_list.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
    
        # Loss of train dataset
        train_loss = round(sum(loss_list) / len(loss_list), 5)

        return train_loss, loss_list

    # Validate the dataset and return ppl and bleu of dev dataset.
    def valid(self, val_loader: DataLoader):
        print('[Validing info] Num examples: {}, Batch size: {}, Batch: {}'
        .format(len(val_loader.dataset), val_loader.batch_size, len(val_loader)))

        self.model.eval()
        eval_loss, tokens_num = 0, 0
        true_ids, pred_ids = [], []
        for batch in tqdm(val_loader, desc='----- [Validing]'):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch

            with torch.no_grad():
                if self.config.model_type.lower() == 'codet5':
                    loss = self.model(input_ids=source_ids, attention_mask=source_mask, 
                                    labels=target_ids, decoder_attention_mask=target_mask).loss
                    eval_loss += loss.item()
                    tokens_num += 1
                    preds = self.model.generate(source_ids, attention_mask=source_mask, use_cache=True, 
                                        num_beams=self.config.beam_size, early_stopping=True, max_length=128)
                else:
                    _, loss, num = self.model(source_ids=source_ids,source_mask=source_mask,
                                          target_ids=target_ids,target_mask=target_mask)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                    preds = self.model(source_ids=source_ids, source_mask=source_mask)[:, 0]
                    self.processor.decode(preds)
                
                true_ids.extend(target_ids)
                pred_ids.extend(preds)

        # Metrics(ppl bleu) of dev dataset    
        eval_ppl = round(np.exp(eval_loss / tokens_num), 5)
        eval_bleu = self.processor.metric(self.processor.decode(true_ids), self.processor.decode(pred_ids), 'dev')

        return eval_ppl, eval_bleu

    # Predicts source_ids for each source in test_loader.
    def predict(self, test_loader: DataLoader):
        print('Predicting info] Num examples: {}, Batch size: {}, Batch: {}'
        .format(len(test_loader.dataset), test_loader.batch_size, len(test_loader)))

        self.model.eval()
        pred_ids = []
        for batch in tqdm(test_loader, desc='----- [Predicting]'):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, _, _ = batch
            with torch.no_grad():
                if self.config.model_type.lower() == 'codet5':
                    preds = self.model.generate(source_ids, attention_mask=source_mask, use_cache=True, 
                            num_beams=self.config.beam_size, early_stopping=True, max_length=128)
                else:
                    preds = self.model(source_ids=source_ids, source_mask=source_mask)[:, 0]
                
                pred_ids.extend(preds)

        return pred_ids