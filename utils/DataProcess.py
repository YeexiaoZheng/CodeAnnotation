import os
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer

from Config import config
from utils import bleu
from utils.common import write_to_file

class InputFeatures(object):
    def __init__(self, 
        example_id,
        source_ids, source_mask,
        target_ids, target_mask,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.target_mask = target_mask     


class Processor:
    def __init__(self, config: config):
        self.config = config
        self.tokenizer = RobertaTokenizer.from_pretrained(config.model_path)

    # Encode examples and params and return a dataloader.
    def __call__(self, examples, params, stage=None):
        features = self.encode(examples, stage)
        dataset = self.to_dataset(features)
        return self.to_dataloader(dataset, params)

    # Encodes a list of examples into a list of InputFeatures.
    def encode(self, examples, stage=None):
        features = []
        for example_index, example in enumerate(tqdm(examples, desc='----- [Encoding]')):
            # source
            source_ids = torch.LongTensor(self.tokenizer.encode(example.source, 
                add_special_tokens=True, max_length=self.config.max_source_length, truncation=True))
            source_mask = torch.ones_like(source_ids)
    
            # target
            target = 'None' if stage == 'test' else example.target           
            target_ids = torch.LongTensor(self.tokenizer.encode(target, 
                add_special_tokens=True, max_length=self.config.max_target_length, truncation=True))
            target_mask = torch.ones_like(target_ids)
        
            features.append(
                InputFeatures(example_index, source_ids, source_mask, target_ids, target_mask)
            )
        return features
    
    # Decode a single record from the database.
    def decode_one(self, pred):
        return self.tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Decode a list of preds.
    def decode(self, preds):
        return [self.decode_one(pred) for pred in tqdm(preds, desc='----- [Decoding]')]

    # Converts a list of input features into a TensorDataset.
    def to_dataset(self, features: InputFeatures):
        all_source_ids = pad_sequence([f.source_ids for f in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        all_source_mask = pad_sequence([f.source_mask for f in features], batch_first=True, padding_value=0)
        all_target_ids = pad_sequence([f.target_ids for f in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        all_target_mask = pad_sequence([f.target_mask for f in features], batch_first=True, padding_value=0)
        return TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

    # Returns a dataloader for the given dataset.
    def to_dataloader(self, dataset, params):
        return DataLoader(dataset, **params)
    
    # Computes the BLEU metric for a set of trues and predictions.
    def metric(self, trues, preds, desc):
        write_to_file(self.config.output_dir, trues, (desc + '.gold'))
        write_to_file(self.config.output_dir, preds, (desc + '.output'))
        predictions = [str(idx) + '\t' + line for idx, line in enumerate(preds)]
        (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(self.config.output_dir, (desc + '.gold')))
        return round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
