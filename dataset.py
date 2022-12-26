import random
import math
import torch
from typing import List, Tuple
from transformers import PreTrainedTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right

class Dataset():
    def __init__(
        self,
        srcs: List[str],
        trgs: List[str],
        tokenizer: PreTrainedTokenizer,
        max_len: int
    ) -> None:
        self.srcs = srcs
        self.trgs = trgs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.orig_index = None
    
    def __getitem__(self, idx: int) -> dict:
        src = self.srcs[idx]
        trg = self.trgs[idx]
        src_encode = self.tokenizer.batch_encode_plus(
            [src],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        trg_encode = self.tokenizer.batch_encode_plus(
            [trg],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        labels = trg_encode['input_ids']
        decoder_input_ids = shift_tokens_right(
            labels,
            self.tokenizer.pad_token_id,
            2 
        )
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        return_dict = {
            'input_ids': src_encode['input_ids'].squeeze(),
            'attention_mask': src_encode['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_input_ids.squeeze(),
            'labels': labels
        }
        if self.orig_index is not None:
            return_dict['orig_index'] = self.orig_index[idx]
        return return_dict

    def __len__(self):
        return len(self.srcs)

    def sort_by_length(self):
        '''This is for prediction.
        '''
        orig_index = range(len(self.srcs))
        data = [
            (src, trg, orig_index) \
            for src, trg, orig_index in zip(self.srcs, self.trgs, orig_index)
        ]
        data = sorted(data, key=lambda x:len(x[0]))
        self.srcs = [x[0] for x in data]
        self.trgs = [x[1] for x in data]
        self.orig_index = [x[2] for x in data]
        

def generate_dataset(
    src_file: str,
    trg_file: str,
    tokenizer: PreTrainedTokenizer,
    max_len: int,
) -> Dataset:
    '''
    This function recieves input file path(s) and returns Dataset instance.
    '''
    srcs = open(src_file).read().rstrip().split('\n')
    trgs = open(trg_file).read().rstrip().split('\n')
    return Dataset(
        srcs=srcs,
        trgs=trgs,
        tokenizer=tokenizer,
        max_len=max_len
    )

    