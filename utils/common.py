import os
import json
import random
from tqdm import tqdm

import torch
import numpy as np


class Example(object):
    def __init__(self,
        idx, source, target,
    ):
        self.idx = idx
        self.source = source
        self.target = target


# Reads the data from a file and returns a list of examples.
def read_data(filename):
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f.readlines(), desc='----- [Reading]')):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js: js['idx'] = idx
            code=' '.join(js['code_tokens']).replace('\n', ' ')
            code=' '.join(code.strip().split())
            nl=' '.join(js['docstring_tokens']).replace('\n', '')
            nl=' '.join(nl.strip().split())
            examples.append(
                Example(idx=idx, source=code, target=nl) 
            )

    return examples


# Write a list of lines to a file.
def write_to_file(output_dir, lines, filename):
    if not os.path.exists(output_dir): os.makedirs(output_dir)      # 没有文件夹则创建
    with open(os.path.join(output_dir, filename), 'w') as f:
        for idx, line in enumerate(lines):
            f.write(str(idx) + '\t' + line + '\n')
        f.close()


# Saves the current checkpoint to the given output_dir.
def save_checkpoint(model, output_dir, desc):
    output_desc_dir = os.path.join(output_dir, desc)
    if not os.path.exists(output_desc_dir): os.makedirs(output_desc_dir)    # 没有文件夹则创建
    model_to_save = model.module if hasattr(model, 'module') else model     # Only save the model it-self
    output_model_file = os.path.join(output_desc_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


def set_seed(config):
    """set random seed."""
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
