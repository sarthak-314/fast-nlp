from torch.utils.data import Dataset, DataLoader
from torch import nn

import accelerator


def get_optimizer_grouped_params(model, weight_decay):
    no_decay = ["bias", "LayerNorm.bias"]
    return [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.00,
        },
    ]