import torch
import torch.nn as nn
from typing import List, Optional
from .victim import Victim
from .plms import PLMVictim
from typing import Union
from .mlms import MLMVictim
from .llms import LLMVictim

Victim_List = {
    'plm': PLMVictim,
    'mlm': MLMVictim,
    'llm': LLMVictim
}


def load_victim(config) -> Union[PLMVictim, LLMVictim]:
    victim = Victim_List[config["type"]](**config)
    return victim

def mlm_to_seq_cls(mlm, config, save_path):
    mlm.plm.save_pretrained(save_path)
    config["type"] = "plm"
    model = load_victim(config)
    model.plm.from_pretrained(save_path)
    return model