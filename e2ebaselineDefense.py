# Defend
import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
from torch import cuda
import re
import platform 

import torch.cuda as cuda
def selectMaxReservedGPU():
    def testGPU(id=0, mem_collect='auto'): 
        GPU_state = os.popen(f'gpustat --id {id}').read()

        matches = re.findall(r'(\d+) / (\d+) MB', GPU_state)
        if matches:
            Usage, Memory = matches[-1]
            IDLE =  int(Memory) - int(Usage) 
            if mem_collect == 'auto': 
                if int(Usage) > -1:
                    return True, IDLE, int(Memory)
                else:
                    return False, IDLE, int(Memory)
            else:
                if IDLE > mem_collect:
                    return True, IDLE, int(Memory)
                else:
                    return False, IDLE, int(Memory)
        else:
            return False, -1, -1 
    memRese = []
    for i in range(4):
        vali, rese, _ = testGPU(id=i)
        if vali:
            memRese.append((i, rese))
            
    sortedMemRese = sorted(memRese, key=lambda x:x[1], reverse=True)
    
    return sortedMemRese[0]

GPUId, memReserve = selectMaxReservedGPU()
print(f'GPU ID = {GPUId}, memory reserved = {memReserve}')
cuda.set_device(GPUId)
# load the maximum memory reserved device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./baselineConfigs/Onion.json')
    parser.add_argument('--poisoner', type=str, default='badnets4', help=['badnets4', 'addsents', 'style', 'hiddenkiller'])
    parser.add_argument('--target_model', type=str, default='bert')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def main(config):
    # choose a victim classification model 
    victim = load_victim(config["victim"])
    config["defender"]["device"] = victim.device
    # choose attacker and initialize it with default parameters 
    attacker = load_attacker(config["attacker"])
    defender = load_defender(config["defender"])
    # choose target and poison dataset
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"]) 
    # target_dataset = attacker.poison(victim, target_dataset)
    # launch attacks 
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config, defender)
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset, defender)
    
    display_results(config, results)
    
    # Fine-tune on clean dataset
    '''
    print("Fine-tune model on {}".format(config["target_dataset"]["name"]))
    CleanTrainer = ob.BaseTrainer(config["train"])
    backdoored_model = CleanTrainer.train(backdoored_model, wrap_dataset(target_dataset, config["train"]["batch_size"]))
    '''

if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    if args.target_model is not None:
        models = {
            'bert':"bert-base-uncased", 
            'roberta':"roberta-base", 
            'bert-large':"bert-large-uncased", 
            'roberta-large':"roberta-large", 
            'llama':"llama-2-7b-hf",
            'gpt':"gpt2-xl"
        }
        config["victim"]["path"] = models[args.target_model]
        config["victim"]["model"] = args.target_model.lower().split('-')[0]
        
    if args.dataset is not None:
        config["target_dataset"]["name"] = args.dataset
        config["poison_dataset"]["name"] = args.dataset
        if args.dataset.lower() in ['agnews', 'miniagnews']:
            config["victim"]["num_classes"] = 4
            if config["defender"]["name"] in ['cube', 'bki']:
                config["defender"]["num_classes"] = 4
        if args.dataset.lower() == 'lingspam' and config["attacker"]["poisoner"]["name"] in ["synbkd", "stylebkd"]:
            config["attacker"]["poisoner"]["longSent"] = True
            print("lingSpam belongs to Long Sentence")
    
    poisoners = {
        'badnets4':{
            "name": "badnets",
            "poison_rate": 0.1,
            "target_label": 1,
            "label_consistency": True,
            "label_dirty": False,
            "triggers": ["cf", "mn", "bb", "tq"],
            "num_triggers": 4,
            "load": True
        },
        'addsents':{
            "name": "addsent",
            "poison_rate": 0.1,
            "target_label": 1,
            "label_consistency": False,
            "label_dirty": False,
            "load": True,
            "triggers": "I watch this 3D movie"
        },
        'style':{
            "name": "stylebkd",
            "poison_rate": 0.1,
            "target_label": 1,
            "label_consistency": False,
            "label_dirty": False,
            "load": True,
            "template_id": 0
        },
        'hiddenkiller':{
            "name": "synbkd",
            "poison_rate": 0.1,
            "target_label": 1,
            "label_consistency": False,
            "label_dirty": False,
            "load": True,
            "template_id": -1
        }
    }        
    
    config["attacker"]["poisoner"] = poisoners[args.poisoner]
    if config["defender"]["name"] in ['cube', 'bki']:
        config["defender"]["model_path"] = config["victim"]["path"]
        config["defender"]["model_name"] = args.target_model.lower()
    config = set_config(config)
    set_seed(args.seed)
    print(json.dumps(config, indent=4))
    main(config)
